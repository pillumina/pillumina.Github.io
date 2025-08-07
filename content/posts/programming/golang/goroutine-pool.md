---
title: "[自建轮]高性能Goroutine Pool"
date: 2020-12-30T11:22:18+08:00
hero: /images/posts/golang2.jpg
menu:
  sidebar:
    name: Goroutine Pool
    identifier: goroutine-pool
    parent: golang odyssey
    weight: 10
draft: false
---

# 高性能Goroutine Pool

go调度器没有限制对goroutine的数量，在goroutine瞬时大规模爆发的场景下来不及复用goroutine从而导致大量goroutine被创建，会导致大量的系统资源占用，尝试池化。

`go调度器本身不应该对goroutine数量有限制，因为语言层面无法界定需要限制多少，毕竟程序跑在不同性能的环境，在并发规模不太大的场景做限制甚至会降低性能，原生支持限制goroutine数量无疑是得不偿失的。如果只是中等规模和比较小规模的并发场景其实pool的性能并没有优势`

目前设计上还需要加上周期性对空闲队列的prune，等写完再加看看benchmark会提升多少。目前来说对大规模goroutine异步并发的场景(1M, 10M)内存优化(10倍往上)和吞吐量优化效果(2-6倍)非常好。

## 需求场景与目标

1. 限制并发goroutine的数量
2. 复用goroutine，减轻runtime调度压力，提升程序性能
3. 规避过多的goroutine创建侵占系统资源，cpu&内存



## 关键技术

1. 锁同步: golang有CAS机制，用spin-lock替代mutex [原理](https://ofstack.com/Golang/27085/implementation-of-golang-spin-lock.html)， [讨论](https://stackoverflow.com/questions/5869825/when-should-one-use-a-spinlock-instead-of-mutex)
2. LIFO/FIFO队列: LIFO队列能直接有时间排序功能，方便对需要关联入队时间的操作进行处理
3. Pool容量限制和弹性伸缩

## 代码实现

### pool.go

```go
package go_pool

import (
	"errors"
	"sync"
	"sync/atomic"
	"time"
)

const(
	OPEN = iota
	CLOSED
)

var (
	ErrPoolClosed = errors.New("this pool has been closed")
	ErrPoolOverload = errors.New("too many goroutines blocked on submit or Nonblocking is set")
	ErrInvalidExpiryTime = errors.New("invalid expiration time")
	ErrInvalidPoolCapacity = errors.New("invalid pool capacity")
	DefaultScanInterval = time.Second
)

type Pool struct {
	capacity int32

	running int32

	lock sync.Locker

	scanDuration time.Duration

	blockingTasksNum int

	maxBlockingTasks int

	state int32

	cond *sync.Cond

	workers WorkerQueue   // LIFO queue

	workerCache sync.Pool

}


func (p *Pool) Submit(task func()) error{
	if atomic.LoadInt32(&p.state) == CLOSED{
		return ErrPoolClosed
	}
	// retrieve worker to do the task
	// return error if no workers available
	var w *Worker
	if w = p.retrieveWorker(); w == nil{
		return ErrPoolOverload
	}
	w.task <- task
	return nil
}

func (p *Pool) Shutdown() {
	atomic.StoreInt32(&p.state, CLOSED)
	p.lock.Lock()
	// reset worker queue
	p.workers.reset()
	p.lock.Unlock()
}

func (p *Pool) isClosed() bool{
	return atomic.LoadInt32(&p.state) == CLOSED
}

// change the capacity of the pool
func (p *Pool) Resize(size int){
	if p.Cap() == size{
		return
	}
	atomic.StoreInt32(&p.capacity, int32(size))
	// need to stop certain workers if #running_workers > #new_capacity
	diff := p.Running() - size
	if diff > 0{
		for i := 0; i< diff; i++{
			p.retrieveWorker().task <- nil
		}
	}
}

func (p *Pool) Reboot() {
	if atomic.CompareAndSwapInt32(&p.state, CLOSED, OPEN){
		// initialize the purging go routine
		go p.scavengerRoutine()
	}
}

func (p *Pool) Running() int{
	return int(atomic.LoadInt32(&p.running))
}

func (p *Pool) Cap() int{
	return int(atomic.LoadInt32(&p.capacity))
}

func (p *Pool) Free() int{
	return p.Cap() - p.Running()
}

func (p *Pool) incRunning(){
	atomic.AddInt32(&p.running, 1)
}

func (p *Pool) decRunning(){
	atomic.AddInt32(&p.running, -1)
}

// put the worker back into the pool for recycling
func (p *Pool) recycleWorker(worker *Worker) bool{
	capacity := p.Cap()
	if p.isClosed() || (capacity >= 0 && p.Running() > capacity){
		return false
	}
	worker.recycleTime = time.Now()
	p.lock.Lock()
	// need to double check if state is CLOSED
	if p.isClosed(){
		p.lock.Unlock()
		return false
	}
	err := p.workers.add(worker)
	if err != nil{
		p.lock.Unlock()
		return false
	}

	// notify any request stuck in retrieveWorker that there is an available worker in pool
	p.cond.Signal()
	p.lock.Unlock()
	return true
}

func (p *Pool) spawnWorker() *Worker{
	worker := p.workerCache.Get().(*Worker)
	worker.Run()
	return worker
}

func (p *Pool) retrieveWorker() (worker *Worker){
	p.lock.Lock()
	worker = p.workers.detach()
	// get worker from queue successfully
	if worker != nil{
		p.lock.Unlock()
	}else if capacity := p.Cap();capacity == -1{
		p.lock.Unlock()
		// spawn worker
		return p.spawnWorker()
	}else if p.Running() < capacity{
		// infinite pool
		p.lock.Unlock()
		// spawn worker
		return p.spawnWorker()
	}else{
		// if the number of blocking tasks reaches the maximum blocking tasks threshold then returns nil
		// and throw the ErrPoolOverload error in Submit method
		if p.maxBlockingTasks != 0 && p.maxBlockingTasks <= p.blockingTasksNum{
			p.lock.Unlock()
			return
		}
		// the pool is full need to wait until worker is available for task handling
		Retry:
			// handle the number of blocking task handling requests
			// wait until condition being notified
			p.blockingTasksNum++
			p.cond.Wait()
			p.blockingTasksNum--
			// ensure there is a worker available because you don't know if the recycled worker being closed then
			if p.Running() == 0{
				p.lock.Unlock()
				// spawn worker
				return p.spawnWorker()
			}

			worker = p.workers.detach()
			if worker == nil{
				goto Retry
			}
			p.lock.Unlock()
	}

	return
}

func (p *Pool) scavengerRoutine(){
	heartbeat := time.NewTicker(p.scanDuration)
	defer heartbeat.Stop()
	for range heartbeat.C{
		if p.isClosed(){
			break
		}
		// all workers get cleaned up and some invokers still get stuck on cond.Wait()
		// we need to wake up all invokers in that situation.
		if p.Running() == 0{
			p.cond.Broadcast()
		}
	}
}

func NewPool(capacity int)(*Pool, error){
	if capacity <= 0{
		capacity = -1
	}

	pool := &Pool{
		capacity:  int32(capacity),
		lock: NewSpinLock(),
	}
	pool.workerCache.New = func() interface{}{
		return &Worker{
			pool: pool,
			task: make(chan func(), 1),
		}
	}
	pool.scanDuration = DefaultScanInterval
	// initialize the worker queue
	if capacity == -1{
		return nil, ErrInvalidPoolCapacity
	}
	pool.workers = NewWorkerQueue(0)

	pool.cond = sync.NewCond(pool.lock)

	// initialize the purging goroutine
	go pool.scavengerRoutine()

	return pool, nil
}


```

### worker.go

```go
package go_pool

import (
	"time"
)

type Worker struct{
	pool *Pool

	task chan func()
	
	recycleTime time.Time
}

func (w *Worker) Run(){
	w.pool.incRunning()
	go func(){
		defer func(){
			w.pool.decRunning()
			w.pool.workerCache.Put(w)
			// todo: panic recovery strategy
		}()
		for f := range w.task{
			// receiving nil indicates that the worker should stop and quit go routine
			if f == nil{
				return
			}
			f()
			// recycle worker back into the pool, if not success quit go routine
			if success := w.pool.recycleWorker(w); !success{
				return
			}
		}
	}()
}

```



### worker_queue.go

```go
package go_pool

type WorkerQueue interface {
	len() int
	isEmpty() bool
	add(worker *Worker) error
	detach() *Worker
	reset()
}

func NewWorkerQueue(size int) WorkerQueue{
	return NewSimpleWorkerQueue(size)
}

func NewSimpleWorkerQueue(size int) *simpleWorkerQueue{
	return &simpleWorkerQueue{
		size: size,
		workers: make([]*Worker, 0, size),
	}
}


type simpleWorkerQueue struct{
	workers []*Worker
	size int
}

func(sq *simpleWorkerQueue) len() int{
	return len(sq.workers)
}

func(sq *simpleWorkerQueue) isEmpty() bool{
	return sq.len() == 0
}

func (sq *simpleWorkerQueue) add(worker *Worker) error{
	sq.workers = append(sq.workers, worker)
	return nil
}

func (sq *simpleWorkerQueue) detach() *Worker{
	length := sq.len()
	if length == 0{
		return nil
	}
	worker := sq.workers[length - 1]
	sq.workers[length - 1] = nil // slice operation should avoid memory leak
	sq.workers = sq.workers[:length-1]
	return worker
}

func (sq *simpleWorkerQueue) reset(){
	for i := 0;i < sq.len(); i++{
		sq.workers[i].task <- nil
		sq.workers[i] = nil
	}
	sq.workers = sq.workers[:0]
}
```



### lock.go

```go
package go_pool

import (
	"runtime"
	"sync"
	"sync/atomic"
)

type spinLock uint32

func (sl *spinLock) Lock() {
	for !atomic.CompareAndSwapUint32((*uint32)(sl), 0, 1) {
		runtime.Gosched()
	}
}

func (sl *spinLock) Unlock() {
	atomic.StoreUint32((*uint32)(sl), 0)
}

// NewSpinLock instantiates a spin-lock.
func NewSpinLock() sync.Locker {
	return new(spinLock)
}
```



### pool_test.go

```go
package go_pool

import (
	"math"
	"runtime"
	"sync"
	"testing"
	"time"
)

const(
	_ = 1 << (10 * iota)
	KiB //1024
	MiB // 1048578
)

const (
	InfinitePoolSize = math.MaxInt32
	PoolSize        = 10000
	SleepTime       = 100
	OverSizeTaskNum = 10 * PoolSize
	UnderSizeTaskNum = 0.2 * PoolSize
)
var currentMem uint64

func demoTaskFunc(args interface{}){
	n := args.(int)
	time.Sleep(time.Duration(n) * time.Millisecond)
}


func TestPoolWaitToGetWorker(t *testing.T){
	var wg sync.WaitGroup
	p, err := NewPool(PoolSize)
	defer p.Shutdown()
	if err != nil {
		t.Errorf("err: %s", err.Error())
	}
	for i:=0; i< OverSizeTaskNum; i++{
		wg.Add(1)
		_ = p.Submit(func(){
			demoTaskFunc(SleepTime)
			wg.Done()
		})
	}
	wg.Wait()
	mem := runtime.MemStats{}
	runtime.ReadMemStats(&mem)
	currentMem = mem.TotalAlloc/KiB - currentMem
	t.Logf("memory usage: %d KB", currentMem)
}

func TestPoolGetWorkerFromCache(t *testing.T){
	var currentMem uint64
	var wg sync.WaitGroup
	p, err := NewPool(PoolSize)
	defer p.Shutdown()
	if err != nil {
		t.Errorf("err: %s", err.Error())
	}
	for i:=0; i< UnderSizeTaskNum; i++{
		wg.Add(1)
		_ = p.Submit(func(){
			demoTaskFunc(SleepTime)
			wg.Done()
		})
	}
	wg.Wait()
	mem := runtime.MemStats{}
	runtime.ReadMemStats(&mem)
	currentMem = mem.TotalAlloc/KiB - currentMem
	t.Logf("memory usage: %d KB", currentMem)
}

func TestNoPool(t *testing.T){
	var wg sync.WaitGroup
	for i:=0; i<UnderSizeTaskNum; i++{
		wg.Add(1)
		go func(){
			defer wg.Done()
			demoTaskFunc(SleepTime)
		}()
	}
	wg.Wait()
	mem := runtime.MemStats{}
	runtime.ReadMemStats(&mem)
	currentMem = mem.TotalAlloc/KiB - currentMem
	t.Logf("memory usage: %d KB", currentMem)
}

func TestWithInfinitePool(t *testing.T){
	var wg sync.WaitGroup
	p, err := NewPool(InfinitePoolSize)
	defer p.Shutdown()
	if err != nil {
		t.Errorf("err: %s", err.Error())
	}
	for i:=0; i< UnderSizeTaskNum; i++{
		wg.Add(1)
		_ = p.Submit(func(){
			demoTaskFunc(SleepTime)
			wg.Done()
		})
	}
	wg.Wait()
	mem := runtime.MemStats{}
	runtime.ReadMemStats(&mem)
	currentMem = mem.TotalAlloc/KiB - currentMem
	t.Logf("memory usage: %d KB", currentMem)
}
```



### pool_benchmark_test.go

```go
package go_pool

import (
	"testing"
	"time"
)

const (
	RunTimes = 5000000
	BenchParam = 10
	BenchPoolSize = 200000
)

func demoFunc() {
	time.Sleep(time.Duration(BenchParam) * time.Millisecond)
}

func BenchmarkPoolThroughput(b *testing.B) {
	p, _ := NewPool(BenchPoolSize)
	defer p.Shutdown()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < RunTimes; j++ {
			_ = p.Submit(demoFunc)
		}
	}
	b.StopTimer()
}

func BenchmarkGoroutinesThroughput(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < RunTimes; j++ {
			go demoFunc()
		}
	}
}
```



