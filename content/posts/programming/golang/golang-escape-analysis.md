---
title: "Golang逃逸分析"
date: 2020-11-23T11:22:18+08:00
hero: /images/posts/golang2.jpg
menu:
  sidebar:
    name: Golang逃逸分析
    identifier: golang-escape-analysis
    parent: golang odyssey
    weight: 10
draft: false
---

***问题： golang函数传参是不是应该和c一样，尽量不要直接传结构体，而是要传结构体指针？***

## 逃逸分析

逃逸分析指的是，在计算机语言编译器优化原理中，分析指针动态范围的方法，和编译器优化原理的指针分析和外形分析相关联。当变量（或者对象）在方法中被分配后，其指针有可能被返回或者被全局引用，这种现象就是指针（或引用）的逃逸（Escape）。

其实在java概念中有一个误解 --- new出来的东西都在堆上，栈上存的是它的引用。 这句话在现代JVM上有问题，就是因为逃逸分析机制。简单来说，就是JVM的逃逸分析会在运行时(runtime)检测当前方法栈帧(frame)内new出来的对象的引用，是否被传出当前的栈帧。如果传出，就会发生逃逸，没有传出则不会。对于未发生逃逸的变量，则会直接在栈上分配内存。因为栈上内存由在函数返回时自动回收，而堆上的的内存需要gc去回收，如果程序中有大量逃逸的对象，那么势必会增加gc的压力。

```java
public void test(){
  List<Integer> a = new ArrayList<>();
  a.add(1); // a 未逃逸，在栈上分配
}

public List<Integer> test1(){
  List<Integer> a = new ArrayList<>();
  a.add(1);
  return a // 发生逃逸，因此分配在堆上
}
```



## 区别

- 不同于JVM运行时的逃逸分析，Golang的逃逸分析是在编译期完成。
- golang的逃逸分析只针对指针。一个值引用变量如果没有被取址，那么它永远不可能逃逸。

```
go version go1.13.4 darwin/amd64
```

验证某个函数的变量是否发生逃逸的方法：

- go run -gcflags "-m -l" (-m打印逃逸分析信息，-l禁止内联编译)

- go tool compile -S xxxx.go | grep runtime.newobject（汇编代码中搜newobject指令，这个指令用于生成堆对象）



备注： 关于-gcflags "-m -l"的输出，有两种情况：

- Moved to heap: xxx
- xxx escapes to heap

二者都表示发生了逃逸，当xxx变量为指针的时候，出现第二种；当xxx变量为值类型时，为上一种，测试代码：

```go
type S int
func main(){
  a := S(0)
  b := make([]*S, 2)
  b[0] = &a
  c := new(S)
  b[1] = c
}
```



## Golang逃逸分析

本文探究什么时候，什么情况下会发生逃逸

### case 1

最基本的情况

```
在某个函数中new或者字面量创建出的变量，将其指针作为函数返回值，则该变量一定发生逃逸
```

下面是例子:

```go
func test() *User{
  a := User{}
  return &a
}
```



### case 2

需要验证文章开头情况的正确性，也就是当某个值取指针并传给另一个函数的时候，是否有逃逸：

```go
type User struct{
  Username string
  Password string
  Age	int
}

func main(){
  a := "aaa"
  u := &User{a, "123", 12}
  Call1(u)
}

func Call1(u *User){
  fmt.Printf("%v", u)
}
```

逃逸情况:

```
-> go run -gcflags "-m -l" main.go
# command-line-arguments
./main.go:18:12: leaking param: u
./main.go:19:12: Call1... argument does bnot escape
./main.go:19:13 u escapes to heap
./main.go:14:23 &User literal escapes to heap
```

可见发生了逃逸，这里将指针传给一个函数并打印，如果不打印，只对u进行读写：

```go
func Call1(u *User) int{
  u.Username = "bbb"
  return u.Age * 20
}
```

结果:

```
-> go run -gcflags "-m -l" main.go
# command-line-arguments
./main.go:19:12: Call1 u does not escape
./main.go:14:23 main &User literal does not escape
```

并没有发生逃逸。其实如果只是对u进行读写，不管调用几次函数，传了几次指针，都不会逃逸。所以我们可以怀疑fmt.Printf的源码有问题，可以发现传入的u被赋值给了pp指针的一个成员变量

```go
// Printf formats according to a format specifier and writes to standard output.
// It returns the number of bytes written and any write error encountered.
func Printf(format string, a ...interface{}) (n int, err error) {
	return Fprintf(os.Stdout, format, a...)
}

// Fprintf formats according to a format specifier and writes to w.
// It returns the number of bytes written and any write error encountered.
func Fprintf(w io.Writer, format string, a ...interface{}) (n int, err error) {
	p := newPrinter()
	p.doPrintf(format, a)
	n, err = w.Write(p.buf)
	p.free()
	return
}

// doPrintf里有
// ....
p.printArg(a[argNum], rune(c))
// ....

func (p *pp) printArg(arg interface{}, verb rune) {
	p.arg = arg
	p.value = reflect.Value{}
  // ....
}
```

这个pp类型的指针p是由构造函数newPrinter返回，根据case1，p一定会发生逃逸，而p引用了传入指针，所以我们可以总结：

```
被已经逃逸的变量引用的指针，一定发生逃逸。
```



### case3

上述备注代码的例子：

```go
func main(){
  a  := make([]*int, 1)
  b := 12
  a[0] = &b
}
```

实际上这个代码中, slice a不会逃逸，而被a引用的b会逃逸。类似的情况会发生在map和chan之中

```go
func main(){
  a := make([]*int, 1)
  b := 12
  a[0] = &b
  
  c := make(map[string]*int)
  d := 14
  c["aaa"] = &d
  
  e := make(chan *int, 1)
  f := 15
  e <- &f
}
```

结果可以发现, b, d, f都逃逸了。所以我们可以得出结论：

```
被指针类型的slice, map和chan引用的指针一定会发生逃逸。
备注： stack overflow上有人提问为何使用指针的chan比使用值得chan慢%30， 答案就在这里。使用指针的chan发生逃逸，gc拖慢了速度。
```



## 总结与深入本质

```
变量的逃逸，本质由于对于stack栈帧的内存分配，对于函数的调用将开辟一个栈帧frame，在这个栈帧内定义局部变量，当传出栈帧内创建的变量引用到前一个栈帧离去，如果函数结束，那么原来这块栈帧有可能被其他覆盖，这个传出去的引用就有问题。所以编译器把这种函数返回的变量可能在后续被引用的情况，将变量逃逸到堆上是一个非常合理的策略。
GopherCon SG 2019
1. When a value could possibly be reference after the function that constructed the value returns.
2. When the compiler determines a value is too large to fit on the stack.
3. When the compiler doesn't know the size of a value at compile time.
```



我们得出指针**必然逃逸**的情况：

- 在某个函数中new或者字面量创建出的变量，将其指针作为函数返回，则该变量一定发生逃逸（构造函数返回的指针变量一定逃逸）
- 被已经逃逸的变量引用的指针，一定发生逃逸
- 被指针类型slice, map和chan引用的指针，一定发生逃逸

同时我们也得出一些**必然不会逃逸**的情况：

- 指针被未发生逃逸的变量引用
- 仅仅在函数内对变量做取址操作，而未将指针传出

有些情况**可能发生逃逸，也可能不会发生逃逸** ：

- 将指针作为入参传给别的函数，这里还是要看指针在被传入的函数中的处理过程，如果发生了上述三种情况，则会逃逸；否则不会发生逃逸。

***因此，对于文章开头的问题，我们不能仅仅依据使用值引用作为函数入参可能因为copy导致额外内存开销而放弃这种值引用类型入参的写法。因为如果函数内有造成变量逃逸的操作情形，gc可能会成为程序效率不高的瓶颈。***

## 对io.Reader的解释

```go
type Reader struct{
  Read(p []byte) (n int, err error)
}

// Instead of 
type Reader struct{
  Read(n int) (b []byte, err error)
}
```

对于一个Reader来说当然第二种写法更为贴近逻辑，但是根据逃逸分析，第二种写法明显在不断的Read时在堆上产生过多的垃圾。

```go
// escape to heap
func main(){
  b := read()
  // use b
}

func read() []byte{
  // return a new slice
  b := make([]byte, 32)
  return b
}

// stay on stack
func main(){
  b := make([]byte, 32)
  read(b)
  // use b
}

func read(b []byte){
  // write into slice
}
```

## 几点强调

- Optimize for correctness, not performance.
- Go only puts function variables on the stakc if it can prove a variable is not used after the function returned.
- Sharing down typically stay on the stack (传递指针给函数)
- Sharing up typically escapes to the heap (返回指针，不过不必须，都加了typically，比如内联可能会让情形不太一样)
- Ask the compiler to find out

## 深入逃逸和内联

### 逃逸的深入解释

​    前面尝试了几个例子去分析逃逸的场景，实际上我们还是需要理解其内部机制，才能把收益最大化（开发效率v.s.运行效率）。逃逸分析的本质是当compiler发现函数变量将脱离函数栈的有效域或被函数栈域外的变量所引用时，把变量分配在堆上而不是栈上，分析一些典型的场景：

- 上述讨论过的，函数返回变量地址，或者返回包含变量地址的结构体。
- 把变量地址写入channel或者sync.Pool，compiler无法获取goroutine如何使用这个变量，也就无法在编译的时候决定变量的生命周期。
- 闭包可能导致闭包上下文逃逸，
- slice变量超过cap重新分配时，将在堆上进行，栈的大小毕竟是固定和有限的。
- 上述讨论过的把变量地址赋值给可扩容容器（map, slice）时。
- 把变量赋给可扩容interface容器（k或v为interface的map，或[]interface）的时候。
- 几乎涉及到interface的地方都有可能导致对象逃逸，MyInterface(x).Foo(a)会导致a逃逸，如果a是引用语义(pointer, slice, map etc.)，那么a也会分配到堆上。涉及到interface的很多逃逸优化都很保守，比如reflect.ValueOf(x)会显式调用escapes(x)导致x逃逸。

   我们分析一下slice重分配的场景。这个场景是在堆上发生的，因为slice重分配时，会发生数据迁移，此时会把原本slice len内的元素**浅拷贝**到新的space。这个浅拷贝会导致新的slice(堆内存)引用了p(栈内存)的内容，而栈内存和堆内存的生命周期不一样，导致了可能出现函数return了以后，堆内存引用无效的栈内存的情形，这无疑会影响到运行的稳定。所以即使slice变量本身没有显式得逃逸，由于隐式的数据迁移，compiler会保守把slice或者map的指针元素逃逸到堆上。

  对于interface相关的，interface{}把值语义变为引用语义，其本质是type+pointer，这个pointer指向实际的data (源码分析开坑)。如果把值语义的变量赋值给interface容器，那么容器会持有变量的引用，所以这个变量会逃逸到堆上分配。

  案例里也分析了，fmt.Printf会导致逃逸，其实fmt.Sprintf或者logrus.Debugf都会导致所有传入参数逃逸，因为不定参数实际上是slice语法糖，编译器无法确定这些函数不会对参数slice进行append操作导致重分配，所以基于保守策略，都会把这些传入的参数分配到堆上以保证浅拷贝是准确的。

  这里我评价golang编译器的逃逸策略为保守应该是比较合适的，好的逃逸分析需要在编译期更深入地理解程序，这无疑非常困难，特别是涉及到interface{}，指针，可扩展容器的时候。



### 内联

  关于内联我需要在另一篇post中深入讨论，这里简单地说些感受。逃逸分析+GC很好用但是如果没有内联就会显得很昂贵，所有函数返回的地方会有一道“墙”，任何想要从墙逃逸到墙外的变量都会分配到堆上，比如：

```go
func NewCoord() *Coord{
  return &Coord{
    x : 1,
    z : 2,
  }
}

func foo(){
  c := NewCoord()
  return c.x
}
```

  像NewCoord这样简单的构造函数都会导致返回值分配在堆上，抽离函数的代价也会更大。所以Go的内联，逃逸分析，GC像是三剑客，共同把其他语言避之不及的指针变得cheap。

  Go1.9开始对内联做了比较大的runtime优化，开始支持[mid-stack inline](https://go.googlesource.com/proposal/+/master/design/19348-midstack-inlining.md) ，并且通过`-l`编译参数指定内联等级([参数定义](https://golang.org/src/cmd/compile/internal/gc/inl.go))。并且只在`-l=4`中提供了mid-stack inline，Go官方统计，这大概可以提升9%的性能，不过也增加了11%左右的二进制大小。

  Go1.10做一些interface相关的优化，比如[devirtualization](https://github.com/golang/go/issues/19361) , compiler能够知道interface具体对象的情况下(如`var i Iface = &myStruct{}`)可以直接生成对象相关代码调用(而非内联)，无需走interface方法查找。不过目前这个优化还不完善，还不能应用于逃逸分析优化。

  Go1.12开始默认支持了mid-stack inline

  在目前的项目中，似乎还不需要去调整内联参数，因为这个操作是个trade-off，过于激进的内联会导致生成的二进制文件更大你，CPU intstruction cache miss也可能会增加。默认等级的内联大部分时候都工作得很好并且保持稳定，到Go1.13为止，对interface方法的调用还不能被内联（哪怕compiler知道其具体的类型）。

  ```go
type I interface {
	F() int
}
type A struct{
	x int
	y int
}
func (a *A) F() int {
	z := a.x + a.y
	return z
}
func BenchmarkX(b *testing.B) {
	b.ReportAllocs()
	for i:=0; i<b.N; i++ {
		// F() 会被内联 0.36 ns/op
		// var a = &A{}
		// a.F()
		// 对Interface的方法调用不能被内联 18.4 ns/op
		var i I = &A{}
		i.F()
	}
}
  ```

  对于一些偏底层基础的结构体，像上述的外层抽象了接口interface用于提供简单的对字段的访问设置，按照目前的分析和测试，内联会把字段访问速度提升一个数量级。

  **PS： 个人的感受是目前Go interfaced的内联做的不够好，或许可以用公共API返回具体类型而不是interface，比如etcdclient.New, grpc.NewServer这些都是这样实践的，它们通过private fields加public methods让外部用起来像interface一样，但是数据逻辑层可能实践起来比较麻烦，因为Go的访问控制太差。**