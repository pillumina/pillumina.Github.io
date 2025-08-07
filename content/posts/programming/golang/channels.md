---
title: "Channels Concurrency Work-Around"
date: 2020-12-22T11:22:18+08:00
hero: /images/posts/golang2.jpg
menu:
  sidebar:
    name: Channels Concurrency
    identifier: channels-concurrency
    parent: golang odyssey
    weight: 10
draft: false
---

  记录了一些channels常见的场景，以及自己的一些感受：

- 使用通道进行异步和并发编程是简单和惬意的；
- 通道同步技术比被很多其它语言采用的其它同步方案（比如[角色模型](https://en.wikipedia.org/wiki/Actor_model)和[async/await模式](https://en.wikipedia.org/wiki/Async/await)）有着更多的应用场景和更多的使用变种。

  通道作为同步手段，并非在任何情况下都是最佳的同步技术，本文也会补充原子操作和sync包内其他的技术作为参考。



### 将通道用做future/promise

很多其它流行语言支持future/promise来实现异步（并发）编程。 Future/promise常常用在请求/回应场合。

#### 返回单向接收通道做为函数返回结果

在下面这个例子中，`sumSquares`函数调用的两个实参请求并发进行。 每个通道读取操作将阻塞到请求返回结果为止。 两个实参总共需要大约3秒钟（而不是6秒钟）准备完毕（以较慢的一个为准）。

```go
package main

import (
	"time"
	"math/rand"
	"fmt"
)

func longTimeRequest() <-chan int32 {
	r := make(chan int32)

	go func() {
		time.Sleep(time.Second * 3) // 模拟一个工作负载
		r <- rand.Int31n(100)
	}()

	return r
}

func sumSquares(a, b int32) int32 {
	return a*a + b*b
}

func main() {
	rand.Seed(time.Now().UnixNano())

	a, b := longTimeRequest(), longTimeRequest()
	fmt.Println(sumSquares(<-a, <-b))
}
```



#### 将单向发送通道类型用做函数实参

和上例一样，在下面这个例子中，`sumSquares`函数调用的两个实参的请求也是并发进行的。 和上例不同的是`longTimeRequest`函数接收一个单向发送通道类型参数而不是返回一个单向接收通道结果。

```go
package main

import (
	"time"
	"math/rand"
	"fmt"
)

func longTimeRequest(r chan<- int32)  {
	time.Sleep(time.Second * 3) // 模拟一个工作负载
	r <- rand.Int31n(100)
}

func sumSquares(a, b int32) int32 {
	return a*a + b*b
}

func main() {
	rand.Seed(time.Now().UnixNano())

	ra, rb := make(chan int32), make(chan int32)
	go longTimeRequest(ra)
	go longTimeRequest(rb)

	fmt.Println(sumSquares(<-ra, <-rb))
}
```

对于上面这个特定的例子，我们可以只使用一个通道来接收回应结果，因为两个参数的作用是对等的。

```go
...

	results := make(chan int32, 2) // 缓冲与否不重要
	go longTimeRequest(results)
	go longTimeRequest(results)

	fmt.Println(sumSquares(<-results, <-results))
}
```

这可以看作是后面将要提到的数据聚合的一个应用。

#### 采用最快回应

本用例可以看作是上例中只使用一个通道变种的增强。

有时候，一份数据可能同时从多个数据源获取。这些数据源将返回相同的数据。 因为各种因素，这些数据源的回应速度参差不一，甚至某个特定数据源的多次回应速度之间也可能相差很大。 同时从多个数据源获取一份相同的数据可以有效保障低延迟。我们只需采用最快的回应并舍弃其它较慢回应。

注意：如果有*N*个数据源，为了防止被舍弃的回应对应的协程永久阻塞，则传输数据用的通道必须为一个容量至少为*N-1*的缓冲通道。

```go
package main

import (
	"fmt"
	"time"
	"math/rand"
)

func source(c chan<- int32) {
	ra, rb := rand.Int31(), rand.Intn(3) + 1
	// 睡眠1秒/2秒/3秒
	time.Sleep(time.Duration(rb) * time.Second)
	c <- ra
}

func main() {
	rand.Seed(time.Now().UnixNano())

	startTime := time.Now()
	c := make(chan int32, 5) // 必须用一个缓冲通道
	for i := 0; i < cap(c); i++ {
		go source(c)
	}
	rnd := <- c // 只有第一个回应被使用了
	fmt.Println(time.Since(startTime))
	fmt.Println(rnd)
}
```

“采用最快回应”用例还有一些其它实现方式，本文后面将会谈及。

#### 更多“请求/回应”用例变种

做为函数参数和返回结果使用的通道可以是缓冲的，从而使得请求协程不需阻塞到它所发送的数据被接收为止。

有时，一个请求可能并不保证返回一份有效的数据。对于这种情形，我们可以使用一个形如`struct{v T; err error}`的结构体类型或者一个空接口类型做为通道的元素类型以用来区分回应的值是否有效。

有时，一个请求可能需要比预期更长的用时才能回应，甚至永远都得不到回应。 我们可以使用本文后面将要介绍的超时机制来应对这样的情况。

有时，回应方可能会不断地返回一系列值，这也同时属于后面将要介绍的数据流的一个用例。



### 使用通道实现通知

通知可以被看作是特殊的请求/回应用例。在一个通知用例中，我们并不关心回应的值，我们只关心回应是否已发生。 所以我们常常使用空结构体类型`struct{}`来做为通道的元素类型，因为空结构体类型的尺寸为零，能够节省一些内存（虽然常常很少量）。

#### 向一个通道发送一个值来实现单对单通知

我们已知道，如果一个通道中无值可接收，则此通道上的下一个接收操作将阻塞到另一个协程发送一个值到此通道为止。 所以一个协程可以向此通道发送一个值来通知另一个等待着从此通道接收数据的协程。

在下面这个例子中，通道`done`被用来做为一个信号通道来实现单对单通知。

```go
package main

import (
	"crypto/rand"
	"fmt"
	"os"
	"sort"
)

func main() {
	values := make([]byte, 32 * 1024 * 1024)
	if _, err := rand.Read(values); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	done := make(chan struct{}) // 也可以是缓冲的

	// 排序协程
	go func() {
		sort.Slice(values, func(i, j int) bool {
			return values[i] < values[j]
		})
		done <- struct{}{} // 通知排序已完成
	}()

	// 并发地做一些其它事情...

	<- done // 等待通知
	fmt.Println(values[0], values[len(values)-1])
}
```

#### 从一个通道接收一个值来实现单对单通知

如果一个通道的数据缓冲队列已满（非缓冲的通道的数据缓冲队列总是满的）但它的发送协程队列为空，则向此通道发送一个值将阻塞，直到另外一个协程从此通道接收一个值为止。 所以我们可以通过从一个通道接收数据来实现单对单通知。一般我们使用非缓冲通道来实现这样的通知。

**这种通知方式不如上例中介绍的方式使用得广泛，基本很少用**。

```go
package main

import (
	"fmt"
	"time"
)

func main() {
	done := make(chan struct{})
		// 此信号通道也可以缓冲为1。如果这样，则在下面
		// 这个协程创建之前，我们必须向其中写入一个值。

	go func() {
		fmt.Print("Hello")
		// 模拟一个工作负载。
		time.Sleep(time.Second * 2)

		// 使用一个接收操作来通知主协程。
		<- done
	}()

	done <- struct{}{} // 阻塞在此，等待通知
	fmt.Println(" world!")
}
```

另一个事实是，上面的两种单对单通知方式其实并没有本质的区别。 它们都可以被概括为较快者等待较慢者发出通知。



#### 多对单和单对多通知

略微扩展一下上面两个用例，我们可以很轻松地实现多对单和单对多通知。

```go
package main

import "log"
import "time"

type T = struct{}

func worker(id int, ready <-chan T, done chan<- T) {
	<-ready // 阻塞在此，等待通知
	log.Print("Worker#", id, "开始工作")
	// 模拟一个工作负载。
	time.Sleep(time.Second * time.Duration(id+1))
	log.Print("Worker#", id, "工作完成")
	done <- T{} // 通知主协程（N-to-1）
}

func main() {
	log.SetFlags(0)

	ready, done := make(chan T), make(chan T)
	go worker(0, ready, done)
	go worker(1, ready, done)
	go worker(2, ready, done)

	// 模拟一个初始化过程
	time.Sleep(time.Second * 3 / 2)
	// 单对多通知
	ready <- T{}; ready <- T{}; ready <- T{}
	// 等待被多对单通知
	<-done; <-done; <-done
}
```

  这种写法是比较少见的，因为not clean enough，一般用`sync.WaitGroup`实现多对单的通知，使用关闭一个通道方式实现单对多。



#### 通过关闭一个通道来实现群发通知（单对多模式优化）

  关闭一个通道进行对多通知更简单。用到的特性是`能够从一个已经关闭的通道接受到无穷多的值`。

  我们可以把上一个例子中的三个数据发送操作`ready <- struct{}{}`替换为一个通道关闭操作`close(ready)`来达到同样的单对多通知效果。

```go
...
	close(ready) // 群发通知Let's go!
...
```

  其实，单对单通知一般也是用关闭通道的方式，这也是实践中用到最多的通知实现方式。`context`库中用这种特性实现了传达`操作取消`消息，后续会介绍具体的cases。



#### 定时通知（timer）

  标准库里的time.After的实现，也就是函数返回一个channel（容量为1的缓冲通道），起一个gorountine等待一段时间后往这个channel里送一个空结构体，类似的逻辑。



### 将通道用做互斥锁（mutex）

  运用容量为1的缓冲通道作为多次性二元semaphore，也就是mutex，这种mutex不如sync标准包里的高效。

有两种方式将一个容量为1的缓冲通道用做互斥锁：

1. 通过发送操作来加锁，通过接收操作来解锁；
2. 通过接收操作来加锁，通过发送操作来解锁。

  写一个发送操作加锁的例子，第二种反一下就行：

```go
package main

import "fmt"

func main() {
	mutex := make(chan struct{}, 1) // 容量必须为1

	counter := 0
	increase := func() {
		mutex <- struct{}{} // 加锁
		counter++
		<-mutex // 解锁
	}

	increase1000 := func(done chan<- struct{}) {
		for i := 0; i < 1000; i++ {
			increase()
		}
		done <- struct{}{}
	}

	done := make(chan struct{})
	go increase1000(done)
	go increase1000(done)
	<-done; <-done
	fmt.Println(counter) // 2000
}
```



### 将通道用做计数信号量（counting semaphore）

  缓冲通道可以用于作为counting semaphore，也就是多主锁 -- 如果一个缓冲通道的容量为`N`，那么它可以被看作是一个在任何时刻最多可有`N`个主人的锁。 上面提到的二元信号量是特殊的计数信号量，每个二元信号量在任一时刻最多只能有一个主人。

  **计数信号量经常被使用于限制最大并发数。**

  和将通道用做互斥锁一样，也有两种方式用来获取一个用做计数信号量的通道的一份所有权。

1. 通过发送操作来获取所有权，通过接收操作来释放所有权；
2. 通过接收操作来获取所有权，通过发送操作来释放所有权。

  下面是一个酒吧在座位数一定的前提下服务客人的例子，以接受操作获取所有权，先来一个简化版:

```go
package main

import (
	"log"
	"time"
	"math/rand"
)

type Seat int
type Bar chan Seat

func (bar Bar) ServeCustomer(c int) {
	log.Print("顾客#", c, "进入酒吧")
	seat := <- bar // 需要一个位子来喝酒
	log.Print("++ customer#", c, " drinks at seat#", seat)
	log.Print("++ 顾客#", c, "在第", seat, "个座位开始饮酒")
	time.Sleep(time.Second * time.Duration(2 + rand.Intn(6)))
	log.Print("-- 顾客#", c, "离开了第", seat, "个座位")
	bar <- seat // 释放座位，离开酒吧
}

func main() {
	rand.Seed(time.Now().UnixNano())

	bar24x7 := make(Bar, 10) // 此酒吧有10个座位
	// 摆放10个座位。
	for seatId := 0; seatId < cap(bar24x7); seatId++ {
		bar24x7 <- Seat(seatId) // 均不会阻塞
	}

	for customerId := 0; ; customerId++ {
		time.Sleep(time.Second)
		go bar24x7.ServeCustomer(customerId)
	}
	for {time.Sleep(time.Second)} // 睡眠不属于阻塞状态
}
```

  在上例中，只有获得一个座位的顾客才能开始饮酒。 所以在任一时刻同时在喝酒的顾客数不会超过座位数10。最后的for循环其实不太好，用`select{}`阻塞比较好。

  这个例子不好的地方，在于尽管在任一时刻同时在喝酒的顾客数不会超过座位数10，但是在某一时刻可能有多于10个顾客进入了酒吧，因为某些顾客在排队等位子。 在上例中，每个顾客对应着一个协程。虽然协程的开销比系统线程小得多，但是如果协程的数量很多，则它们的总体开销还是不能忽略不计的。 所以，最好当有空位的时候才创建顾客协程。我们可以做个小的优化：

```go
... // 省略了和上例相同的代码

func (bar Bar) ServeCustomerAtSeat(c int, seat Seat) {
	log.Print("++ 顾客#", c, "在第", seat, "个座位开始饮酒")
	time.Sleep(time.Second * time.Duration(2 + rand.Intn(6)))
	log.Print("-- 顾客#", c, "离开了第", seat, "个座位")
	bar <- seat // 释放座位，离开酒吧
}

func main() {
	rand.Seed(time.Now().UnixNano())

	bar24x7 := make(Bar, 10)
	for seatId := 0; seatId < cap(bar24x7); seatId++ {
		bar24x7 <- Seat(seatId)
	}

	// 这个for循环和上例不一样。
	for customerId := 0; ; customerId++ {
		time.Sleep(time.Second)
		seat := <- bar24x7 // 需要一个空位招待顾客
		go bar24x7.ServeCustomerAtSeat(customerId, seat)
	}
	for {time.Sleep(time.Second)}
}
```

  在上面这个修改后的例子中，在任一时刻最多只有10个顾客协程在运行（但是在程序的生命期内，仍旧会有大量的顾客协程不断被创建和销毁）。

  所以考虑每个gorountine作为消费者，不断从customers的通道去获取客人，这样消费者的数量是一定的。在下面这个更加高效的实现中，在程序的生命期内最多只会有10个顾客协程被创建出来:

```go
... // 省略了和上例相同的代码

func (bar Bar) ServeCustomerAtSeat(consumers chan int) {
	for c := range consumers {
		seatId := <- bar
		log.Print("++ 顾客#", c, "在第", seatId, "个座位开始饮酒")
		time.Sleep(time.Second * time.Duration(2 + rand.Intn(6)))
		log.Print("-- 顾客#", c, "离开了第", seatId, "个座位")
		bar <- seatId // 释放座位，离开酒吧
	}
}

func main() {
	rand.Seed(time.Now().UnixNano())

	bar24x7 := make(Bar, 10)
	for seatId := 0; seatId < cap(bar24x7); seatId++ {
		bar24x7 <- Seat(seatId)
	}

	consumers := make(chan int)
	for i := 0; i < cap(bar24x7); i++ {
		go bar24x7.ServeCustomerAtSeat(consumers)
	}
	
	for customerId := 0; ; customerId++ {
		time.Sleep(time.Second)
		consumers <- customerId
	}
}
```

通过发送操作来获取所有权的实现相对简单一些，省去了摆放座位的步骤：

```go
package main

import (
	"log"
	"time"
	"math/rand"
)

type Customer struct{id int}
type Bar chan Customer

func (bar Bar) ServeCustomer(c Customer) {
	log.Print("++ 顾客#", c.id, "开始饮酒")
	time.Sleep(time.Second * time.Duration(3 + rand.Intn(16)))
	log.Print("-- 顾客#", c.id, "离开酒吧")
	<- bar // 离开酒吧，腾出位子
}

func main() {
	rand.Seed(time.Now().UnixNano())

	bar24x7 := make(Bar, 10) // 最多同时服务10位顾客
	for customerId := 0; ; customerId++ {
		time.Sleep(time.Second * 2)
		customer := Customer{customerId}
		bar24x7 <- customer // 等待进入酒吧
		go bar24x7.ServeCustomer(customer)
	}
	for {time.Sleep(time.Second)}
}
```



### 使用通道传送传输通道(special case)

  一个通道类型的元素类型可以是另一个通道类型。 在下面这个例子中， 单向发送通道类型`chan<- int`是另一个通道类型`chan chan<- int`的元素类型。

  下面的例子在这个场景其实不是最好的方案，还在关注类似的场景。

 ```go
package main

import "fmt"

var counter = func (n int) chan<- chan<- int {
	requests := make(chan chan<- int)
	go func() {
		for request := range requests {
			if request == nil {
				n++ // 递增计数
			} else {
				request <- n // 返回当前计数
			}
		}
	}()
	return requests // 隐式转换到类型chan<- (chan<- int)
}(0)

func main() {
	increase1000 := func(done chan<- struct{}) {
		for i := 0; i < 1000; i++ {
			counter <- nil
		}
		done <- struct{}{}
	}

	done := make(chan struct{})
	go increase1000(done)
	go increase1000(done)
	<-done; <-done

	request := make(chan int, 1)
	counter <- request
	fmt.Println(<-request) // 2000
}

 ```



### 检查通道的长度和容量

  我们可以使用内置函数`cap`和`len`来查看一个通道的容量和当前长度。 但是在实践中我们很少这样做。我们很少使用内置函数`cap`的原因是一个通道的容量常常是已知的或者不重要的。 我们很少使用内置函数`len`的原因是一个**`len`调用的结果并不能总能准确地反映出的一个通道的当前长度。**

  但有时确实有一些场景需要调用这两个函数。比如，有时一个协程欲将一个未关闭的并且不会再向其中发送数据的缓冲通道中的所有数据接收出来，在确保只有此一个协程从此通道接收数据的情况下，我们可以用下面的代码来实现：

```go
for len(c) > 0 {
	value := <-c
	// 使用value ...
}
```

   不过这种场景可以用尝试接收机制来实现，这两种方式的运行效率差距不大，但尝试接收机制的优点是多个协程可以并发地进行读取操作。

  有时一个协程欲将一个缓冲通道写满而又不阻塞，在确保只有此一个协程向此通道发送数据的情况下，我们可以用下面的代码实现这一目的：

```go
for len(c) < cap(c) {
	c <- aValue
}
```

  当然，尝试发送机制也能cover。



### 尝试发送和尝试接收(无阻塞select/default)

含有一个`default`分支和一个`case`分支的`select`代码块可以被用做一个尝试发送或者尝试接收操作，取决于`case`关键字后跟随的是一个发送操作还是一个接收操作。

- 如果`case`关键字后跟随的是一个发送操作，则此`select`代码块为一个尝试发送操作。 如果`case`分支的发送操作是阻塞的，则`default`分支将被执行，发送失败；否则发送成功，`case`分支得到执行。
- 如果`case`关键字后跟随的是一个接收操作，则此`select`代码块为一个尝试接收操作。 如果`case`分支的接收操作是阻塞的，则`default`分支将被执行，接收失败；否则接收成功，`case`分支得到执行。

***尝试发送和尝试接收代码块永不阻塞。***

标准编译器对尝试发送和尝试接收代码块做了特别的优化，使得它们的执行效率比多`case`分支的普通`select`代码块执行效率高得多。

```go
package main

import "fmt"

func main() {
	type Book struct{id int}
	bookshelf := make(chan Book, 3)

	for i := 0; i < cap(bookshelf) * 2; i++ {
		select {
		case bookshelf <- Book{id: i}:
			fmt.Println("成功将书放在书架上", i)
		default:
			fmt.Println("书架已经被占满了")
		}
	}

	for i := 0; i < cap(bookshelf) * 2; i++ {
		select {
		case book := <-bookshelf:
			fmt.Println("成功从书架上取下一本书", book.id)
		default:
			fmt.Println("书架上已经没有书了")
		}
	}
}
```

  输出结果很简单:

```
成功将书放在书架上 0
成功将书放在书架上 1
成功将书放在书架上 2
书架已经被占满了
书架已经被占满了
书架已经被占满了
成功从书架上取下一本书 0
成功从书架上取下一本书 1
成功从书架上取下一本书 2
书架上已经没有书了
书架上已经没有书了
书架上已经没有书了
```

 

#### 无阻塞地检查一个通道是否已经关闭

  假设我们可以保证没有任何协程会向一个通道发送数据，则我们可以使用下面的代码来（并发安全地）检查此通道是否已经关闭，此检查不会阻塞当前协程。

```go
func IsClosed(c chan T) bool {
	select {
	case <-c:
		return true
	default:
	}
	return false
}
```

   常用场景：此方法常用来查看某个期待中的通知是否已经来临。此通知将由另一个协程通过**关闭一个通道**来发送。



#### 峰值限制（peak/burst limiting）

  通道用作counting semaphore + 通道尝试(发送/接收)可实现峰值限制。**目的是防止过大的并发请求数。**

  还是那个酒吧招待客户的例子，现在做一些修改能让顾客不再等待而是离去或者寻找其它酒吧。

```go
...
	bar24x7 := make(Bar, 10) // 此酒吧只能同时招待10个顾客
	for customerId := 0; ; customerId++ {
		time.Sleep(time.Second)
		consumer := Consumer{customerId}
		select {
		case bar24x7 <- consumer: // 试图进入此酒吧
			go bar24x7.ServeConsumer(consumer)
		default:
			log.Print("顾客#", customerId, "不愿等待而离去")
		}
	}
...
```



#### 另一种“采用最快回应”的实现方式(复杂场景用)

  在上面的“采用最快回应”用例一节已经提到，我们也可以使用选择机制来实现“采用最快回应”用例。 每个数据源协程只需使用一个**缓冲为1的通道**并向其尝试发送回应数据即可。

  ```go
package main

import (
	"fmt"
	"math/rand"
	"time"
)

func source(c chan<- int32) {
	ra, rb := rand.Int31(), rand.Intn(3)+1
	// 休眠1秒/2秒/3秒
	time.Sleep(time.Duration(rb) * time.Second)
	select {
	case c <- ra:
	default:
	}
}

func main() {
	rand.Seed(time.Now().UnixNano())

	c := make(chan int32, 1) // 此通道容量必须至少为1
	for i := 0; i < 5; i++ {
		go source(c)
	}
	rnd := <-c // 只采用第一个成功发送的回应数据
	fmt.Println(rnd)
}
  ```

  **注意，使用选择机制来实现“采用最快回应”的代码中使用的通道的容量必须至少为1，以保证最快回应总能够发送成功。 否则，如果数据请求者因为种种原因未及时准备好接收，则所有回应者的尝试发送都将失败，从而所有回应的数据都将被错过。** 



#### 少量数据源“采用最快回应”的实现方式

  如果一个“采用最快回应”用例中的数据源的数量很少，比如两个或三个，我们可以让每个数据源使用一个单独的缓冲通道来回应数据，然后使用一个`select`代码块来同时接收这三个通道。 示例代码如下：

```go
package main

import (
	"fmt"
	"math/rand"
	"time"
)

func source() <-chan int32 {
	c := make(chan int32, 1) // 必须为一个缓冲通道
	go func() {
		ra, rb := rand.Int31(), rand.Intn(3)+1
		time.Sleep(time.Duration(rb) * time.Second)
		c <- ra
	}()
	return c
}

func main() {
	rand.Seed(time.Now().UnixNano())

	var rnd int32
	// 阻塞在此直到某个数据源率先回应。
	select{
	case rnd = <-source():
	case rnd = <-source():
	case rnd = <-source():
	}
	fmt.Println(rnd)
}
```

  **注意：如果上例中使用的通道是非缓冲的，未被选中的`case`分支对应的两个`source`函数调用中开辟的协程将处于永久阻塞状态，从而造成内存泄露。**



#### 超时机制（timeout）

  在一些请求/回应用例中，一个请求可能因为种种原因导致需要超出预期的时长才能得到回应，有时甚至永远得不到回应。 对于这样的情形，我们可以使用一个超时方案给请求者返回一个错误信息。 使用选择机制可以很轻松地实现这样的一个超时方案。

下面这个例子展示了如何实现一个支持超时设置的请求：

```go
func requestWithTimeout(timeout time.Duration) (int, error) {
	c := make(chan int)
	go doRequest(c) // 可能需要超出预期的时长回应

	select {
	case data := <-c:
		return data, nil
	case <-time.After(timeout):
		return 0, errors.New("超时了！")
	}
}
```



#### 脉搏器（ticker）

我们可以使用尝试发送操作来实现一个每隔一定时间发送一个信号的脉搏器。

```go
package main

import "fmt"
import "time"

func Tick(d time.Duration) <-chan struct{} {
	c := make(chan struct{}, 1) // 容量最好为1
	go func() {
		for {
			time.Sleep(d)
			select {
			case c <- struct{}{}:
			default:
			}
		}
	}()
	return c
}

func main() {
	t := time.Now()
	for range Tick(time.Second) {
		fmt.Println(time.Since(t))
	}
}
```

  事实上，`time`标准库包中的`Tick`函数提供了同样的功能，但效率更高。 我们应该尽量使用标准库包中的实现。



#### 速率限制（rate limiting）

  前面实现了峰值限制，同样地我们可以使用尝试机制实现速率限制，但是这里要和定时器配合实现。速率限制常用来限制吞吐和确保在一段时间内的资源使用不会超标。

  下面的例子借鉴了[官方Go维基中的例子](https://github.com/golang/go/wiki/RateLimiting)。 在此例中，任何一分钟时段内处理的请求数不会超过200

```go
package main

import "fmt"
import "time"

type Request interface{}
func handle(r Request) {fmt.Println(r.(int))}

const RateLimitPeriod = time.Minute
const RateLimit = 200 // 任何一分钟内最多处理200个请求

func handleRequests(requests <-chan Request) {
	quotas := make(chan time.Time, RateLimit)

	go func() {
		tick := time.NewTicker(RateLimitPeriod / RateLimit)
		defer tick.Stop()
		for t := range tick.C {
			select {
			case quotas <- t:
			default:
			}
		}
	}()

	for r := range requests {
		<-quotas
		go handle(r)
	}
}

func main() {
	requests := make(chan Request)
	go handleRequests(requests)
	// time.Sleep(time.Minute)
	for i := 0; ; i++ {requests <- i}
}
```

  上例的代码虽然可以保证任何一分钟时段内处理的请求数不会超过200，但是如果在开始的一分钟内没有任何请求，则接下来的某个瞬时时间点可能会同时处理最多200个请求（试着将`time.Sleep`行的注释去掉), 这可能会造成卡顿情况。我们可以将**速率限制和峰值限制**一并使用来避免出现这样的情况。

  

#### 开关

  向一个nil通道发送数据或者从中接收数据都属于阻塞操作。 利用这一事实，我们可以将一个`select`流程控制中的`case`操作中涉及的通道设置为不同的值，以使此`select`流程控制选择执行不同的分支。



#### 控制代码被执行的几率（少用）

  我们可以通过在一个`select`流程控制中使用重复的`case`操作来增加对应分支中的代码的执行几率。这种操作比较少见，下面这个例子， 函数`f`的调用执行几率大致为函数`g`的两倍:

```go
package main

import "fmt"

func main() {
	foo, bar := make(chan struct{}), make(chan struct{})
	close(foo); close(bar) // 仅为演示目的
	x, y := 0.0, 0.0
	f := func(){x++}
	g := func(){y++}
	for i := 0; i < 100000; i++ {
		select {
		case <-foo: f()
		case <-foo: f()
		case <-bar: g()
		}
	}
	fmt.Println(x/y) // 大致为2
}
```



#### 从动态数量的分支中选择（少用）

*reflection ain't good*

  每个`select`控制流程中的分支数量在运行中是固定的，但是我们可以使用`reflect`标准库包中提供的功能在运行时刻来构建动态分支数量的`select`控制流程。 但是请注意：一个`select`控制流程中的分支越多，此`select`控制流程的执行效率就越低（这是我们常常只使用不多于三个分支的`select`控制流程的原因）。 `reflect`标准库包中也提供了模拟尝试发送和尝试接收代码块的`TrySend`和`TryRecv`函数。涉及到反射的，肯定有更好的解决方案:-)。



### 数据流操纵（Data Flow）

  下面把数据流处理程序大致分成了常见的几类，概念上来说是不同模块由一个或者多个并行处理的协程组成：

- 数据生成/搜集/加载；
- 数据服务/存盘；
- 数据计算/处理；
- 数据验证/过滤；
- 数据聚合/分流；
- 数据组合/拆分；
- 数据复制/增殖；

  一个模块中的工作协程从一些其它模块接收数据做为输入，并向另一些模块发送输出数据。 换句话数，一个模块可能同时兼任数据消费者和数据产生者的角色。多个模块一起组成了一个数据流处理系统。后续的一些实现可能并不高效，只是为了描述这些分类模块的实现，比较简单。



#### 数据生成/搜集/加载

  一个数据产生者可能通过以下途径生成数据：

- 加载一个文件、或者读取一个数据库、或者用爬虫抓取网页数据；
- 从一个软件或者硬件系统搜集各种数据；
- 产生一系列随机数；
- etc.

  这里的例子是一个随机数生成器作为数据生产者，生产者只有数据输出，所以返回只读通道。实际上此随机数产生器是一个多返回值的future/promise，**一个数据产生者可以在任何时刻关闭返回的通道以结束数据生成**。

```go
import (
	"crypto/rand"
	"encoding/binary"
)

func RandomGenerator() <-chan uint64 {
	c := make(chan uint64)
	go func() {
		rnds := make([]byte, 8)
		for {
			_, err := rand.Read(rnds)
			if err != nil {
				close(c)  // 若读取错误则关闭通道结束数据生成
				break
			}
			c <- binary.BigEndian.Uint64(rnds)
		}
	}()
	return c
}
```



#### 数据聚合

  例如把多个数据流合为一个数据流，下面这个函数把任意数量的数据流合为一个：

```go
func Aggregator(inputs ...<-chan uint64) <-chan uint64 {
	out := make(chan uint64)
	for _, in := range inputs {
		go func(in <-chan uint64) {
			for {
				out <- <-in // <=> out <- (<-in)
			}
		}(in)
	}
	return out
}
```

![merge cahnnels](https://miro.medium.com/max/1400/0*oNO9-NZHuTerKDym.png)

但是这个例子，最好需要考虑一个输入数据流是否已经关闭：

```go
import "sync"

func Aggregator(inputs ...<-chan uint64) <-chan uint64 {
	output := make(chan uint64)
	var wg sync.WaitGroup
	for _, in := range inputs {
		wg.Add(1)
		go func(int <-chan uint64) {
			defer wg.Done()
			// 如果通道in被关闭，此循环将最终结束。
			for x := range in {
				output <- x
			}
		}(in)
	}
	go func() {
		wg.Wait()
		close(output)
	}()
	return output
}
```

如果被聚合的数据流的数量很小，我们也可以使用一个`select`控制流程代码块来聚合这些数据流。

```go
// 假设数据流的数量为2。
...
	output := make(chan uint64)
	go func() {
		inA, inB := inputs[0], inputs[1]
		for {
			select {
			case v := <- inA: output <- v
			case v := <- inB: output <- v
			}
		}
	}
...
```

#### 数据分流

  数据分流是数据聚合的逆过程。数据分流的实现很简单，但在实践中用的并不多。

```go
func Divisor(input <-chan uint64, outputs ...chan<- uint64) {
	for _, out := range outputs {
		go func(o chan<- uint64) {
			for {
				o <- <-input // <=> o <- (<-input)
			}
		}(out)
	}
}
```

#### 数据合成

  数据合成将多个数据流中读取的数据合成一个。

  下面是一个数据合成工作函数的实现中，从两个不同数据流读取的两个`uint64`值组成了一个新的`uint64`值。 当然，在实践中，数据的组合比这复杂得多。

```go
func Composor(inA, inB <-chan uint64) <-chan uint64 {
	output := make(chan uint64)
	go func() {
		for {
			a1, b, a2 := <-inA, <-inB, <-inA
			output <- a1 ^ b & a2
		}
	}()
	return output
}
```

#### 数据分解

  数据分解是数据合成的逆过程。一个数据分解者从一个通道读取一份数据，并将此数据分解为多份数据。 这里就不举例了。

#### 数据复制/增殖

  数据复制（增殖）可以看作是特殊的数据分解。一份输入数据将被复制多份并输出给多个数据流。

```go
func Duplicator(in <-chan uint64) (<-chan uint64, <-chan uint64) {
	outA, outB := make(chan uint64), make(chan uint64)
	go func() {
		for x := range in {
			outA <- x
			outB <- x
		}
	}()
	return outA, outB
}
```

#### 数据计算/分析

  数据计算和数据分析模块的功能因具体程序不同而有很大的差异。 一般来说，数据分析者接收一份数据并对之加工处理后转换为另一份数据。

  下面的简单示例中，每个输入的`uint64`值将被进行位反转后输出。

```go
func Calculator(in <-chan uint64, out chan uint64) (<-chan uint64) {
	if out == nil {
		out = make(chan uint64)
	}
	go func() {
		for x := range in {
			out <- ^x
		}
	}()
	return out
}
```



#### 数据验证/过滤

  一个数据验证或过滤者的任务是检查输入数据的合理性并抛弃不合理的数据。 比如，下面的工作者协程将抛弃所有的非素数。

```go
import "math/big"

func Filter0(input <-chan uint64, output chan uint64) <-chan uint64 {
	if output == nil {
		output = make(chan uint64)
	}
	go func() {
		bigInt := big.NewInt(0)
		for x := range input {
			bigInt.SetUint64(x)
			if bigInt.ProbablyPrime(1) {
				output <- x
			}
		}
	}()
	return output
}

func Filter(input <-chan uint64) <-chan uint64 {
	return Filter0(input, nil)
}
```

#### 数据服务/存盘

  一般，一个数据服务或者存盘模块为一个数据流系统中的最后一个模块。 这里的实现值是简单地将数据输出到终端。

```go
import "fmt"

func Printer(input <-chan uint64) {
	for x := range input {
		fmt.Println(x)
	}
}
```

#### 组装数据流系统

Now，让我们使用上面的模块工作者函数实现来组装一些数据流系统。 组装数据流仅仅是创建一些工作者协程函数调用，并为这些调用指定输入数据流和输出数据流。

数据流系统例子1（一个流线型系统）

```go
package main

... // 上面的模块工作者函数实现

func main() {
	Printer(
		Filter(
			Calculator(
				RandomGenerator(), nil,
			),
		),
	)
}
```

![pipe](https://gfw.go101.org/article/res/data-flow-linear.png)

数据流系统例子2（一个单向无环图系统）：

```go
package main

... // 上面的模块工作者函数实现

func main() {
	filterA := Filter(RandomGenerator())
	filterB := Filter(RandomGenerator())
	filterC := Filter(RandomGenerator())
	filter := Aggregator(filterA, filterB, filterC)
	calculatorA := Calculator(filter, nil)
	calculatorB := Calculator(filter, nil)
	calculator := Aggregator(calculatorA, calculatorB)
	Printer(calculator)
}
```

![no circle](https://gfw.go101.org/article/res/data-flow-dag.png)

  更复杂的数据流系统可以表示为任何拓扑结构的图。比如一个复杂的数据流系统可能有多个输出模块。 但是有环拓扑结构的数据流系统在实践中很少用。

  从上面两个例子可以看出，使用通道来构建数据流系统是很简单和直观的。而且，通过使用数据聚合模块，我们可以很轻松地实现各个模块的工作协程数量的扇入（fan-in）和扇出（fan-out）。

  事实上，我们也可以使用一个简单的通道来代替数据聚合模块的角色。比如，下面的代码使用两个通道代替了上例中的两个数据聚合器。

```go
package main

... // 上面的模块工作者函数实现

func main() {
	c1 := make(chan uint64, 100)
	Filter0(RandomGenerator(), c1) // filterA
	Filter0(RandomGenerator(), c1) // filterB
	Filter0(RandomGenerator(), c1) // filterC
	c2 := make(chan uint64, 100)
	Calculator(c1, c2) // calculatorA
	Calculator(c1, c2) // calculatorB
	Printer(c2)
}
```

![new topo](https://gfw.go101.org/article/res/data-flow-dag-b.png)

上面的代码示例并没有太多考虑如何关闭一个数据流，会单独开一个文章介绍。