---
title: "Close Channels Gracefully"
date: 2020-12-24T11:22:18+08:00
hero: /images/posts/golang2.jpg
menu:
  sidebar:
    name: 优雅关闭通道
    identifier: channels-gracefully-closed
    parent: golang odyssey
    weight: 10
draft: false
---

## 优雅地关闭通道

### 场景一：M个接收者和一个发送者。发送者通过关闭用来传输数据的通道来传递发送结束信号

  这是最简单的一种情形。当发送者欲结束发送，让它关闭用来传输数据的通道即可。

```go
package main

import (
	"time"
	"math/rand"
	"sync"
	"log"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	log.SetFlags(0)

	// ...
	const Max = 100000
	const NumReceivers = 100

	wgReceivers := sync.WaitGroup{}
	wgReceivers.Add(NumReceivers)

	// ...
	dataCh := make(chan int)

	// 发送者
	go func() {
		for {
			if value := rand.Intn(Max); value == 0 {
				// 此唯一的发送者可以安全地关闭此数据通道。
				close(dataCh)
				return
			} else {
				dataCh <- value
			}
		}
	}()

	// 接收者
	for i := 0; i < NumReceivers; i++ {
		go func() {
			defer wgReceivers.Done()

			// 接收数据直到通道dataCh已关闭
			// 并且dataCh的缓冲队列已空。
			for value := range dataCh {
				log.Println(value)
			}
		}()
	}

	wgReceivers.Wait()
}
```



### 场景二： 一个接收者和N个发送者，此唯一接收者通过关闭一个额外的信号通道来通知发送者不要在发送数据了

  此情形比上一种情形复杂一些。我们不能让接收者关闭用来传输数据的通道来停止数据传输，因为这样做违反了**通道关闭原则**。 但是我们可以让接收者关闭一个额外的信号通道来通知发送者不要在发送数据了。

```go
package main

import (
	"time"
	"math/rand"
	"sync"
	"log"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	log.SetFlags(0)

	// ...
	const Max = 100000
	const NumSenders = 1000

	wgReceivers := sync.WaitGroup{}
	wgReceivers.Add(1)

	// ...
	dataCh := make(chan int)
	stopCh := make(chan struct{})
		// stopCh是一个额外的信号通道。它的
		// 发送者为dataCh数据通道的接收者。
		// 它的接收者为dataCh数据通道的发送者。

	// 发送者
	for i := 0; i < NumSenders; i++ {
		go func() {
			for {
				// 这里的第一个尝试接收用来让此发送者
				// 协程尽早地退出。对于这个特定的例子，
				// 此select代码块并非必需。
				select {
				case <- stopCh:
					return
				default:
				}

				// 即使stopCh已经关闭，此第二个select
				// 代码块中的第一个分支仍很有可能在若干个
				// 循环步内依然不会被选中。如果这是不可接受
				// 的，则上面的第一个select代码块是必需的。
				select {
				case <- stopCh:
					return
				case dataCh <- rand.Intn(Max):
				}
			}
		}()
	}

	// 接收者
	go func() {
		defer wgReceivers.Done()

		for value := range dataCh {
			if value == Max-1 {
				// 此唯一的接收者同时也是stopCh通道的
				// 唯一发送者。尽管它不能安全地关闭dataCh数
				// 据通道，但它可以安全地关闭stopCh通道。
				close(stopCh)
				return
			}

			log.Println(value)
		}
	}()

	// ...
	wgReceivers.Wait()
}
```

如此例中的注释所述，对于此额外的信号通道`stopCh`，它只有一个发送者，即`dataCh`数据通道的唯一接收者。 `dataCh`数据通道的接收者关闭了信号通道`stopCh`，这是不违反**通道关闭原则**的。

在此例中，数据通道`dataCh`并没有被关闭。是的，我们不必关闭它。 **当一个通道不再被任何协程所使用后，它将逐渐被垃圾回收掉**，无论它是否已经被关闭。 所以这里的优雅性体现在通过**不关闭一个通道**来停止使用此通道。



### 场景三：M个接收者和N个发送者。它们中的任何协程都可以让一个中间调解协程帮忙发出停止数据传送的信号

这是最复杂的一种情形。我们不能让接收者和发送者中的任何一个关闭用来传输数据的通道，我们也不能让多个接收者之一关闭一个额外的信号通道。 这两种做法都违反了**通道关闭原则**。 然而，我们可以引入一个中间调解者角色并让其关闭额外的信号通道来通知所有的接收者和发送者结束工作。 具体实现见下例。注意其中使用了一个尝试发送操作来向中间调解者发送信号。

```go
package main

import (
	"time"
	"math/rand"
	"sync"
	"log"
	"strconv"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	log.SetFlags(0)

	// ...
	const Max = 100000
	const NumReceivers = 10
	const NumSenders = 1000

	wgReceivers := sync.WaitGroup{}
	wgReceivers.Add(NumReceivers)

	// ...
	dataCh := make(chan int)
	stopCh := make(chan struct{})
		// stopCh是一个额外的信号通道。它的发送
		// 者为中间调解者。它的接收者为dataCh
		// 数据通道的所有的发送者和接收者。
	toStop := make(chan string, 1)
		// toStop是一个用来通知中间调解者让其
		// 关闭信号通道stopCh的第二个信号通道。
		// 此第二个信号通道的发送者为dataCh数据
		// 通道的所有的发送者和接收者，它的接收者
		// 为中间调解者。它必须为一个缓冲通道。

	var stoppedBy string

	// 中间调解者
	go func() {
		stoppedBy = <-toStop
		close(stopCh)
	}()

	// 发送者
	for i := 0; i < NumSenders; i++ {
		go func(id string) {
			for {
				value := rand.Intn(Max)
				if value == 0 {
					// 为了防止阻塞，这里使用了一个尝试
					// 发送操作来向中间调解者发送信号。
					select {
					case toStop <- "发送者#" + id:
					default:
					}
					return
				}

				// 此处的尝试接收操作是为了让此发送协程尽早
				// 退出。标准编译器对尝试接收和尝试发送做了
				// 特殊的优化，因而它们的速度很快。
				select {
				case <- stopCh:
					return
				default:
				}

				// 即使stopCh已关闭，如果这个select代码块
				// 中第二个分支的发送操作是非阻塞的，则第一个
				// 分支仍很有可能在若干个循环步内依然不会被选
				// 中。如果这是不可接受的，则上面的第一个尝试
				// 接收操作代码块是必需的。
				select {
				case <- stopCh:
					return
				case dataCh <- value:
				}
			}
		}(strconv.Itoa(i))
	}

	// 接收者
	for i := 0; i < NumReceivers; i++ {
		go func(id string) {
			defer wgReceivers.Done()

			for {
				// 和发送者协程一样，此处的尝试接收操作是为了
				// 让此接收协程尽早退出。
				select {
				case <- stopCh:
					return
				default:
				}

				// 即使stopCh已关闭，如果这个select代码块
				// 中第二个分支的接收操作是非阻塞的，则第一个
				// 分支仍很有可能在若干个循环步内依然不会被选
				// 中。如果这是不可接受的，则上面尝试接收操作
				// 代码块是必需的。
				select {
				case <- stopCh:
					return
				case value := <-dataCh:
					if value == Max-1 {
						// 为了防止阻塞，这里使用了一个尝试
						// 发送操作来向中间调解者发送信号。
						select {
						case toStop <- "接收者#" + id:
						default:
						}
						return
					}

					log.Println(value)
				}
			}
		}(strconv.Itoa(i))
	}

	// ...
	wgReceivers.Wait()
	log.Println("被" + stoppedBy + "终止了")
}
```

在此例中，**通道关闭原则**依旧得到了遵守。

请注意，信号通道`toStop`的容量必须至少为1。 如果它的容量为0，则在**中间调解者还未准备好的情况下就已经有某个协程向`toStop`发送信号时，此信号将被抛弃。因为停止信号是通过非阻塞的尝试发送传递的。**

我们也可以不使用尝试发送操作向中间调解者发送信号，但信号通道`toStop`的容量必须至少为数据发送者和数据接收者的数量之和，以防止向其发送数据时（有一个极其微小的可能）导致某些发送者和接收者协程永久阻塞。

```go
...
toStop := make(chan string, NumReceivers + NumSenders)
...
			value := rand.Intn(Max)
			if value == 0 {
				toStop <- "sender#" + id
				return
			}
...
				if value == Max-1 {
					toStop <- "receiver#" + id
					return
				}
...
```

### 场景四： M个接收者和一个发送者”情形的一个变种：用来传输数据的通道的关闭请求由第三方发出

有时，数据通道（`dataCh`）的关闭请求需要由某个第三方协程发出。对于这种情形，我们可以使用一个额外的信号通道来通知唯一的发送者关闭数据通道（`dataCh`）。

```go
package main

import (
	"time"
	"math/rand"
	"sync"
	"log"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	log.SetFlags(0)

	// ...
	const Max = 100000
	const NumReceivers = 100
	const NumThirdParties = 15

	wgReceivers := sync.WaitGroup{}
	wgReceivers.Add(NumReceivers)

	// ...
	dataCh := make(chan int)
	closing := make(chan struct{}) // 信号通道
	closed := make(chan struct{})
	
	// 此stop函数可以被安全地多次调用。
	stop := func() {
		select {
		case closing<-struct{}{}:
			<-closed
		case <-closed:
		}
	}
	
	// 一些第三方协程
	for i := 0; i < NumThirdParties; i++ {
		go func() {
			r := 1 + rand.Intn(3)
			time.Sleep(time.Duration(r) * time.Second)
			stop()
		}()
	}

	// 发送者
	go func() {
		defer func() {
			close(closed)
			close(dataCh)
		}()

		for {
			select{
			case <-closing: return
			default:
			}

			select{
			case <-closing: return
			case dataCh <- rand.Intn(Max):
			}
		}
	}()

	// 接收者
	for i := 0; i < NumReceivers; i++ {
		go func() {
			defer wgReceivers.Done()

			for value := range dataCh {
				log.Println(value)
			}
		}()
	}

	wgReceivers.Wait()
}
```

上述代码中的`stop`函数中使用的技巧偷自Roger Peppe在[此贴](https://groups.google.com/forum/#!msg/golang-nuts/lEKehHH7kZY/SRmCtXDZAAAJ)中的一个留言。



### 场景五：“N个发送者”的一个变种：用来传输数据的通道必须被关闭以通知各个接收者数据发送已经结束了

在上面的提到的“N个发送者”情形中，为了遵守**通道关闭原则**，我们避免了关闭数据通道（`dataCh`）。 但是有时候，数据通道（`dataCh`）必须被关闭以通知各个接收者数据发送已经结束。 对于这种“N个发送者”情形，我们可以使用一个中间通道将它们转化为“一个发送者”情形，然后继续使用上一节介绍的技巧来关闭此中间通道，从而避免了关闭原始的`dataCh`数据通道。

```go
package main

import (
	"time"
	"math/rand"
	"sync"
	"log"
	"strconv"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	log.SetFlags(0)

	// ...
	const Max = 1000000
	const NumReceivers = 10
	const NumSenders = 1000
	const NumThirdParties = 15

	wgReceivers := sync.WaitGroup{}
	wgReceivers.Add(NumReceivers)

	// ...
	dataCh := make(chan int)   // 将被关闭
	middleCh := make(chan int) // 不会被关闭
	closing := make(chan string)
	closed := make(chan struct{})

	var stoppedBy string

	stop := func(by string) {
		select {
		case closing <- by:
			<-closed
		case <-closed:
		}
	}
	
	// 中间层
	go func() {
		exit := func(v int, needSend bool) {
			close(closed)
			if needSend {
				dataCh <- v
			}
			close(dataCh)
		}

		for {
			select {
			case stoppedBy = <-closing:
				exit(0, false)
				return
			case v := <- middleCh:
				select {
				case stoppedBy = <-closing:
					exit(v, true)
					return
				case dataCh <- v:
				}
			}
		}
	}()
	
	// 一些第三方协程
	for i := 0; i < NumThirdParties; i++ {
		go func(id string) {
			r := 1 + rand.Intn(3)
			time.Sleep(time.Duration(r) * time.Second)
			stop("3rd-party#" + id)
		}(strconv.Itoa(i))
	}

	// 发送者
	for i := 0; i < NumSenders; i++ {
		go func(id string) {
			for {
				value := rand.Intn(Max)
				if value == 0 {
					stop("sender#" + id)
					return
				}

				select {
				case <- closed:
					return
				default:
				}

				select {
				case <- closed:
					return
				case middleCh <- value:
				}
			}
		}(strconv.Itoa(i))
	}

	// 接收者
	for range [NumReceivers]struct{}{} {
		go func() {
			defer wgReceivers.Done()

			for value := range dataCh {
				log.Println(value)
			}
		}()
	}

	// ...
	wgReceivers.Wait()
	log.Println("stopped by", stoppedBy)
}
```



### 结论

并没有什么情况非得逼得我们违反**通道关闭原则**。 如果你遇到了此情形，就请考虑修改你的代码流程和结构设计。

