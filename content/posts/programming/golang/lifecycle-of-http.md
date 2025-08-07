---
title: "Life of an HTTP request in a Go server"
date: 2021-02-20T11:22:18+08:00
hero: /images/posts/golang2.jpg
menu:
  sidebar:
    name: Go server中http请求的生命周期
    identifier: lifecycle-of-http-request
    parent: golang odyssey
    weight: 10
draft: false
---

  这篇文章的启发是我在阅读Go的http源码时获得的，之前对这块缺乏深入的了解，这篇文章会结合源码讨论包括典型http request的路由，还会涉及到一些并发和中间件的issue。

  我们先从一个简单的go server谈起，下面的代码从https://gobyexample.com/http-servers 截取：

```go
package main

import (
  "fmt"
  "net/http"
)

func hello(w http.ResponseWriter, req *http.Request) {
  fmt.Fprintf(w, "hello\n")
}

func headers(w http.ResponseWriter, req *http.Request) {
  for name, headers := range req.Header {
    for _, h := range headers {
      fmt.Fprintf(w, "%v: %v\n", name, h)
    }
  }
}

func main() {
  http.HandleFunc("/hello", hello)
  http.HandleFunc("/headers", headers)

  http.ListenAndServe(":8090", nil)
}
```

  追踪请求的生命周期我们从`http.ListenAndServe`这个方法开始，下面的图示说明了这一层的调用关系:

![diagram](https://eli.thegreenplace.net/images/2021/http-request-listenandserve.png)

  这里实际上`inlined`了一些代码，因为初始的代码有很多其他的细节不好追踪。

  主要的flow其实和我们预期的一致：`ListenAndServe`方法对你一个目标地址监听一个TCP端口，而后循环不断接受新的连接。每一个连接，它会起一个新的goroutine去serve，serve的具体操作是:

1. 从连接里解析HTTP请求： 产生`http.Request`
2. 将`http.Request`传给用户自定义的handler

  一个handler实际上就是实现了`http.Handler`接口：

```go
type Handler interface {
    ServeHTTP(ResponseWriter, *Request)
}
```

 ## 默认Handler

  在我们上述的代码中，`ListenAndServe`方法的第二个参数为`nil`，实际上应该是用户自定义的handler, 这是为何？我们的图解中省去了很多细节，实际上当HTTP包serve一个请求的时候，它并没有直接调用用户的handlers而是使用一个adaptor：

```go
type serverHandler struct {
  srv *Server
}

func (sh serverHandler) ServeHTTP(rw ResponseWriter, req *Request) {
  handler := sh.srv.Handler
  if handler == nil {
    handler = DefaultServeMux
  }
  if req.RequestURI == "*" && req.Method == "OPTIONS" {
    handler = globalOptionsHandler{}
  }
  handler.ServeHTTP(rw, req)
}
```

  上述代码表示了，如果`handler == nil`, `http.DefaultServeMux`会作为默认的handler。这个*default server mux*是在`http`包中一个`http.ServeMux`类全局实例。而当我们的样例代码通过`http.HandleFunc`注册handlers的时候，同样会注册到default mux中。

  所以我们可以重写我们的样例代码如下:

```go
func main() {
  mux := http.NewServeMux()
  mux.HandleFunc("/hello", hello)
  mux.HandleFunc("/headers", headers)

  http.ListenAndServe(":8090", mux)
}
```

 ## ServeMux只是一个Handler

  在看了很多Go的server例子以后，很容易会把`ListenAndServe`想象成把`mux`作为参数，但是这个明显是不准确的。从上面的例子看到，`ListenAndServe`实际传入的是实现了`http.Handler`接口的值，我们可以重写一下代码并且不用任何的muxes：

```go
type PoliteServer struct {
}

func (ms *PoliteServer) ServeHTTP(w http.ResponseWriter, req *http.Request) {
  fmt.Fprintf(w, "Welcome! Thanks for visiting!\n")
}

func main() {
  ps := &PoliteServer{}
  log.Fatal(http.ListenAndServe(":8090", ps))
}
```

  这个snippet里面没有路由，所有的HTTP请求直接传进`PoliteServer`的`ServeHTTP`参数里，并且所有的请求都有相同的响应。可以尝试用不同的路径和方法去`curl`一下这个server。

  然后我们再用`http.HandlerFunc`简化一下这个polite server:

```go
func politeGreeting(w http.ResponseWriter, req *http.Request) {
  fmt.Fprintf(w, "Welcome! Thanks for visiting!\n")
}

func main() {
  log.Fatal(http.ListenAndServe(":8090", http.HandlerFunc(politeGreeting)))
}
```

  `http.HadnlerFunc`是`http`包里的一个很好用的adaptor:

```go
// The HandlerFunc type is an adapter to allow the use of
// ordinary functions as HTTP handlers. If f is a function
// with the appropriate signature, HandlerFunc(f) is a
// Handler that calls f.
type HandlerFunc func(ResponseWriter, *Request)

// ServeHTTP calls f(w, r).
func (f HandlerFunc) ServeHTTP(w ResponseWriter, r *Request) {
  f(w, r)
}
```

  在这篇文章最开始的例子里，用到了`http.HandleFunc`，注意和`http.HandlerFunc`很像，但是他们是完全不同的实体，也承担着不同的任务。

  如同`PoliteServer`表现的那样，`http.ServeMux`是实现`http.Handler`接口的一个类，[这里查看源码](https://go.googlesource.com/go/+/go1.15.8/src/net/http/server.go)

1. `ServeMux`维护了一个以长度排序的`{pattern, handler}`切片
2. `Handle`或者`HandleFunc`向这个切片添加新的handler
3. `ServeHTTP`:
   - 通过查询这个排序好的切片，找到对应请求path的handler
   - 调用handler的`ServeHTTP`方法

  至此，mux可以被看作为一个`forwarding handler`，这种编程模式在HTTP server中很常见，也就是`middleware`。



## `http.Handler` Middleware

  如何去定义清楚middleware的含义是比较困难的，因为在不同的上下文、语言以及框架里它的概念都有一些不同。我们再看一下文章一开始的信息流图解，这里我们再简化一下，隐藏一些`http`包做的细节：

![diagram2](https://eli.thegreenplace.net/images/2021/http-request-simplified.png)

  下面是我们增加了middleware以后的图解:

![diagram3](https://eli.thegreenplace.net/images/2021/http-request-with-middleware.png)

  在Go中，middleware只是一个HTTP handler，而这个handler包了一个不同的handler。middleware handler通过调用`ListenAndServe`被注册，当这个middleware被调用到，他可以做任意的预处理，调用到被包的handler然后做任意的后处理。

  我们在上面了解了一个middleware的例子--`http.ServeMux`, 在那个例子中，预处理指的是基于特定的请求path去选择用户定义的handler，然后去调用。并且没有对应的后处理。

  举一个另外的例子，我们可以在`polite server`中加一个基本的`logging middleware`， 这个middleware能够对所有请求的的细节记录日志，包括了请求执行的时间等：

```go
type LoggingMiddleware struct {
  handler http.Handler
}

func (lm *LoggingMiddleware) ServeHTTP(w http.ResponseWriter, req *http.Request) {
  start := time.Now()
  lm.handler.ServeHTTP(w, req)
  log.Printf("%s %s %s", req.Method, req.RequestURI, time.Since(start))
}

type PoliteServer struct {
}

func (ms *PoliteServer) ServeHTTP(w http.ResponseWriter, req *http.Request) {
  fmt.Fprintf(w, "Welcome! Thanks for visiting!\n")
}

func main() {
  ps := &PoliteServer{}
  lm := &LoggingMiddleware{handler: ps}
  log.Fatal(http.ListenAndServe(":8090", lm))
}
```

  请注意`logging middleware`其本身就是一个`http.Handler`包含了用户定义的handler作为一个field。当`ListenAndServe`调用其`ServeHTTP`方法的时候，做了以下的事情:

1. 预处理： 在user handler被执行前打时间戳
2. 调用user handler，传入请求体和response writer
3. 后处理：日志记录请求细节，包括耗费的时间

  middleware一个巨大的优点是composable（组合性），被middleware包着的handler可以是另一个middleware等等。所以这个是一个相互包裹的`http.Handler`链。实际上，这个是在Go中的常见模式，这个例子也像我们展现一个经典的Go middleware是怎么样的。下面是一个`logging polite server`的详细例子，写法上更容易辨认：

```go
func politeGreeting(w http.ResponseWriter, req *http.Request) {
  fmt.Fprintf(w, "Welcome! Thanks for visiting!\n")
}

func loggingMiddleware(next http.Handler) http.Handler {
  return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
    start := time.Now()
    next.ServeHTTP(w, req)
    log.Printf("%s %s %s", req.Method, req.RequestURI, time.Since(start))
  })
}

func main() {
  lm := loggingMiddleware(http.HandlerFunc(politeGreeting))
  log.Fatal(http.ListenAndServe(":8090", lm))
}
```

  这里省去了通过方法对结构体的创建，`loggingMiddleware`利用了`http.HandlerFunc`以及闭包让代码变得更为简洁，当然功能还是和前面代码相同。但是这个写法，彰显了一个middleware的标准特征：一个函数传入一个`http.Handler`以及其他状态，然后返回另一个`http.Handler`。被返回的handler可以视作传入middleware的handler的替代品，而且会`magically`执行middleware所拥有的功能。

  例如，标准库里有如下的middleware:

```go
func TimeoutHandler(h Handler, dt time.Duration, msg string) Handler
```

  所以我们可以这样玩:

```go
handler = http.TimeoutHandler(handler, 2 * time.Second, "timed out")
```

  这样就能创建一个2秒超时机制的handler了。

  而middleware的组合可以由如下所示：

```go
handler = http.TimeoutHandler(handler, 2 * time.Second, "timed out")
handler = loggingMiddleware(handler)
```

  仅仅两行，`handler`能够有超时和记录日志的功能，你或许会感觉middleware的链条写起来可能比较繁琐，不过Go有很多流行的包会解决这个问题，当然已经超出了这篇文章讨论的范围，后续我也会补充。

  除此之外，`http`包本身也在按照其需求使用middleware，比如之前`serverHandler`适应器的例子，它能够使用非常简洁的手段去默认处理`nil`handler的情况（通过把请求传给`default mux`）

  因此，middleware可以说是一种attractive design aid，我们能够聚焦在`业务逻辑`handler，同时利用一般性的middleware去增强handler的功能，更多的探讨会新开一些文章。

  

## 并发和panic处理

  最后我们来研究额外的两个主题：并发和panic处理，作为我们探究Go HTTP  Server中HTTP请求路径问题的结尾。

  首先关于并发的问题，前面讨论了对于每一个连接，其都由`http.Server.Serve`去起一个新的gorountine去处理。这利用了Go强大的并发能力，因为goroutine非常cheap并且这种简洁的并发模型对于HTTP handlers的处理也很适宜。一个handler可以阻塞（例如读取数据库）且不会停止其他handlers。不过在处理一些共享数据的goroutine并发时，还是要注意一些东西，这点我会在另外的文章谈。

  最后，panic处理。HTTP Server一般来说是一个长期运行的程序。如果在一个用户定义的handler中发生了问题，例如一些导致runtime panic的bug，有可能会让整个server都挂掉。所以最好能够在`main`里用`recover`来保护你的server，不过这种方式还是有以下的问题:

1. 当控制返回到`main`中时，`ListenAndServe`已经结束了所以其他serving也结束了。
2. 因为每一个独立的goroutine处理一个connection，handlers里的panic甚至不会到达`main`而是挂掉整个进程。

  为了防止这些问题，`net/http`内置了对每个goroutine的recovery(在`conn.serve`方法中)，我们可以看一个例子:

```go
func hello(w http.ResponseWriter, req *http.Request) {
  fmt.Fprintf(w, "hello\n")
}

func doPanic(w http.ResponseWriter, req *http.Request) {
  panic("oops")
}

func main() {
  http.HandleFunc("/hello", hello)
  http.HandleFunc("/panic", doPanic)

  http.ListenAndServe(":8090", nil)
}
```

  如果我们起这个server并且用`/panic`去curl:

```
$ curl localhost:8090/panic
curl: (52) Empty reply from server
```

  server端会打下以下的日志:

```
2021/02/16 09:44:31 http: panic serving 127.0.0.1:52908: oops
goroutine 8 [running]:
net/http.(*conn).serve.func1(0xc00010cbe0)
  /usr/local/go/src/net/http/server.go:1801 +0x147
panic(0x654840, 0x6f0b80)
  /usr/local/go/src/runtime/panic.go:975 +0x47a
main.doPanic(0x6fa060, 0xc0001401c0, 0xc000164200)
[... rest of stack dump here ...]
```

  当然server还在持续运行。

  虽然这种内置的方式比挂掉整个进程好，不过开发者还是觉得这样有很多限制。它能做的只有关闭连接然后记录下日志，但是一般的情形下，最好给client端返回一些错误信息（例如错误码500等）。

  

