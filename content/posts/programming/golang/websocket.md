---
title: "A Million WebSocket and Go"
date: 2021-01-16T11:22:18+08:00
hero: /images/posts/golang2.jpg
menu:
  sidebar:
    name: Million WebSocket
    identifier: websocket
    parent: golang odyssey
    weight: 10
draft: false
---

*这篇文章是我研究高负载网络服务器架构看到的的一个有趣的story，添加了我自身学习websocket的感受和记录，希望我能在飞机落地前写完:-)*



## Preface

  我们先描述一个问题作为讨论的中心：用户邮件的存储方法。

  对于这种主题，有很多种方式在系统内对邮件状态进行持续的追踪，比如系统事件是一个方式，另一种方式可以通过定期的系统轮询有关状态变化。

  这两种方式各有利弊，不过当我们讨论到邮件的时候，用户希望收到新邮件的速度越快越好。邮件轮询每秒约有50000个HTTP请求，其中60%返回304状态，也就是邮箱内没有任何修改。

  因此，为了减少服务器的负载并加快向用户传递邮件的速度，我们决定通过编写publisher-subscriber服务器(即bus, message broker, event channel)来重新发明轮子。一方面接受有关状态变更的通知，另外一个方面接受此类通知的订阅。

  改进前：

```
+--------------+     (2)    +-------------+      (1)    +-----------+
|              | <--------+ |             |  <--------+ |           |
|    Storage   |            |     API     |     HTTP    |  Browser  |
|              | +--------> |             |  +--------> |           |
+--------------+     (3)    +-------------+      (4)    +-----------+

```

  改进后:

```
+--------------+            +-------------+   WebSocket  +-----------+
|    Storage   |            |     API     | +----------> |  Browser  |
+--------------+            +-------------+      (3)     +-----------+
       +                           ^
       | (1)                       | (2)
       v                           +
+-----------------------------------------+
|                  Bus                    |
+-----------------------------------------+
```

  改进前的方案也就是browser定期去查询api并访问存储更改

  改进后的方案描述了新的架构，browser和通知api建立websocket连接，通知api是总线服务器的客户端，收到新的电子邮件后，storage会将它的通知发送到总线，并将总线发送给其subscribers。api确定发送接收通知的连接，并将其发送到用户的浏览器。

  这里我们将讨论API或Websocket服务器，最后我会告诉你这个服务器能够保持三百万的在线连接。



## 常见方式

  我们先来看在没有任何优化的情况下使用Go功能实现服务器的某个部分。在使用`net/http`	之前，先来看看如何去接受和发送数据。注意，基于WebSocket协议的数据(例如JSON对象)在上下文中被称为packets(分组)。

### Channel struct

  先来实现`Channel`，它包含通过WebSocket连接发送和接受此类数据包的逻辑结构

```go
// Packet represents application level data.
type Packet struct {
    
}

// Channel wraps user connection.
type Channel struct {
    conn net.Conn    // WebSocket connection
    send chan Packet // Outgoing packets queue
}

func NewChannel(conn net.Conn) *Channel {
    c := &Channel{
        conn: conn,
        send: make(chan Packet, N),
    }
    go c.reader()
    go c.writer()
	return c
}
```

  这里有个信息需要重视，也就是这两个reader/writer的goroutine，每一个goroutine需要自己的内存栈，初始大小为2~8KB，取决于操作系统和Go版本。根据上面提到的三百万在线连接的数量，我们需要24GB的内存(设堆栈为4KB)来用于存储所有连接，这里甚至还没有为Channel结构，以及传出数据库包`ch.send`和其他内部字段分配内存。可见问题比较大。



### I/O goroutine

  我们来看看 `reader` 的实现：

```go
func (c *Channel) reader() {
    // We make a buffered read to reduce read syscalls.
    buf := bufio.NewReader(c.conn)
    for {
        pkt, _ := readPacket(buf)
        c.handle(pkt)
    }
}
```

   这里我们使用 `bufio.Reader` 来减少 `read()` 系统调用的数量，并读取 `buf` 缓冲区大小允许的数量。在无限循环中，我们_期待新数据的到来_。注意：是_期待新数据的到来_，我们一会儿再仔细讨论这一点。

  我们不考虑传入数据包的解析和处理，因为它对我们将讨论的优化并不重要。但是，`buf` 现在值得我们注意：默认情况下，它为 4KB，这意味着我们的连接还剩余 12 GB 内存没有使用。同样的，我们可以实现 `writer`：

```go
func (c *Channel) writer() {
    // we make buffered write to reduce write syscalls.
    buf := bufio.NewWriter(c.conn)
    
    for pkt := range c.send {
        _ := writePacket(buf, pkt)
        buf.Flush()
    }
}
```

 

### HTTP

  我们已经写好了一个简单的 `Channel` 实现，现在我们需要制造一个 WebSocket 连接来协同工作。由于我们任然处于_常见做法_一节中，因此我们不妨也用常见的方式来完成。

  注意：如果你不知道 WebSocket 的工作原理，值得一提的就是客户端通过一个特殊的 HTTP Upgrade 机制来切换到 WebSocket 协议。成功处理 Upgrade 请求后，服务器和客户端将使用 TCP 连接来交换 Websocket 的二进制帧。[这里](https://tools.ietf.org/html/rfc6455#section-5.2) 给出了连接内帧结构的描述。

```GO
import (
    "net/http"
    "some/websocket"
)

http.HandleFunc("/v1/ws", func(w http.ResponseWriter, r *http.Request) {
    conn, _ := websocket.Upgrade(r, w)
    ch := NewChannel(conn)
    // ...
})
```

  请注意，`http.ResponseWriter` 会为 `bufio.Reader` 和 `bufio.Writer` 分配内存（各需要 4KB 的缓存）来初始化 `*http.Request` 和之后的响应写入。

  无论使用哪种 WebSocket 库，在成功响应 Upgrade 请求后，在 `responseWriter.Hijack()` 调用后[服务器会收到](https://github.com/golang/go/blob/143bdc27932451200f3c8f4b304fe92ee8bba9be/src/net/http/server.go#L1862-L1869) IO 缓存和 TCP 连接。

  提示：在某些情况下，`go:linkname` 可以使用 `net/http.putBufio{Read,Writer}` 将缓存返回给 `net/http` 内部的 `sync.Pool` 。

  因此，我们还需要 24 GB 内存来支撑三百万的链接。

  终上所述，我们需要 72GB 内存来支撑一个什么都还没做的应用。



## 优化

  我们来回顾一下我们介绍部分中讨论的内容，并记住用户连接的行为方式。切换到 WebSocket 后，客户端发送包含相关事件的数据包，或者说订阅事件。然后（不考虑诸如技术消息 `ping/pong`），客户端可以在整个生命周期中不发送任何其他内容。连接寿命可能持续几秒到几天。

  因此对于大多数的时间来说，我们的 `Channel.reader()` 和 `Channel.writer()` 在等待数据的处理用于接受或发送。与他们一起等待的是每个 4KB 的 IO 缓存。



### Netpoller

  

