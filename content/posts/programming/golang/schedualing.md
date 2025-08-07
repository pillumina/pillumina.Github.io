---

title: "Golang并发调度"
date: 2020-12-17T11:22:18+08:00
hero: /images/posts/golang2.jpg
menu:
  sidebar:
    name: Golang并发调度
    identifier: golang-schedualing
    parent: golang odyssey
    weight: 10
draft: false
---

*性能提升不会凭空出现，它总是伴随着代码复杂度的上升。*
*The performance improvement does not materialize from the air, it comes with code complexity increase.*

*-- Dmitry Vyukov*



  Go 语言的调度器我认为应该是整个运行时最有趣的组件了。对于Go本身，它的设计和实现直接牵动了Go运行时的其他组件，也是和用户态代码直接打交道的部分；对于Go用户而言，调度器将其极为复杂的运行机制隐藏在了简单的关键字`go`下。为了保证高性能，调度器必须有效得利用计算的并行性和局部性原理；为了保证用户态的简洁，调度器必须高效得对调度用户态不可见的网络轮训器、垃圾回收器进行调度；为了保证代码执行的正确性，必须严格实现用户态代码的内存顺序等。总而言之，调度器的设计直接决定了Go运行时源码的表现形式。



## 设计原理



## 数据结构: MPG

## 调度器启动

## 创建Goroutine

## 调度循环

## 触发调度

## 线程管理

## 总结



