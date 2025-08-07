---
title: "书单记录"
date: 2020-12-19T12:40:29+08:00
hero: /images/posts/hero-google.svg
menu:
  sidebar:
    name: Tracking
    identifier: tracking
    weight: 10
draft: false
---

`这个post为记录目前正在阅读与研究的section`

### Go语言设计

[Go语言设计与实现](https://draveness.me/golang/)

[Go Under The Hood](https://golang.design/under-the-hood/)

```
这两本在写作目的和内容规划都是一致的，不过第二个原本不再维护内容，作者开了下面的新的项目，把撰写原本而积累的与Go相关的资源进行了重新的整理。
```

[Go设计历史]([golang.design/history](https://changkun.de/s/go-history))



## pprof对服务端性能影响的研究

考虑一些极端场景，比如极度追求性能，压榨系统资源以及技术栈必须是Go的业务场景下，是否能自己构建Reactor网络模型

## GRPC框架对服务侧性能的影响



## Russ Cox正则表达式系列

*You should not be permitted to write production code if you do not have an journeyman license in regular expressions or floating point math. -- Rob Pike*

[Regular Expression Matching Can Be Simple And Fast](https://swtch.com/~rsc/regexp/regexp1.html)

[编译器词法分析:正则语言和正则表达式](https://www.cnblogs.com/Ninputer/archive/2011/06/08/2075714.html)



## Go内存原理与调度模型

正在整理专栏



## Bound Checking Elimination



## Crafting Interpreter

时常看PL和Compiler的基础

[crafting interpreters](https://craftinginterpreters.com/contents.html)



## Kosaraju's Algorithm

看William Lin的coding interview觉得用来处理树和图很好，算法4里也有



## Heilmeier问题系列

思考某篇paper的选题

1. **What are you trying to do?** Articulate your objectives using absolutely no jargon.
2. **How is it done today, and what are the limits of current practice?**
3. **Who cares?** [Support other’s research? Shape research landscape? Power applications in industry?]
4. **What's new in your approach** and why do you think it will be successful?
5. If you're successful, **what difference will it make?** [e.g. Contributions in theory/modeling? Improve accuracy by 5% on dataset A, B, C…?]
6. **What are the risks and the payoffs?** [Further, how would you mitigate the risks? If your proposed method does not work, what could be alternative design? These can end up as discussions such as ablation studies in your paper.]
7. **How much will it cost?** [e.g. How many GPUs do your experiments require? How long is each training process? How about data storage?]
8. **How long will it take?** [How many hours are you going to work on this per week? When is the submission DDL? Can you make it?]
9. **What are the midterm and final "exams" to check for success?**

