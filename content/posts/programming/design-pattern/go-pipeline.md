---
title: "Go编程模式：Pipeline"
date: 2021-02-03T11:22:18+08:00
hero: /images/posts/golang-banner-2.png
menu:
  sidebar:
    name: Go Pipeline Pattern
    identifier: go-pattern-pipeline
    parent: programming pattern
    weight: 10
draft: false
---



## 概述

  这篇文章介绍Go编程里的Pipeline模式。如果是对Unix/Linux命令行熟悉的人会知道，Pipeline其实就是把每个命令拼接起来完成一个组合功能的技术。当下诸如流式处理，函数式编程，以及应用Gateway对微服务进行简单API编排，其实都受pipeline技术方式的影响。换句话说，这种技术能够很容易得把代码按照`单一职责`的原则拆分成多个`高内聚低耦合`的小模块，然后拼装起来去完成比较复杂的功能。

​    