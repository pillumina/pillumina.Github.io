---
title: "Go编程模式：Visitor（k8s）"
date: 2021-03-31T11:22:18+08:00
hero: /images/posts/golang-banner-2.png
menu:
  sidebar:
    name: Go Visitor Pattern (k8s)
    identifier: go-pattern-visitor
    parent: programming pattern
    weight: 10
draft: false
---



## 概述

最近在看kubernetes的`kubectl`部分源码，记录一下其中用到的visitor编程模式(实际上`kubectl`主要用到了builder和visitor)。visitor模式是将**算法和操作对象结构分离**的一种方法。换句话说，这样的分离能够在不修改对象结构的情况下向原有对象新增操作，是符合开闭原则的。这个文章以一些例子去讨论`kubectl`中到底如何玩的。



## 从一个例子出发

写一个简单的Visitor模式示例：

- 我们的代码中有一个`Visitor`的函数定义，还有一个`Shape`接口，其需要使用 `Visitor`函数做为参数
- 我们的实例的对象 `Circle`和 `Rectangle`实现了 `Shape` 的接口的 `accept()` 方法，这个方法就是等外面给我传递一个Visitor。

```go
package main
import (
    "encoding/json"
    "encoding/xml"
    "fmt"
)
type Visitor func(shape Shape)
type Shape interface {
    accept(Visitor)
}
type Circle struct {
    Radius int
}
func (c Circle) accept(v Visitor) {
    v(c)
}
type Rectangle struct {
    Width, Heigh int
}
func (r Rectangle) accept(v Visitor) {
    v(r)
}
```



然后，我们实现两个Visitor，一个是用来做JSON序列化的，另一个是用来做XML序列化的:

```python
func JsonVisitor(shape Shape) {
    bytes, err := json.Marshal(shape)
    if err != nil {
        panic(err)
    }
    fmt.Println(string(bytes))
}
func XmlVisitor(shape Shape) {
    bytes, err := xml.Marshal(shape)
    if err != nil {
        panic(err)
    }
    fmt.Println(string(bytes))
}
```

下面是我们的使用Visitor这个模式的代码：

```python
func main() {
  c := Circle{10}
  r :=  Rectangle{100, 200}
  shapes := []Shape{c, r}
  for _, s := range shapes {
    s.accept(JsonVisitor)
    s.accept(XmlVisitor)
  }
}
```

写这些代码的目的是为了解耦数据结构和算法，其实用`Strategy`模式也可以做到，在模式上也更简单点。但是需要注意的一点：**在有些情况下，多个Visitor是来访问一个数据结构的不同部分，这种情况下，数据结构有点像一个数据库，而各个Visitor会成为一个个小应用**。那么`kubectl`无疑是这样的场景。



## k8s一些背景

- 在博客的`kubernetes & docker`的专栏里，介绍了k8s的一些基本知识。其实对于k8s来说，其抽象出了很多资源Resource：Pod，ReplicaSet，ConfigMap，Volumes，Namespace, Roles...等等。而这些构成了k8s的数据模型( [Kubernetes Resources 地图](https://github.com/kubernauts/practical-kubernetes-problems/blob/master/images/k8s-resources-map.png))

- `kubectl`为k8s的客户端命令，其对接Kubernetes API Server，开发和运维通过此去和k8s进行交互。而API Server则联系到每个节点的`kubelet`控制每个节点。
- `kubectl`主要的工作就是处理用户提交的例如：命令行参数、yaml/json文件等。将用户提交的这些组织成数据结构体，发送给API Server。
- 源码：`src/k8s.io/cli-runtime/pkg/resource/visitor.go` ([链接](https://github.com/kubernetes/kubernetes/blob/cea1d4e20b4a7886d8ff65f34c6d4f95efcb4742/staging/src/k8s.io/cli-runtime/pkg/resource/visitor.go))

当然`kubectl`的源码复杂，用简单的话阐述其基本原理就是：**它从命令行和yaml文件中获取信息，通过Builder模式并把其转成一系列的资源，最后用 Visitor 模式模式来迭代处理这些Reources**。

我先用一个小的例子来说明，忽略掉很多复杂的代码逻辑



## `kubectl`的实现

### Visitor模式的定义

首先，`kubectl` 主要是用来处理 `Info`结构体，下面是相关的定义：

```go
type VisitorFunc func(*Info, error) error
type Visitor interface {
    Visit(VisitorFunc) error
}
type Info struct {
    Namespace   string
    Name        string
    OtherThings string
}
func (info *Info) Visit(fn VisitorFunc) error {
  return fn(info, nil)
}
```

上述拆解一下：

- 有一个`VisitorFunc`函数类型的定义
- `Visitor`接口，需要实现一个`Visit(VisitorFunc) error`的方法
- 最后，为`Info`实现`Visitor`接口中的`Visit()`方法，其就是直接调用传进来的`fn`

接下来再定义几种不同类型的Visitor

### Name Visitor

这个Visitor 主要是用来访问 `Info` 结构中的 `Name` 和 `NameSpace` 成员

```go
type NameVisitor struct {
  visitor Visitor
}
func (v NameVisitor) Visit(fn VisitorFunc) error {
  return v.visitor.Visit(func(info *Info, err error) error {
    fmt.Println("NameVisitor() before call function")
    err = fn(info, err)
    if err == nil {
      fmt.Printf("==> Name=%s, NameSpace=%s\n", info.Name, info.Namespace)
    }
    fmt.Println("NameVisitor() after call function")
    return err
  })
}
```

拆解代码，可以看到:

- 声明了一个`NameVistor`结构体，多态得加了一个`Visitor`接口成员
- 实现`Visit()`方法时，调用内部`Visitor`的`Visit()`方法，这也是一种修饰器模式。

### Other Visitor

这个Visitor主要用来访问 `Info` 结构中的 `OtherThings` 成员

```go
type OtherThingsVisitor struct {
  visitor Visitor
}
func (v OtherThingsVisitor) Visit(fn VisitorFunc) error {
  return v.visitor.Visit(func(info *Info, err error) error {
    fmt.Println("OtherThingsVisitor() before call function")
    err = fn(info, err)
    if err == nil {
      fmt.Printf("==> OtherThings=%s\n", info.OtherThings)
    }
    fmt.Println("OtherThingsVisitor() after call function")
    return err
  })
}
```



### Log Visitor

```go
type LogVisitor struct {
  visitor Visitor
}
func (v LogVisitor) Visit(fn VisitorFunc) error {
  return v.visitor.Visit(func(info *Info, err error) error {
    fmt.Println("LogVisitor() before call function")
    err = fn(info, err)
    fmt.Println("LogVisitor() after call function")
    return err
  })
}
```



### 如何使用

```go
func main() {
  info := Info{}
  var v Visitor = &info
  v = LogVisitor{v}
  v = NameVisitor{v}
  v = OtherThingsVisitor{v}
  loadFile := func(info *Info, err error) error {
    info.Name = "Hao Chen"
    info.Namespace = "MegaEase"
    info.OtherThings = "We are running as remote team."
    return nil
  }
  v.Visit(loadFile)
}

```

拆解上述代码：

- Visitor为嵌套式的
- `LoadFile`模拟读取文件数据
- 最后一条`v.Visit()`激活上述流程

上述的代码输出如下:

```shell
LogVisitor() before call function
NameVisitor() before call function
OtherThingsVisitor() before call function
==> OtherThings=We are running as remote team.
OtherThingsVisitor() after call function
==> Name=Hao Chen, NameSpace=MegaEase
NameVisitor() after call function
LogVisitor() after call function
```

我们可以看到，这种做法实现了几点功能：

1. 解耦了数据和算法程序
2. 使用修饰器模式
3. 有pipeline模式的味道

我们接下来再以修饰器模式重构下上述代码



### Visitor修饰器

```go
type DecoratedVisitor struct {
  visitor    Visitor
  decorators []VisitorFunc
}
func NewDecoratedVisitor(v Visitor, fn ...VisitorFunc) Visitor {
  if len(fn) == 0 {
    return v
  }
  return DecoratedVisitor{v, fn}
}
// Visit implements Visitor
func (v DecoratedVisitor) Visit(fn VisitorFunc) error {
  return v.visitor.Visit(func(info *Info, err error) error {
    if err != nil {
      return err
    }
    if err := fn(info, nil); err != nil {
      return err
    }
    for i := range v.decorators {
      if err := v.decorators[i](info, nil); err != nil {
        return err
      }
    }
    return nil
  })
}
```

上述代码，实际上做了以下几点事情:

- 以`DecoratedVisitor`结构存放所有的`VisitorFunc`
- `NewDecoratedVisitor`把所有的`VisitorFunc`传进去，构造`DecoratedVisitor`对象
- `DecoratedVisitor`实现了`Visit()`方法，里面实际上就是个for-loop，以非嵌套的方式调用所有的`VisitorFunc`

所以我们可以这么使用这个重构：

```go
info := Info{}
var v Visitor = &info
v = NewDecoratedVisitor(v, NameVisitor, OtherVisitor)
v.Visit(LoadFile)
```

这样看上去能简单很多。

基本上如果读懂了上述的逻辑，`kubectl`的代码也差不多能看明白。