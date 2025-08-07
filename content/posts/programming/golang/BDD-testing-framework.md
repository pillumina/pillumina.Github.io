---
title: "BDD: Ginkgo测试框架"
date: 2020-12-04T11:22:18+08:00
hero: /images/posts/golang2.jpg
menu:
  sidebar:
    name: Ginkgo 测试框架
    identifier: golang-behavior-driven-testing
    parent: golang odyssey
    weight: 10
draft: false
---

## Preface

  BDD和TDD都是test case first的实现，无非是把后者的test改成前者的behavior。在TDD中，关注的核心点是function，即认为程序最基本单元是function，其test case可以认为是unit test，TDD和unit test的区别是TDD强调测试和开发结合而成的工作流: 写test case -> 写代码 -> 通过测试，继续写更多测试，写一次循环。

  而BDD比TDD更关注高层的行为，而不是函数级别的行为，也就是在BDD中，不会强调函数的功能正确，这是unit test应该做的事。BDD更关注user story，即用户在特定场景，与软件交互发生的行为，这个behavior指的就是高层模块的行为。

  如何区分BDD和TDD，简单理解，TDD是给programmer的，用来验证开发者的最基本模块的功能：在什么输入，应该产生什么输出，保证实现的边界，健全性。而BDD，其test case描述的是更高级的模块行为，脱离了具体的实现，容易用自然语言去描述，也就是BDD是给product manager的，告诉其系统的行为。



## BDD in golang

​	实现的时候，我们需要把Given-When-Then这种story格式组织test case翻译为测试代码，通过一系列的assertion来检查实现是否符合test case的预期，我们完全可以直接通过golang自带的testing模块来实现，不过testing的功能有时候比较简陋，本文记录了用Ginkgo+Gomega来组织test case，让我们的测试语言更加接近自然语言。

   二者结合的目的是，ginkgo实现了test case的组织，并加入了其他方便的功能: 初始化，后续处理，异步等等。而gomega设计的目的是与ginkgo一起工作，实现易读的assertion(ginkgo中称为match)功能。

```
Gomega is ginkgo's preferred matcher library
```



## 初始化

  ginkgo依托golang原生testing框架，即可以用`go test ./..` 执行，也可以通过ginkgo binrary安装`go install github.com/onsi/ginkgo`，封装了ginkgo测试框架的各种feature。

  初始化首先进入待测试的package:

```
cd /path/to/package
```

  执行初始化:

```
ginkgo bootstrap
```

生成以suite_test.go文件，接下来向suite添加测试specs，生成比如ginkgo_cart package测试文件。

```
ginkgo generate ginkgo_cart
```



## 运行

  生成`ginkgo_cart_test.go`，注意测试文件在`ginkgo_cart_test`package， 需要import package `ginkgo_cart`，即BDD层级高于unit test, 不应该了解package内部的具体实现，测试package的外部接口即可。编写测试代码，运行`go test ./..`即可。



## Ginkgo Keyword

Ginkgo测试代码骨架由一系列keyword关联的闭包组成，常用的有：

1. Describe/Context/When: 测试逻辑块
2. BeforeEach/AfterEach/JustBeforeEach/JustAfterEach: 初始化测试用例块
3. It: 单一Spec，测试case

keyword的声明均为传入Body参数，比如Describe:

```go
Describe(text string, body func()) bool
```

一个样例：

```go
var _ = Describe("Nest Test Demo", func() {
	Context("MyTest level1", func() {
		BeforeEach(func() {
			fmt.Println("beforeEach level 1")
		})
		It("spec 3-1 in level1", func(){
			fmt.Println("sepc on level 1")
		})
		Context("MyTest level2", func() {
			BeforeEach(func() {
				fmt.Println("beforeEach level 2")
			})
			Context("MyTest level3", func() {
				BeforeEach(func() {
					fmt.Println("beforeEach level 3")
				})
				It("spec 3-1 in level3", func() {
					fmt.Println("A simple spec in level 3")
				})
				It("3-2 in level3", func() {
					fmt.Println("A simple spec in level 3")
				})
			})
		})
	})
})
```

### Describe, Context, When

这三种都称为Container，对于ginkgo属于同一类，只是名称不同

一般Describe用于最顶层：描述完整的测试场景，包含Context/When，而Context/When本身可以嵌套包含下级的Context/When。

三者组织成Tree结构：Describe是root, Context和When是普通的TreeNode。

三者包含的节点，除了自身，还包括其他keyword节点：BeforeEach, JustBeforeEach, It。

测试代码逻辑应该包含在BeforeEach, It等类别中，而不应该在container类别中体现。



### It

Ginkgo执行以It为基本单元，以定义的顺序执行，It一般包含Assertion逻辑: Expect(...)，即最终的测试结果和预期的比较，测试执行逻辑实现于BeforeEach, JustBeforeEach中



### BeforeEach, JustBeforeEach

BeforeEach声明于Container节点内部，container node每个child执行前都会执行BeforeEach，一般用来Setup test env：声明测试用例变量，初始化。

JustBeforeEach类似，区别是永远执行于BeforeEach之后：等从root到lt node所有BeforeEach执行完: 才再从root到lt node执行所有JustBeforeEach；一般实现测试执行逻辑：如request http，以便It node与expect比较。



### Demo code 示意

示例中各种节点的内部组成为如下tree：

![demo tree](https://raw.githubusercontent.com/eliteGoblin/images/master/blog/img/picgo/20200412194916.png)

运行示例可以得到:

```
beforeEach level 1
sepc 1-1 on level 1
•beforeEach level 1
beforeEach level 2
beforeEach level 3
Spec 3-1 in level 3
•beforeEach level 1
beforeEach level 2
beforeEach level 3
Spec 3-2 in level 3

```

我么可以得到一些结论:

1. 执行是以It node定义顺序执行
2. 每个It执行前，走了从root到It的path，顺序执行各context node的BeforeEach函数



## It 与 Matcher

购物车demo中，其中一个lt:

```go
Expect(cart.TotalItems()).To(Equal(3))
```

这种自然语言风格的assertion是由Ginkgo配套的Gomega实现的: expect返回封装了测试输出值的Assertion:

```go
func Expect(actual interface{}, extra ...interface{}) Assertion
```

Assertion是interface, 简化版本(为语义通顺，还包含几个类似function):

```go
type Assertion interface {
	To(matcher types.GomegaMatcher, optionalDescription ...interface{}) bool
	ToNot(matcher types.GomegaMatcher, optionalDescription ...interface{}) bool
}
```

`To`接收`GomegaMatcher`, 其封装了Expect value: Equal调用了Ginkgo的EqualMatcher.

```go
func Equal(expected interface{}) types.GomegaMatcher {
	return &matchers.EqualMatcher{
		Expected: expected,
	}
}
```

加上Assertion封装了实际value, 两者的比较可得出结论.而`ToNot`是`To`的相反情况.

如果想比较自定义的复杂类型: 可实现GomegaMatcher:

```go
加上Assertion封装了实际value, 两者的比较可得出结论.而ToNot是To的相反情况.

如果想比较自定义的复杂类型: 可实现GomegaMatcher:
```



## 其他features

Focus:

仅执行特定Node及之下的It: 在keyword之前加`F`: `FContext`, `FIt`, 但会使`go test`fail(返回 1), CI集成Ginkgo需注意.

Pending

与Focus相反: 不执行特定Node及之下的It. 在keyword之前加`X`.但默认不会使`go test` fail(若想让其fail, 加 —failOnPending)

Skip:

根据代码runtime结果决定是否跳过某It(Pending是编译时):

```go
It("spec 1-1 in level1", func(){
    if somecondition {
        Skip("special condition wasn't met")
    }
    fmt.Println("sepc 1-1 on level 1")
})
```

Skip仅能置于It之下，否则会Panic.

Eventually

测试异步逻辑: 如发送请求到队列, 需持续polling. 在Gomega实现:

```go
Eventually(func() []int {
    return thing.SliceImMonitoring
}, TIMEOUT, POLLING_INTERVAL).Should(HaveLen(2))

```

TIMTOUT为总超时时间, 默认１s;POLLING_INTERVAL为每次polling间隔, 默认10ms.

Ginkgo还支持benchmark及run in parallel, 可参考[Ginkgo doc](https://onsi.github.io/ginkgo/#parallel-specs)

