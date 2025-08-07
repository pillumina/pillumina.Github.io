---
title: "Golang TDD"
date: 2020-12-19T11:22:18+08:00
hero: /images/posts/golang2.jpg
menu:
  sidebar:
    name: Golang Testing Kits
    identifier: golang-tdd
    parent: golang odyssey
    weight: 10
draft: false
---

## Preface

本文整理golang编码的单元测试常用示例，以及TDD的简要流程。



## 单元测试基础

单元测试文件以`_test.go`结尾，需要记住以下原则：

- 文件名必须是`_test.go`结尾的，这样在执行`go test`的时候才会执行到相应的代码
- 你必须import `testing`这个包
- 所有的测试用例函数必须是`Test`开头
- 测试用例会按照源代码中写的顺序依次执行
- 测试函数`TestXxx()`的参数是`testing.T`，我们可以使用该类型来记录错误或者是测试状态
- 测试格式：`func TestXxx (t *testing.T)`,`Xxx`部分可以为任意的字母数字的组合，但是首字母不能是小写字母[a-z]，例如`Testintdiv`是错误的函数名。
- 函数中通过调用`testing.T`的`Error`, `Errorf`, `FailNow`, `Fatal`, `FatalIf`方法，说明测试不通过，调用`Log`方法用来记录测试的信息。



### Table-Driven-Testing

测试讲究 case 覆盖，当我们要覆盖更多 case 时，显然通过修改代码的方式很笨拙。这时我们可以采用 Table-Driven 的方式写测试，标准库中有很多测试是使用这种方式写的。

```go
func TestFib(t *testing.T) {
    var fibTests = []struct {
        in       int // input
        expected int // expected result
    }{
        {1, 1},
        {2, 1},
        {3, 2},
        {4, 3},
        {5, 5},
        {6, 8},
        {7, 13},
    }

    for _, tt := range fibTests {
        actual := Fib(tt.in)
        if actual != tt.expected {
            t.Errorf("Fib(%d) = %d; expected %d", tt.in, actual, tt.expected)
        }
    }
}
```

由于我们使用的是 `t.Errorf`，即使其中某个 case 失败，也不会终止测试执行。

### T类型

单元测试中，传递给测试函数的参数是 `*testing.T` 类型。它用于管理测试状态并支持格式化测试日志。测试日志会在执行测试的过程中不断累积，并在测试完成时转储至标准输出。

当测试函数返回时，或者当测试函数调用 `FailNow`、 `Fatal`、`Fatalf`、`SkipNow`、`Skip`、`Skipf` 中的任意一个时，则宣告该测试函数结束。跟 `Parallel` 方法一样，以上提到的这些方法只能在运行测试函数的 goroutine 中调用。

至于其他报告方法，比如 `Log` 以及 `Error` 的变种， 则可以在多个 goroutine 中同时进行调用。



### 报告方式

上面提到的系列包括方法，带 `f` 的是格式化的，格式化语法参考 `fmt` 包。

T 类型内嵌了 common 类型，common 提供这一系列方法，我们经常会用到的（注意，这里说的测试中断，都是指当前测试函数）：

1）当我们遇到一个断言错误的时候，标识这个测试失败，会使用到：

```
Fail : 测试失败，测试继续，也就是之后的代码依然会执行
FailNow : 测试失败，测试中断
```

在 `FailNow` 方法实现的内部，是通过调用 `runtime.Goexit()` 来中断测试的。

2）当我们遇到一个断言错误，只希望跳过这个错误，但是不希望标识测试失败，会使用到：

```
SkipNow : 跳过测试，测试中断
```

在 `SkipNow` 方法实现的内部，是通过调用 `runtime.Goexit()` 来中断测试的。

3）当我们只希望打印信息，会用到 :

```
Log : 输出信息
Logf : 输出格式化的信息
```

注意：默认情况下，单元测试成功时，它们打印的信息不会输出，可以通过加上 `-v` 选项，输出这些信息。但对于基准测试，它们总是会被输出。

4）当我们希望跳过这个测试，并且打印出信息，会用到：

```
Skip : 相当于 Log + SkipNow
Skipf : 相当于 Logf + SkipNow
```

5）当我们希望断言失败的时候，标识测试失败，并打印出必要的信息，但是测试继续，会用到：

```
Error : 相当于 Log + Fail
Errorf : 相当于 Logf + Fail
```

6）当我们希望断言失败的时候，标识测试失败，打印出必要的信息，但中断测试，会用到：

```
Fatal : 相当于 Log + FailNow
Fatalf : 相当于 Logf + FailNow
```



### Parallel并行测试

这里简单测试一个对Map的读写并行测试。**注意：Parallel方法表示只与其他带有Parallel方法的测试并行进行测试。**

```go
var (
    data   = make(map[string]string)
    locker sync.RWMutex
)

func WriteToMap(k, v string) {
    locker.Lock()
    defer locker.Unlock()
    data[k] = v
}

func ReadFromMap(k string) string {
    locker.RLock()
    defer locker.RUnlock()
    return data[k]
}
```

测试用例：

```go
var pairs = []struct {
    k string
    v string
}{
    {"polaris", "calvin1"},
    {"studygolang", "oops1"},
    {"stdlib", "go demo1"},
    {"polaris1", "calvin2"},
    {"studygolang1", "oops2"},
    {"stdlib1", "go demo2"},
    {"polaris2", " calvin3"},
}

// 注意 TestWriteToMap 需要在 TestReadFromMap 之前
func TestWriteToMap(t *testing.T) {
    t.Parallel()
    for _, tt := range pairs {
        WriteToMap(tt.k, tt.v)
    }
}

func TestReadFromMap(t *testing.T) {
    t.Parallel()
    for _, tt := range pairs {
        actual := ReadFromMap(tt.k)
        if actual != tt.v {
            t.Errorf("the value of key(%s) is %s, expected: %s", tt.k, actual, tt.v)
        }
    }
}
```

试验步骤：

1. 注释掉 WriteToMap 和 ReadFromMap 中 locker 保护的代码，同时注释掉测试代码中的 t.Parallel，执行测试，测试通过，即使加上 `-race`，测试依然通过；
2. 只注释掉 WriteToMap 和 ReadFromMap 中 locker 保护的代码，执行测试，测试失败（如果未失败，加上 `-race` 一定会失败）；

如果代码能够进行并行测试，在写测试时，尽量加上 Parallel，这样可以测试出一些可能的问题。



### 子测试与子基准测试(Run)

Go1.7开始引入的特性，即能够执行嵌套测试，对于过滤执行特性测试用例非常有用。

T 和 B 的 `Run` 方法允许定义子单元测试和子基准测试，而不必为它们单独定义函数。这便于创建基于 Table-Driven 的基准测试和层级测试。它还提供了一种共享通用 `setup` 和 `tear-down` 代码的方法：

```go
func TestFoo(t *testing.T) {
    // <setup code>
    t.Run("A=1", func(t *testing.T) { ... })
    t.Run("A=2", func(t *testing.T) { ... })
    t.Run("B=1", func(t *testing.T) { ... })
    // <tear-down code>
}
```

每个子测试和子基准测试都有一个唯一的名称：由顶层测试的名称与传递给 `Run` 的名称组成，以斜杠分隔，并具有可选的尾随序列号，用于消除歧义。

命令行标志 `-run` 和 `-bench` 的参数是非固定的正则表达式，用于匹配测试名称。对于由斜杠分隔的测试名称，例如子测试的名称，它名称本身即可作为参数，依次匹配由斜杠分隔的每部分名称。因为参数是非固定的，一个空的表达式匹配任何字符串，所以下述例子中的 “匹配” 意味着 “顶层/子测试名称包含有”：

```
go test -run ''      # 执行所有测试。
go test -run Foo     # 执行匹配 "Foo" 的顶层测试，例如 "TestFooBar"。
go test -run Foo/A=  # 对于匹配 "Foo" 的顶层测试，执行其匹配 "A=" 的子测试。
go test -run /A=1    # 执行所有匹配 "A=1" 的子测试。
```

子测试也可用于程序**并行控制**。只有子测试全部执行完毕后，父测试才会完成。在下述例子中，所有子测试之间并行运行，此处的 “并行” 只限于这些子测试之间，并不影响定义在其他顶层测试中的子测试：

```go
func TestGroupedParallel(t *testing.T) {
    for _, tc := range tests {
        tc := tc // capture range variable
        t.Run(tc.Name, func(t *testing.T) {
            t.Parallel()
            ...
        })
    }
}
```

在所有子测试并行运行完毕之前，`Run` 方法不会返回。下述例子提供了一种方法，用于在子测试并行运行完毕后清理资源：

```go
func TestTeardownParallel(t *testing.T) {
    // This Run will not return until the parallel tests finish.
    t.Run("group", func(t *testing.T) {
        t.Run("Test1", parallelTest1)
        t.Run("Test2", parallelTest2)
        t.Run("Test3", parallelTest3)
    })
    // <tear-down code>
}
```



### Test Coverage

测试覆盖率，这里讨论的是基于代码的测试覆盖率。

Go 从 1.2 开始，引入了对测试覆盖率的支持，使用的是与 cover 相关的工具（`go test -cover`、`go tool cover`）。虽然 `testing` 包提供了 cover 相关函数，不过它们是给 cover 的工具使用的。

关于测试覆盖率的更多信息，可以参考官方的博文：[The cover story](https://blog.golang.org/cover)



### gotest变量(参考)

gotest 的变量有这些：

- test.short : 一个快速测试的标记，在测试用例中可以使用 testing.Short() 来绕开一些测试
- test.outputdir : 输出目录
- test.coverprofile : 测试覆盖率参数，指定输出文件
- test.run : 指定正则来运行某个 / 某些测试用例
- test.memprofile : 内存分析参数，指定输出文件
- test.memprofilerate : 内存分析参数，内存分析的抽样率
- test.cpuprofile : cpu 分析输出参数，为空则不做 cpu 分析
- test.blockprofile : 阻塞事件的分析参数，指定输出文件
- test.blockprofilerate : 阻塞事件的分析参数，指定抽样频率
- test.timeout : 超时时间
- test.cpu : 指定 cpu 数量
- test.parallel : 指定运行测试用例的并行数



### gotest结构体(参考)

- B : 压力测试
- BenchmarkResult : 压力测试结果
- Cover : 代码覆盖率相关结构体
- CoverBlock : 代码覆盖率相关结构体
- InternalBenchmark : 内部使用的结构体
- InternalExample : 内部使用的结构体
- InternalTest : 内部使用的结构体
- M : main 测试使用的结构体
- PB : Parallel benchmarks 并行测试使用的结构体
- T : 普通测试用例
- TB : 测试用例的接口



## 压力测试基础

压测检测函数(方法)的性能，和编写UT类似，所以不再赘述，但需要注意以下几点：

- 压力测试用例必须遵循如下格式，其中XXX可以是任意字母数字的组合，但是首字母不能是小写字母

```
	func BenchmarkXXX(b *testing.B) { ... }
```

- `go test`不会默认执行压力测试的函数，如果要执行压力测试需要带上参数`-test.bench`，语法:`-test.bench="test_name_regex"`,例如`go test -test.bench=".*"`表示测试全部的压力测试函数
- 在压力测试用例中,请记得在循环体内使用`testing.B.N`,以使测试可以正常的运行
- 文件名也必须以`_test.go`结尾

下面是一个压测的例子，测试除法函数的性能：

```go
package gotest

import (
	"testing"
)

func Benchmark_Division(b *testing.B) {
	for i := 0; i < b.N; i++ { //use b.N for looping 
		Division(4, 5)
	}
}

func Benchmark_TimeConsumingFunction(b *testing.B) {
	b.StopTimer() //调用该函数停止压力测试的时间计数

	//做一些初始化的工作,例如读取文件数据,数据库连接之类的,
	//这样这些时间不影响我们测试函数本身的性能

	b.StartTimer() //重新开始时间
	for i := 0; i < b.N; i++ {
		Division(4, 5)
	}
}
```

我们执行命令`go test webbench_test.go -test.bench=".*"`，可以看到如下结果：

```
Benchmark_Division-4   	                     500000000	      7.76 ns/op	     456 B/op	      14 allocs/op
Benchmark_TimeConsumingFunction-4            500000000	      7.80 ns/op	     224 B/op	       4 allocs/op
PASS
ok  	gotest	9.364s
```

上面的结果显示我们没有执行任何`TestXXX`的单元测试函数，显示的结果只执行了压力测试函数，第一条显示了`Benchmark_Division`执行了500000000次，每次的执行平均时间是7.76纳秒，第二条显示了`Benchmark_TimeConsumingFunction`执行了500000000，每次的平均执行时间是7.80纳秒。最后一条显示总共的执行时间。



## 性能测试进阶(benchstat)

### sync.Map优化例子

在sync.Map中存储一个值，然后再并发删除该值:

```go
func BenchmarkDeleteCollision(b *testing.B){
  benchMap(b, bench{
    setup: func(_ *testing.B, m mapInterface){m,LoadOrStore(0, 0)},
    perG: func(b *testing.B, pb *testing.PB, i int, m mapInterface){
      for; pb.Next(); i++ {m.Delete(0)}
    }
  })
}
```

```
优化 src/sync/map.go
275 -delete(m.dirty, key)
275 +e, ok = m.dirty[key]
276 +m.misslocked()
```

```
$ git stash
$ git test -run=none -bench=BenchmarkDeleteCollision -count=20 | tee old.txt
$ git stash pop
$ git test -run=none -bench=BenchmarkDeleteCollision -count=20 | tee new.txt
$ benchstat old.txt new.ext
```



### 编译器优化例子

查看编译器优化，测试函数被编译成了什么

```go
package compile

func comp1(s1, s2 []byte)bool{
  return string(s1) == string(s2)
}

func comp2(s1, s2 []byte)bool{
  return conv(s1) == conv(s2)
}

func conv(s []byte) string{
  return string(s)
}
```

```
$GOSSAFUNC=com1 go build 
// 会生成ssa.html，open它即可看到comp1函数编译后的代码
```



### 假设性检验

- 统计是一套在总体分布函数完全未知或者只知道形式、不知道参数的情况下，为了由样本推断总体的某些未知特性，形成的一套方法论。
- 多次抽样：对同一个性能基准测试运行多次，根据中心极限定理，如果理论均值存在，则抽样噪声服从正态分布。
- 当重复执行完某个性能基准测试后，benchstat先帮我们剔除掉了一些异常值，我们得到了关于某段代码在可控的环境条件E下的性能分布的一组样本。
- T检验：参数检验，假设数据服从正态分布，且方差相同 (最严格)
- Welch T检验(ttest)： 参数检验，假设服从正态分布，但方差不一定相同
- Mann-Whitney U检验(utest， benchstat的default):  非参数检验，假设最少，最通用，值假设两组样本来自于同一个总体（例如两个性能测试是否在同一个机器跑的），只有均值的差异。当对数据的假设减少时，结论的不确定性增大，p值会因此增大，进而使得性能基准测试的条件更加严格。



### 局限和应对

`perflock`降低系统噪音，作用是限制CPU时钟频率，从而一定程度上消除系统对性能测试程序的影响，仅支持Linux。

```
$ go get github.com/aclements/perflock/cmd/perflock
$ sudo install $GOPATH/bin/perflock /usr/bin/perflock
$ sudo -b perflock -daemon
$ perflock

$ perflock -governer 70% go test -test=none -bench=.
```





## Mocking

### GoMock

  GoMock为很常用的测试mock框架，虽然我自己不常用:0（因为我自身并不非常喜欢mock), 并且对在生产开发环境使用mock有点意见，代码增长（和Injection类似），以及如果不单独部署一个mock server很多修改并不能很好得share。

  虽然如此，这里还是记录一下GoMock的quick start。

#### Install

  首先就是安装`gomock`包，以及`mockgen`代码生成工具，后者其实并不是必要的，但是如果没有自己就要写一个容易出错并且繁琐的mock代码。

```
go get github.com/golang/mock/gomock
go get github.com/golang/mock/mockgen
```

  检查一下有没有成功，会打印一些使用帮助信息:

```
$GOPATH/bin/mockgen
```



#### 基本使用

  基本上使用`gomock`遵循以下几个步骤：

1. 使用`mockgen`去对你想要mock的interface生成mock对象
2. 在测试代码中，创建一个`gomock.Controller`实例，并且将其传入mock对象的constructor中获取一个mock对象
3. 在你的mock中调用`EXPECT()`去设置测试期望以及返回值
4. 在mock controller调用`FINISH()`去设置进行mock期望的assert（断言）

  下面记录一个小的demo展示上述的workflow，为了让展示简单，我们可以只是聚焦两个文件- 一个接口文件`doer.go`中的`Doer`接口（希望mock的），以及`user.go`文件中的结构体`User`，这个接口体用到了`Doer`接口。

  `doer.go`：

```go
package doer

type Doer interface {
    DoSomething(int, string) error
}
```

  `user.go`

```go
package user

import "github.com/sgreben/testing-with-gomock/doer"

type User struct {
    Doer doer.Doer
}

func (u *User) Use() error {
    return u.Doer.DoSomething(123, "Hello GoMock")
}
```

  下面是project的layout：

```
'-- doer
    '-- doer.go
'-- user
    '-- user.go
```

  我们接下来要在mocks文件夹内添加`Doer`的mock，并且新增一个`user_test.go`文件：

```go
'-- doer
    '-- doer.go
'-- mocks
    '-- mock_doer.go
'-- user
    '-- user.go
    '-- user_test.go
```

  为了生成这个`mock_doer.go`，我们创建mocks目录后调用：

```
mockgen -destination=mocks/mock_doer.go -package=mocks github.com/sgreben/testing-with-gomock/doer Doer
```

  这里的`mockgen`传入以下几个参数:

1. `-destination=mocks/mock_doer.go` 目标路径
2. `-package=mocks`：在`mocks`package内生成mocks
3. `github.com/sgreben/testing-with-gomock/doer`： 为这个package生成mocks (包名而已，根据实际情况定)
4. `Doer`: 为这个interface生成mocks，如果想要mock多个接口，可以传入以逗号分隔的列表`Doer1,Doer2`，对接口的声明必须清楚。

  *注意如果`$GOPATH/bin`不在`$PATH`中，`mockgen`要改成`$GOPATH/bin/mockgen`*

  最终`mockgen`会生成`mock_doer.go`这个文件：

```go
// Code generated by MockGen. DO NOT EDIT.
// Source: github.com/sgreben/testing-with-gomock/doer (interfaces: Doer)

package mocks

import (
	gomock "github.com/golang/mock/gomock"
)

// MockDoer is a mock of Doer interface
type MockDoer struct {
	ctrl     *gomock.Controller
	recorder *MockDoerMockRecorder
}

// MockDoerMockRecorder is the mock recorder for MockDoer
type MockDoerMockRecorder struct {
	mock *MockDoer
}

// NewMockDoer creates a new mock instance
func NewMockDoer(ctrl *gomock.Controller) *MockDoer {
	mock := &MockDoer{ctrl: ctrl}
	mock.recorder = &MockDoerMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use
func (_m *MockDoer) EXPECT() *MockDoerMockRecorder {
	return _m.recorder
}

// DoSomething mocks base method
func (_m *MockDoer) DoSomething(_param0 int, _param1 string) error {
	ret := _m.ctrl.Call(_m, "DoSomething", _param0, _param1)
	ret0, _ := ret[0].(error)
	return ret0
}

// DoSomething indicates an expected call of DoSomething
func (_mr *MockDoerMockRecorder) DoSomething(arg0, arg1 interface{}) *gomock.Call {
	return _mr.mock.ctrl.RecordCall(_mr.mock, "DoSomething", arg0, arg1)
}
```

  浏览一下代码，可以看到生成的`EXPECT()`方法和mock接口的方法在一个层级，这里是`DoSomething`，因为要避免名字冲突，所以这里把`EXPECT`定义成全大写。

  下面，我们在测试中创建一个*mock controller*。 mock controller的作用是跟踪以及对相关mocks对象的进行期望断言(asserting the expectations)。

  创建controller的方法就是，传入构建函数代表`*testing.T`的`t`，而后将其作为参数传入`Doer`mock对象的构建函数:

```go
mockCtrl := gomock.NewController(t)
defer mockCtrl.Finish()

mockDoer := mocks.NewMockDoer(mockCtrl)
```

  上述对`Finish`的defer后面再说。

  假设我们想要断言`mockerDoer`的`Do`方法将会被调用一次，传入`123`以及`Hello GoMock`作为参数并且返回`nil`。

  为了实现这个断言，我们在`mockDoer`对象上调用`EXPECT()`设置期望。`EXPECT()`其实返回的是一个`mock recorder`的对象，它包含了真实对象的所有同名方法。

  我们能够进行如下的链式调用:

```go
mockDoer.EXPECT().DoSomething(123, "Hello GoMock").Return(nil).Times(1)
```

  从这个调用其实你也能理解每个的意义，如果要设置方法被调用的次数，除了上述的`Times(number)`，还有诸如`MaxTimes(number)`以及`MinTimes(numbers)`这种显性的限制。

  看上去差不多了，接下来写一个完整的user_test.go`:

```go
package user_test

import (
  "github.com/sgreben/testing-with-gomock/mocks"
  "github.com/sgreben/testing-with-gomock/user"
)

func TestUse(t *testing.T) {
    mockCtrl := gomock.NewController(t)
    defer mockCtrl.Finish()

    mockDoer := mocks.NewMockDoer(mockCtrl)
    testUser := &user.User{Doer:mockDoer}

    // Expect Do to be called once with 123 and "Hello GoMock" as parameters, and return nil from the mocked call.
    mockDoer.EXPECT().DoSomething(123, "Hello GoMock").Return(nil).Times(1)

    testUser.Use()
}
```

  可能这个代码里对mock期望的断言并不明显，断言发生在defer掉的`Finish()`。相当于对`Finish`的调用发生在mock controller的声明的时候 - 这样我们不会忘记在后面加上期望断言。

  最后跑一下测试:

```
$ go test -v github.com/sgreben/testing-with-gomock/user
=== RUN   TestUse
--- PASS: TestUse (0.00s)
PASS
ok      github.com/sgreben/testing-with-gomock/user     0.007s
```

  当然如果你想构建多个mock对象，你可以对mock controller进行复用，它的`Finish`相当于会发生在所有和controller关联的mock对象的期望断言被设置之后。

  我们也可以测试一下mock方法的返回值，这里改写一下测试返回一个`dummyError`：

```go
func TestUseReturnsErrorFromDo(t *testing.T) {
    mockCtrl := gomock.NewController(t)
    defer mockCtrl.Finish()

    dummyError := errors.New("dummy error")
    mockDoer := mocks.NewMockDoer(mockCtrl)
    testUser := &user.User{Doer:mockDoer}

    // Expect Do to be called once with 123 and "Hello GoMock" as parameters, and return dummyError from the mocked call.
    mockDoer.EXPECT().DoSomething(123, "Hello GoMock").Return(dummyError).Times(1)

    err := testUser.Use()

    if err != dummyError {
        t.Fail()
    }
}
```



 #### 通过`go:generate`使用*GoMock*

  有些人可能发现一个workflow的问题，如果对每个package以及interface都用mockgen肯定是非常繁琐的，特别是如果我们开发的项目有大量的接口和包定义。为了解决这个问题，`mockgen`命令行能够被特殊的`go:generate`注释去替代。

  比如，在我们的例子里，我们能够在`doer.go`的`package`声明下面添加注释: 

```go
package doer

//go:generate mockgen -destination=../mocks/mock_doer.go -package=mocks github.com/sgreben/testing-with-gomock/doer Doer

type Doer interface {
    DoSomething(int, string) error
}
```

  但是这种写法也有个问题，因为代码文件目录和mocks目录的不一致，导致我们需要添加`../mocks`类似的路径而不是简单的`mocks/`，我们可以在项目的根路径下生成所有mocks:

```
go generate ./...
```

  写法上注意代码里`//`和`go:generate`之间没有空格。

  对于添加`go:generate`注释的原则以及一些mock的构建命名原则如下:

1. 每个包含需要mock的interfaces的文件中添加一个`go:generate`注释
2. 如果要用`mockgen`要传入清晰的interface名
3. 把mock文件放在`mocks`包下，名称改写`X.go`到`mocks/mock_X.go`

  

#### 使用参数匹配器

  有些情况下，你对mock中的特定参数不太关心，当然我们可以清楚地固定参数，也可以用参数匹配器去匹配参数，我们称之为`Matcher`，熟悉Ginkgo框架的同学应该很清楚。

  `GoMock`中预设了几个matchers：

1. `gomock.Any()`： 匹配所有类型、所有值
2. `gomock.Eq(x)`:  使用反射去匹配任何与`x`为`DeepEqual`的值
3. `gomock.Nil()`： 匹配`nil`
4. `gomock.Not(m)`:  这里`m`是一个Matcher，也就是匹配所有没有被`m`匹配的值
5. `gomock.Not(x)`:  这里`x`不是一个Matcher，匹配所有与`x`不`DeepEqual`的值

  举个例子，如果我们不关心`Do`方法的第一个参数:

```go
mockDoer.EXPECT().DoSomething(gomock.Any(), "Hello GoMock")
```

  `GoMock`会自动把非匹配类型的参数转化为`Eq`匹配器：

```go
mockDoer.EXPECT().DoSomething(gomock.Any(), gomock.Eq("Hello GoMock"))
```

  当然我们也可以自定义Matchers，实现接口就行, `gomock/matchers.go` :

```go
type Matcher interface {
    Matches(x interface{}) bool
    String() string
}
```

  这里的`Matches`方法是实例匹配发生的地方，`String`方法针对测试失败时生成human-readable的信息，我们可以自己写一个matcher去检查参数类型：

  `match/oftype.go`

```go
package match

import (
    "reflect"
    "github.com/golang/mock/gomock"
)

type ofType struct{ t string }

func OfType(t string) gomock.Matcher {
    return &ofType{t}
}

func (o *ofType) Matches(x interface{}) bool {
    return reflect.TypeOf(x).String() == o.t
}

func (o *ofType) String() string {
    return "is of type " + o.t
}
```

  然后我们就可以使用我们的matcher:

```go
// Expect Do to be called once with 123 and any string as parameters, and return nil from the mocked call.
mockDoer.EXPECT().
    DoSomething(123, match.OfType("string")).
    Return(nil).
    Times(1)
```

  注意下上述我们分行写，要把`.`写在行末尾，不然编译器会报错。



#### 断言调用顺序

  对一个对象的调用顺序也是很重要的，*GoMock*提供了`.After`方法显式地定义一个方法必须在另一个方法后面被调用:

```go
callFirst := mockDoer.EXPECT().DoSomething(1, "first this")
callA := mockDoer.EXPECT().DoSomething(2, "then this").After(callFirst)
callB := mockDoer.EXPECT().DoSomething(2, "or this").After(callFirst)
```

  这个代码都能理解。

  此外还提供了一个更直观的手段去定义断言顺序，也就是`gomock.InOrder`，这种写法更容易阅读:

```go
gomock.InOrder(
    mockDoer.EXPECT().DoSomething(1, "first this"),
    mockDoer.EXPECT().DoSomething(2, "then this"),
    mockDoer.EXPECT().DoSomething(3, "then this"),
    mockDoer.EXPECT().DoSomething(4, "finally this"),
)
```

 

#### 定义mock的actions

  本质上就是mock其实不会执行其他行为，我们可以人为使用`.Do`方法，并且传入调用的函数，意味着如果调用的参数匹配上了，就会执行`.Do`提供的函数：

```go
mockDoer.EXPECT().
    DoSomething(gomock.Any(), gomock.Any()).
    Return(nil).
    Do(func(x int, y string) {
        fmt.Println("Called with x =",x,"and y =", y)
    })
```

  一些复杂的动作，比如下面这个例子，`DoSomething`方法的第一个`int`参数应该小于或者等于第二个`string`参数的长度:

```go
mockDoer.EXPECT().
    DoSomething(gomock.Any(), gomock.Any()).
    Return(nil).
    Do(func(x int, y string) {
        if x > len(y) {
            t.Fail()
        }
    })
```

 **这种写法不能通过自定义matcher实现，因为我们关联了多个具体的值，而matcher每次只能访问一个参数。**



### sql-mock(GORM)

  常规的`database/sql/driver`的接口mocking可以用GoMock，但是像`gorm`之类的ORM框架就很难用常规的mock方法，以为有其他很多额外的苦力活。sql-mock的介绍为`Sql mock driver for golang to test database interactions. `可以帮助解决这个问题。

  下面用BDD框架`Ginkgo`写测试用例，展示一个如何使用`Sqlmock`去测试一个简单blog应用的例子，这个例子的后端为`pg`并且使用了`gorm`。

  [源码](https://github.com/dche423/dbtest)

#### 定义GORM数据模型与Repository

```go
// modle.go
import "github.com/lib/pq"
...
type Blog struct {
	ID        uint
	Title     string
	Content   string
	Tags      pq.StringArray // string array for tags
	CreatedAt time.Time
}


// repository.go
import "github.com/jinzhu/gorm"
...

type Repository struct {
	db *gorm.DB
}

func (p *Repository) ListAll() ([]*Blog, error) {
	var l []*Blog
	err := p.db.Find(&l).Error
	return l, err
}

func (p *Repository) Load(id uint) (*Blog, error) {
	blog := &Blog{}
	err := p.db.Where(`id = ?`, id).First(blog).Error
	return blog, err
}

...
```

  `Repository`结构非常简单，有着`*gorm.DB`字段，所有的DB操作依赖于此。这里为了简洁把一些多余的代码省略了。除了`Load`、`ListAll`当然还有类似`Save`、`Delete`、`SearchByTitle`等方法。

 #### 单元测试

```go
import (
	...
  
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/DATA-DOG/go-sqlmock"
	"github.com/jinzhu/gorm"
)

var _ = Describe("Repository", func() {
	var repository *Repository
	var mock sqlmock.Sqlmock

	BeforeEach(func() {
		var db *sql.DB
		var err error

		db, mock, err = sqlmock.New() // mock sql.DB
		Expect(err).ShouldNot(HaveOccurred())

		gdb, err := gorm.Open("postgres", db) // open gorm db
		Expect(err).ShouldNot(HaveOccurred())

		repository = &Repository{db: gdb}
	})
	AfterEach(func() {
		err := mock.ExpectationsWereMet() // make sure all expectations were met
		Expect(err).ShouldNot(HaveOccurred())
	})
  
	It("test something", func(){
	    ...
	})
})
```

  如果读者对`Ginkgo`的测试语法表示不熟悉的，可以去参阅posts里的`BDD`相关章节。在这里，`BeforeEach`中做一些测试初始化，例如`Repository`的实例化等。在`AfterEach`中加入各种断言。

  `BeforeEach`中的初始化分为几个步骤：

1. 创建`*sql.DB`的mock实例，利用`sqlmock.New()`创建mock控制器。
2. `gorm.Open("postgres", db)`使用GORM。
3. 创建`Repository`实例。

  在`AfterEach`中，我们使用`mock.ExpectationsWereMet()`确保所有的期望都被满足。

#### 测试ListAll方法

```go
// repository.go
...
func (p *Repository) ListAll() ([]*Blog, error) {
	var l []*Blog
	err := p.db.Find(&l).Error
	return l, err
}
...



// repository_test.go
...
Context("list all", func() {
	It("empty", func() {
		
		const sqlSelectAll = `SELECT * FROM "blogs"`
		
		mock.ExpectQuery(sqlSelectAll).
			WillReturnRows(sqlmock.NewRows(nil))

		l, err := repository.ListAll()
		Expect(err).ShouldNot(HaveOccurred())
		Expect(l).Should(BeEmpty())
	})
})
...
```

  上述snippet中，`ListAll`找到DB中的所有记录，并map到`*Blog`的切片中。测试语句非常直观，我们设置了该查询语句返回的是`nil`，也就是空集合。跑一下测试：

```
➜ ginkgo     
Running Suite: Pg Suite
=======================
Random Seed: 1585542357
Will run 8 of 8 specs


(/Users/dche423/dbtest/pg/repository.go:24) 
[2020-03-30 12:26:01]  Query: could not match actual sql: "SELECT * FROM "blogs"" with expected regexp "SELECT * FROM "blogs"" 
• Failure [0.001 seconds]
Repository
/Users/dche423/dbtest/pg/repository_test.go:16
  list all
  /Users/dche423/dbtest/pg/repository_test.go:37
    empty [It]
    /Users/dche423/dbtest/pg/repository_test.go:38

...
Test Suite Failed
➜  
```

  测试失败了...不过回显可以知道信息: `could not match actual sql with expected regexp.`。实际上Sqlmock使用`sqlmock.QueryMatcherRegex`为默认的SQL匹配器。在这个例子中，`sqlmock.ExpectQuery`输入一个正则表达式字符串而不是一个SQL的文本。所以我们有两种方式去解决这个问题:

1. 使用`regexp.QuoteMeta`， 也就是`mock.ExpectQuery(regexp.QuoteMeta(sqlSelectAll))`
2. 更改默认的SQL匹配器，当我们在创建mock实例的时候可以配置: `sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherEqual))`

  其实一般来说，正则表达式匹配器能更灵活一些。

#### 测试Load方法

```go
// repository.go
func (p *Repository) Load(id uint) (*Blog, error) {
	blog := &Blog{}
	err := p.db.Where(`id = ?`, id).First(blog).Error
	return blog, err
}
...


// repository_test.go
Context("load", func() {
        It("found", func() {
                blog := &Blog{
                        ID:        1,
                        Title:     "post",
                        ...
                }

                rows := sqlmock.
                        NewRows([]string{"id", "title", "content", "tags", "created_at"}).
                        AddRow(blog.ID, blog.Title, blog.Content, blog.Tags, blog.CreatedAt)

                const sqlSelectOne = `SELECT * FROM "blogs" WHERE (id = $1) ORDER BY "blogs"."id" ASC LIMIT 1`

                mock.ExpectQuery(regexp.QuoteMeta(sqlSelectOne)).WithArgs(blog.ID).WillReturnRows(rows)

                dbBlog, err := repository.Load(blog.ID)
                Expect(err).ShouldNot(HaveOccurred())
                Expect(dbBlog).Should(Equal(blog))
        })

        It("not found", func() {
                // ignore sql match
                mock.ExpectQuery(`.+`).WillReturnRows(sqlmock.NewRows(nil))
                _, err := repository.Load(1)
                Expect(err).Should(Equal(gorm.ErrRecordNotFound))
        })
})
...
```

  `Load`方法输入一个blog id作为参数，找到这个id对应的第一条记录。

  我们测试两种场景:

- 名为`found`的场景，我们创建blog实例并将其转换为`sql.Row`。随后调用`ExpectQuery`定义期望，在语句的最后，我们断言loaded blog实例和原来的一样。  **注意：如果你不清楚GORM使用的是什么SQL，可以打开debug flag -- gorm.DB的Debug()**
- 名为`not found`的场景，这里使用正则匹配来简化，表示不管什么sql都返回空。这里我们期望的是当找不到对应的blog时候，`gorm.ErrRecordNotFound`会被抛出。



#### 测试Save方法

  ```go
// repository.go
...
func (p *Repository) Save(blog *Blog) error {
	return p.db.Save(blog).Error
}


// repository_test.go
...
Context("save", func() {
        var blog *Blog
        BeforeEach(func() {
                blog = &Blog{
                        Title:     "post",
                        Content:   "hello",
                        Tags:      pq.StringArray{"a", "b"},
                        CreatedAt: time.Now(),
                }
        })

        It("insert", func() {
                // gorm use query instead of exec
                // https://github.com/DATA-DOG/go-sqlmock/issues/118
                const sqlInsert = `
                                INSERT INTO "blogs" ("title","content","tags","created_at") 
                                        VALUES ($1,$2,$3,$4) RETURNING "blogs"."id"`
                const newId = 1
                mock.ExpectBegin() // begin transaction
                mock.ExpectQuery(regexp.QuoteMeta(sqlInsert)).
                        WithArgs(blog.Title, blog.Content, blog.Tags, blog.CreatedAt).
                        WillReturnRows(sqlmock.NewRows([]string{"id"}).AddRow(newId))
                mock.ExpectCommit() // commit transaction

                Expect(blog.ID).Should(BeZero())

                err := repository.Save(blog)
                Expect(err).ShouldNot(HaveOccurred())

                Expect(blog.ID).Should(BeEquivalentTo(newId))
        })
	
	It("update", func() {
		...		
	})
		

})
  ```

  当data模型有已有的主键，`Save`方法能够更新DB记录；反之则插入一条新的记录。上面的snippet表现的插入的测试。

  创建一个新的blog实例，并且不给其设置主键。而后定义`mock.ExpectQuery`。在Query开始前begin一个事务，在之后commit。*一般情况下，非查询语句(`Insert/Update`)应该被`mock.ExepectExec`定义，但是这个是个特殊场景。因为某些原因，对于pg的语法，GORM使用`QueryRow`而非`Exec`。*

  最后，使用`Expect(blog.ID).Should(BeEquivalentTo(newId))` 来断言`blog.ID`在`Save`方法调用之后被设置了。其实一般来说，不太需要去对简单的`Insert/Update`语句进行单元测试，但是这里只是对一些GORM会进行的一些特殊场景进行说明，像其他的后端场景不用太多关注。



## 依赖注入

## Test Driven Development

[TDD Reference](https://studygolang.gitbook.io/learn-go-with-tests/go-ji-chu/maps)

### channel TDD 过程

#### 目标

目标： 写一个 `CheckWebsites` 的函数检查 URL 列表的状态。

```go
package concurrency

type WebsiteChecker func(string) bool

func CheckWebsites(wc WebsiteChecker, urls []string) map[string]bool {
    results := make(map[string]bool)

    for _, url := range urls {
        results[url] = wc(url)
    }

    return results
}
```

它返回一个 map，由每个 url 检查后的得到的布尔值组成，成功响应的值为 `true`，错误响应的值为 `false`。

你还必须传入一个 `WebsiteChecker` 处理单个 URL 并返回一个布尔值。它会被函数调用以检查所有的网站。

使用 [依赖注入](https://github.com/studygolang/learn-go-with-tests/tree/d8b18269a68c1cf4b8e8b0900f2815dc9d66d87e/zh-CN/zh-CN/dependency-injection.md)，允许在不发起真实 HTTP 请求的情况下测试函数，这使测试变得可靠和快速。

下面是简单的测试：

```go
package concurrency

import (
    "reflect"
    "testing"
)

func mockWebsiteChecker(url string) bool {
    if url == "waat://furhurterwe.geds" {
        return false
    }
    return true
}

func TestCheckWebsites(t *testing.T) {
    websites := []string{
        "http://google.com",
        "http://blog.gypsydave5.com",
        "waat://furhurterwe.geds",
    }

    actualResults := CheckWebsites(mockWebsiteChecker, websites)

    want := len(websites)
    got := len(actualResults)
    if want != got {
        t.Fatalf("Wanted %v, got %v", want, got)
    }

    expectedResults := map[string]bool{
        "http://google.com":          true,
        "http://blog.gypsydave5.com": true,
        "waat://furhurterwe.geds":    false,
    }

    if !reflect.DeepEqual(expectedResults, actualResults) {
        t.Fatalf("Wanted %v, got %v", expectedResults, actualResults)
    }
}
```

该功能在生产环境中被用于检查数百个网站。但是它速度很慢，所以需要为程序提速。



#### 写一个测试

首先我们对 `CheckWebsites` 做一个基准测试，这样就能看到我们修改的影响。

```go
package concurrency

import (
    "testing"
    "time"
)

func slowStubWebsiteChecker(_ string) bool {
    time.Sleep(20 * time.Millisecond)
    return true
}

func BenchmarkCheckWebsites(b *testing.B) {
    urls := make([]string, 100)
    for i := 0; i < len(urls); i++ {
        urls[i] = "a url"
    }

    for i := 0; i < b.N; i++ {
        CheckWebsites(slowStubWebsiteChecker, urls)
    }
}
```

基准测试使用一百个网址的 slice 对 `CheckWebsites` 进行测试，并使用 `WebsiteChecker` 的伪造实现。`slowStubWebsiteChecker` 故意放慢速度。它使用 `time.Sleep` 明确等待 20 毫秒，然后返回 true。

当我们运行基准测试时使用 `go test -bench=.` 命令 (如果在 Windows Powershell 环境下使用 `go test -bench="."`)：

```
pkg: github.com/gypsydave5/learn-go-with-tests/concurrency/v0
BenchmarkCheckWebsites-4               1        2249228637 ns/op
PASS
ok      github.com/gypsydave5/learn-go-with-tests/concurrency/v0        2.268s
```

`CheckWebsite` 经过基准测试的时间为 2249228637 纳秒，大约 2.25 秒。

让我们尝试去让它运行得更快。



#### 编写足够的代码让它通过

现在我们终于可以谈论并发了，以下内容是为了说明「不止一件事情正在进行中」。这是我们每天很自然在做的事情。

比如，今天早上我泡了一杯茶。我放上水壶，然后在等待它煮沸时，从冰箱里取出了牛奶，把茶从柜子里拿出来，找到我最喜欢的杯子，把茶袋放进杯子里，然后等水壶沸了，把水倒进杯子里。

我 *没有* 做的事情是放上水壶，然后呆呆地盯着水壶等水煮沸，然后在煮沸后再做其他事情。

如果你能理解为什么第一种方式泡茶更快，那你就可以理解我们如何让 `CheckWebsites` 变得更快。与其等待网站响应之后再发送下一个网站的请求，不如告诉计算机在等待时就发起下一个请求。

通常在 Go 中，当调用函数 `doSomething()` 时，我们等待它返回（即使它没有值返回，我们仍然等待它完成）。我们说这个操作是 *阻塞* 的 —— 它让我们等待它完成。Go 中不会阻塞的操作将在称为 *goroutine* 的单独 *进程* 中运行。将程序想象成从上到下读 Go 的 代码，当函数被调用执行读取操作时，进入每个函数「内部」。当一个单独的进程开始时，就像开启另一个 reader（阅读程序）在函数内部执行读取操作，原来的 reader 继续向下读取 Go 代码。

要告诉 Go 开始一个新的 goroutine，我们把一个函数调用变成 `go` 声明，通过把关键字 `go` 放在它前面：`go doSomething()`。

```go
package concurrency

type WebsiteChecker func(string) bool

func CheckWebsites(wc WebsiteChecker, urls []string) map[string]bool {
    results := make(map[string]bool)

    for _, url := range urls {
        go func() {
            results[url] = wc(url)
        }()
    }

    return results
}
```

因为开启 goroutine 的唯一方法就是将 `go` 放在函数调用前面，所以当我们想要启动 goroutine 时，我们经常使用 *匿名函数（anonymous functions）*。一个匿名函数文字看起来和正常函数声明一样，但没有名字（意料之中）。你可以在 上面的 `for` 循环体中看到一个。

匿名函数有许多有用的特性，其中两个上面正在使用。首先，它们可以在声明的同时执行 —— 这就是匿名函数末尾的 `()` 实现的。其次，它们维护对其所定义的词汇作用域的访问权 —— 在声明匿名函数时所有可用的变量也可在函数体内使用。

上面匿名函数的主体和之前循环体中的完全一样。唯一的区别是循环的每次迭代都会启动一个新的 goroutine，与当前进程（`WebsiteChecker` 函数）同时发生，每个循环都会将结果添加到 `results` map 中。

但是当我们执行 `go test`：

```
-------- FAIL: TestCheckWebsites (0.00s)
        CheckWebsites_test.go:31: Wanted map[http://google.com:true http://blog.gypsydave5.com:true waat://furhurterwe.geds:false], got map[]
FAIL
exit status 1
FAIL    github.com/gypsydave5/learn-go-with-tests/concurrency/v1        0.010s
```

#### 不可预知的问题

你可能不会得到这个结果。你可能会得到一个 panic 信息，这个稍后再谈。如果你得到的是那些结果，不要担心，只要继续运行测试，直到你得到上述结果。或假装你得到了，这取决于你。欢迎来到并发编程的世界：如果处理不正确，很难预测会发生什么。别担心 —— 这就是我们编写测试的原因，当处理并发时，测试帮助我们预测可能发生的情况。

让我们困惑的是，原来的测试 `WebsiteChecker` 现在返回空的 map。哪里出问题了？

我们 `for` 循环开始的 `goroutines` 没有足够的时间将结果添加结果到 `results` map 中；`WebsiteChecker` 函数对于它们来说太快了，以至于它返回时仍为空的 map。

为了解决这个问题，我们可以等待所有的 goroutine 完成他们的工作，然后返回。两秒钟应该能完成了，对吧？

```go
package concurrency

import "time"

type WebsiteChecker func(string) bool

func CheckWebsites(wc WebsiteChecker, urls []string) map[string]bool {
    results := make(map[string]bool)

    for _, url := range urls {
        go func() {
            results[url] = wc(url)
        }()
    }

    time.Sleep(2 * time.Second)

    return results
}
```

现在当我们运行测试时获得的结果（如果没有得到 —— 参考上面的做法）：

```
-------- FAIL: TestCheckWebsites (0.00s)
        CheckWebsites_test.go:31: Wanted map[http://google.com:true http://blog.gypsydave5.com:true waat://furhurterwe.geds:false], got map[waat://furhurterwe.geds:false]
FAIL
exit status 1
FAIL    github.com/gypsydave5/learn-go-with-tests/concurrency/v1        0.010s
```

这不是很好 - 为什么只有一个结果？我们可以尝试通过增加等待的时间来解决这个问题 —— 如果你愿意，可以试试。但没什么作用。这里的问题是变量 `url` 被重复用于 `for` 循环的每次迭代 —— 每次都会从 `urls` 获取新值。但是我们的每个 goroutine 都是 `url` 变量的引用 —— 它们没有自己的独立副本。所以他们 *都* 会写入在迭代结束时的 `url` —— 最后一个 url。这就是为什么我们得到的结果是最后一个 url ---- **注意：闭包情况下的引用关系一直是需要注意的**

解决这个问题:

```go
import (
    "time"
)

type WebsiteChecker func(string) bool

func CheckWebsites(wc WebsiteChecker, urls []string) map[string]bool {
    results := make(map[string]bool)

    for _, url := range urls {
        go func(u string) {
            results[u] = wc(u)
        }(url)
    }

    time.Sleep(2 * time.Second)

    return results
}
```

通过给每个匿名函数一个参数 url(`u`)，然后用 `url` 作为参数调用匿名函数，我们确保 `u` 的值固定为循环迭代的 `url` 值，重新启动 `goroutine`。`u` 是 `url` 值的副本，因此无法更改。

现在，如果你幸运的话，你会得到：

```
PASS
ok      github.com/gypsydave5/learn-go-with-tests/concurrency/v1        2.012s
```

但是，如果你不走运（如果你运行基准测试，这很可能会发生，因为你将发起多次的尝试）。

```
fatal error: concurrent map writes

goroutine 8 [running]:
runtime.throw(0x12c5895, 0x15)
        /usr/local/Cellar/go/1.9.3/libexec/src/runtime/panic.go:605 +0x95 fp=0xc420037700 sp=0xc4200376e0 pc=0x102d395
runtime.mapassign_faststr(0x1271d80, 0xc42007acf0, 0x12c6634, 0x17, 0x0)
        /usr/local/Cellar/go/1.9.3/libexec/src/runtime/hashmap_fast.go:783 +0x4f5 fp=0xc420037780 sp=0xc420037700 pc=0x100eb65
github.com/gypsydave5/learn-go-with-tests/concurrency/v3.WebsiteChecker.func1(0xc42007acf0, 0x12d3938, 0x12c6634, 0x17)
        /Users/gypsydave5/go/src/github.com/gypsydave5/learn-go-with-tests/concurrency/v3/websiteChecker.go:12 +0x71 fp=0xc4200377c0 sp=0xc420037780 pc=0x12308f1
runtime.goexit()
        /usr/local/Cellar/go/1.9.3/libexec/src/runtime/asm_amd64.s:2337 +0x1 fp=0xc4200377c8 sp=0xc4200377c0 pc=0x105cf01
created by github.com/gypsydave5/learn-go-with-tests/concurrency/v3.WebsiteChecker
        /Users/gypsydave5/go/src/github.com/gypsydave5/learn-go-with-tests/concurrency/v3/websiteChecker.go:11 +0xa1

        ... many more scary lines of text ...
```

这看上去冗长、可怕，我们需要深呼吸并阅读错误：`fatal error: concurrent map writes`。有时候，当我们运行我们的测试时，两个 goroutines 完全同时写入 `results` map。Go 的 Maps 不喜欢多个事物试图一次性写入，所以就导致了 `fatal error`。

这是一种 *race condition（竞争条件）*，当软件的输出取决于事件发生的时间和顺序时，因为我们无法控制，bug 就会出现。因为我们无法准确控制每个 goroutine 写入结果 map 的时间，两个 goroutines 同一时间写入时程序将非常脆弱。

Go 可以帮助我们通过其内置的 [race detector](https://blog.golang.org/race-detector) 来发现竞争条件。要启用此功能，请使用 `race` 标志运行测试：`go test -race`。

你应该得到一些如下所示的输出：

```
==================
WARNING: DATA RACE
Write at 0x00c420084d20 by goroutine 8:
  runtime.mapassign_faststr()
      /usr/local/Cellar/go/1.9.3/libexec/src/runtime/hashmap_fast.go:774 +0x0
  github.com/gypsydave5/learn-go-with-tests/concurrency/v3.WebsiteChecker.func1()
      /Users/gypsydave5/go/src/github.com/gypsydave5/learn-go-with-tests/concurrency/v3/websiteChecker.go:12 +0x82

Previous write at 0x00c420084d20 by goroutine 7:
  runtime.mapassign_faststr()
      /usr/local/Cellar/go/1.9.3/libexec/src/runtime/hashmap_fast.go:774 +0x0
  github.com/gypsydave5/learn-go-with-tests/concurrency/v3.WebsiteChecker.func1()
      /Users/gypsydave5/go/src/github.com/gypsydave5/learn-go-with-tests/concurrency/v3/websiteChecker.go:12 +0x82

Goroutine 8 (running) created at:
  github.com/gypsydave5/learn-go-with-tests/concurrency/v3.WebsiteChecker()
      /Users/gypsydave5/go/src/github.com/gypsydave5/learn-go-with-tests/concurrency/v3/websiteChecker.go:11 +0xc4
  github.com/gypsydave5/learn-go-with-tests/concurrency/v3.TestWebsiteChecker()
      /Users/gypsydave5/go/src/github.com/gypsydave5/learn-go-with-tests/concurrency/v3/websiteChecker_test.go:27 +0xad
  testing.tRunner()
      /usr/local/Cellar/go/1.9.3/libexec/src/testing/testing.go:746 +0x16c

Goroutine 7 (finished) created at:
  github.com/gypsydave5/learn-go-with-tests/concurrency/v3.WebsiteChecker()
      /Users/gypsydave5/go/src/github.com/gypsydave5/learn-go-with-tests/concurrency/v3/websiteChecker.go:11 +0xc4
  github.com/gypsydave5/learn-go-with-tests/concurrency/v3.TestWebsiteChecker()
      /Users/gypsydave5/go/src/github.com/gypsydave5/learn-go-with-tests/concurrency/v3/websiteChecker_test.go:27 +0xad
  testing.tRunner()
      /usr/local/Cellar/go/1.9.3/libexec/src/testing/testing.go:746 +0x16c
==================
```

细节还是难以阅读 - 但 `WARNING: DATA RACE` 相当明确。阅读错误的内容，我们可以看到两个不同的 goroutines 在 map 上执行写入操作：

```
Write at 0x00c420084d20 by goroutine 8:
```

正在写入相同的内存块

```
Previous write at 0x00c420084d20 by goroutine 7:
```

最重要的是，我们可以看到发生写入的代码行：

```
/Users/gypsydave5/go/src/github.com/gypsydave5/learn-go-with-tests/concurrency/v3/websiteChecker.go:12
```

和 goroutines 7 和 8 开始的代码行号：

```
/Users/gypsydave5/go/src/github.com/gypsydave5/learn-go-with-tests/concurrency/v3/websiteChecker.go:11
```

你需要知道的所有内容都会打印到你的终端上 - 你只需耐心阅读就可以了。



#### 使用channels处理race condition

我们可以通过使用 *channels* 协调我们的 goroutines 来解决这个数据竞争。channels 是一个 Go 数据结构，可以同时接收和发送值。这些操作以及细节允许不同进程之间的通信。

在这种情况下，我们想要考虑父进程和每个 goroutine 之间的通信，goroutine 使用 url 来执行 `WebsiteChecker` 函数。

```go
package concurrency

type WebsiteChecker func(string) bool
type result struct {
    string
    bool
}

func CheckWebsites(wc WebsiteChecker, urls []string) map[string]bool {
    results := make(map[string]bool)
    resultChannel := make(chan result)

    for _, url := range urls {
        go func(u string) {
            resultChannel <- result{u, wc(u)}
        }(url)
    }

    for i := 0; i < len(urls); i++ {
        result := <-resultChannel
        results[result.string] = result.bool
    }

    return results
}
```

除了 `results` map 之外，我们现在还有一个 `resultChannel` 的变量，同样使用 `make` 方法创建。`chan result` 是 channel 类型的 —— `result` 的 channel。新类型的 `result` 是将 `WebsiteChecker` 的返回值与正在检查的 url 相关联 —— 它是一个 `string` 和 `bool` 的结构。因为我们不需要任何一个要命名的值，它们中的每一个在结构中都是匿名的；这在很难知道用什么命名值的时候可能很有用。

现在，当我们迭代 urls 时，不是直接写入 `map`，而是使用 *send statement* 将每个调用 `wc` 的 `result` 结构体发送到 `resultChannel`。这使用 `<-` 操作符，channel 放在左边，值放在右边：

```go
// send statement
resultChannel <- result{u, wc(u)
```

下一个 `for` 循环为每个 url 迭代一次。 我们在内部使用 *receive expression*，它将从通道接收到的值分配给变量。这也使用 `<-` 操作符，但现在两个操作数颠倒过来：现在 channel 在右边，我们指定的变量在左边：

```go
// receive expression
result := <-resultChannel
```

然后我们使用接收到的 `result` 更新 map。

通过将结果发送到通道，我们可以控制每次写入 `results` map 的时间，确保每次写入一个结果。虽然 `wc` 的每个调用都发送给结果通道，但是它们在其自己的进程内并行发生，因为我们将结果通道中的值与接收表达式一起逐个处理一个结果。

我们已经将想要加快速度的那部分代码并行化，同时确保不能并发的部分仍然是线性处理。我们使用 channel 在多个进程间通信。

当我们运行基准时：

```
pkg: github.com/gypsydave5/learn-go-with-tests/concurrency/v2
BenchmarkCheckWebsites-8             100          23406615 ns/op
PASS
ok      github.com/gypsydave5/learn-go-with-tests/concurrency/v2        2.377s
```

23406615 纳秒 —— 0.023 秒，速度大约是最初函数的一百倍，这是非常成功的。



#### 总结

某种程度说，我们已经参与了 `CheckWebsites` 函数的一个长期重构；输入和输出从未改变，它只是变得更快了。但是我们所做的测试以及我们编写的基准测试允许我们重构 `CheckWebsites`，让我们有信心保证软件仍然可以工作，同时也证明它确实变得更快了。

在使它更快的过程中，我们明白了

- *goroutines* 是 Go 的基本并发单元，它让我们可以同时检查多个网站。
- *anonymous functions（匿名函数）*，我们用它来启动每个检查网站的并发进程。
- *channels*，用来组织和控制不同进程之间的交流，使我们能够避免 *race condition（竞争条件）* 的问题。
- *the race detector（竞争探测器）* 帮助我们调试并发代码的问题。



