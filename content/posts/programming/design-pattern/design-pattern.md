---
title: "Design Pattern: Overview"
date: 2020-11-11T11:22:18+08:00
hero: /images/posts/golang-banner-2.png
menu:
  sidebar:
    name: Design Pattern Overview
    identifier: design-pattern
    parent: programming pattern
    weight: 10
draft: false
---
# Design pattern

## Builder Pattern

### scenario：build complicated object

```go
package msg

type Message struct {
	Header *Header
	Body   *Body
}
type Header struct {
	SrcAddr  string
	SrcPort  uint64
	DestAddr string
	DestPort uint64
	Items    map[string]string
}
type Body struct {
	Items []string
}

// Message对象的复杂对象
type builder struct{
  once *sync.Once
  msg *Message
}

// 返回Builder对象
func Builder() *builder{
  return &builder{
    once: &sync.Once{},
    msg: &Message{Header: &Header{}, Body: &Body{}},
  }
}

func (b *builder) WithSrcAddr(srcAddr string) *builder{
  b.msg.Header.SrcAddr = srcAddr
  return b
}

//......
func (b *builder) WithHeaderItem(key, value string) *builder{
  //map只初始化一次
  b.once.Do(func(){
    b.msg.Header.Items = make(map[string]string)
  })
  b.msg.Header.Items[key] = value
  return b
}

func (b *builder) WithBodyItem(record string) *builder{
  b.msg.Body.Items = append(b.msg.Body.Items, record)
  return b
} 

func (b *builder) Build() *Message{
  return b.msg
}
```

### Test code

```go
package test
func TestMessageBuilder(t *testing.T) {
  // 使用消息建造者进行对象创建
	message := msg.Builder().
		WithSrcAddr("192.168.0.1").
		WithSrcPort(1234).
		WithDestAddr("192.168.0.2").
		WithDestPort(8080).
		WithHeaderItem("contents", "application/json").
		WithBodyItem("record1").
		WithBodyItem("record2").
		Build()
	if message.Header.SrcAddr != "192.168.0.1" {
		t.Errorf("expect src address 192.168.0.1, but actual %s.", message.Header.SrcAddr)
	}
	if message.Body.Items[0] != "record1" {
		t.Errorf("expect body item0 record1, but actual %s.", message.Body.Items[0])
	}
}
```



##  Abstract Factory Pattern

![abstract factory](https://tva1.sinaimg.cn/large/007S8ZIlgy1ghkw23e4r3j31bs0nge82.jpg?imageslim)

常规的工厂模式，如果新增一个对象，需要修改原来的工厂对象代码，违反单一职责原则，最好增加一个抽象层。

### interfaces definitions

```go
package plugin

//插件抽象接口定义
type Plugin interface{}

// 输入插件，用于接收消息
type Input interface{
  Plugin
  Receive() string
}

// 过滤插件，用于处理消息
type Filter interface{
  Plugin
  Process(msg string) string
}

// 输出插件，用于发送消息
type Output interface{
  Plugin
  Send(msg string)
}
```

### pipeline composition

管道由上述三种插件定义

```go
package pipeline
...
// 消息管道的定义
type Pipeline struct {
	input  plugin.Input
	filter plugin.Filter
	output plugin.Output
}
// 一个消息的处理流程为 input -> filter -> output
func (p *Pipeline) Exec() {
	msg := p.input.Receive()
	msg = p.filter.Process(msg)
	p.output.Send(msg)
}
```

### plugins implementation

定义三种插件的具体实现

```go
package plugin
...
// input插件名称与类型的映射关系，主要用于通过反射创建input对象
var inputNames = make(map[string]reflect.Type)
// Hello input插件，接收“Hello World”消息
type HelloInput struct {}

func (h *HelloInput) Receive() string {
	return "Hello World"
}
// 初始化input插件映射关系表
func init() {
	inputNames["hello"] = reflect.TypeOf(HelloInput{})
}
```

```go
package plugin
...
// filter插件名称与类型的映射关系，主要用于通过反射创建filter对象
var filterNames = make(map[string]reflect.Type)
// Upper filter插件，将消息全部字母转成大写
type UpperFilter struct {}

func (u *UpperFilter) Process(msg string) string {
	return strings.ToUpper(msg)
}
// 初始化filter插件映射关系表
func init() {
	filterNames["upper"] = reflect.TypeOf(UpperFilter{})
}
```

```go
package plugin
...
// output插件名称与类型的映射关系，主要用于通过反射创建output对象
var outputNames = make(map[string]reflect.Type)
// Console output插件，将消息输出到控制台上
type ConsoleOutput struct {}

func (c *ConsoleOutput) Send(msg string) {
	fmt.Println(msg)
}
// 初始化output插件映射关系表
func init() {
	outputNames["console"] = reflect.TypeOf(ConsoleOutput{})
}
```

### abstract factory interface definition & implementation

定义抽象工厂接口，和对应插件的工厂实现

```go
package plugin
...
// 插件抽象工厂接口
type Factory interface {
	Create(conf Config) Plugin
}
// input插件工厂对象，实现Factory接口
type InputFactory struct{}
// 读取配置，通过反射机制进行对象实例化
func (i *InputFactory) Create(conf Config) Plugin {
	t, _ := inputNames[conf.Name]
	return reflect.New(t).Interface().(Plugin)
}
// filter和output插件工厂实现类似
type FilterFactory struct{}
func (f *FilterFactory) Create(conf Config) Plugin {
	t, _ := filterNames[conf.Name]
	return reflect.New(t).Interface().(Plugin)
}
type OutputFactory struct{}
func (o *OutputFactory) Create(conf Config) Plugin {
	t, _ := outputNames[conf.Name]
	return reflect.New(t).Interface().(Plugin)
}

```

### pipeline factory definition 

最后定义pipeline工厂方法，调用plugin.Factory 抽象工厂完成pipeline对象的实例化

```go
package pipeline
...
// 保存用于创建Plugin的工厂实例，其中map的key为插件类型，value为抽象工厂接口
var pluginFactories = make(map[plugin.Type]plugin.Factory)
// 根据plugin.Type返回对应Plugin类型的工厂实例
func factoryOf(t plugin.Type) plugin.Factory {
	factory, _ := pluginFactories[t]
	return factory
}
// pipeline工厂方法，根据配置创建一个Pipeline实例
func Of(conf Config) *Pipeline {
	p := &Pipeline{}
	p.input = factoryOf(plugin.InputType).Create(conf.Input).(plugin.Input)
	p.filter = factoryOf(plugin.FilterType).Create(conf.Filter).(plugin.Filter)
	p.output = factoryOf(plugin.OutputType).Create(conf.Output).(plugin.Output)
	return p
}
// 初始化插件工厂对象
func init() {
	pluginFactories[plugin.InputType] = &plugin.InputFactory{}
	pluginFactories[plugin.FilterType] = &plugin.FilterFactory{}
	pluginFactories[plugin.OutputType] = &plugin.OutputFactory{}
}
```



## Prototype Pattern

场景：对象的复制，如果对象成员变量复杂，或者对象有不可见变量，即会有问题

```go
package prototype
...
// 原型复制抽象接口
type Prototype interface {
	clone() Prototype
}

type Message struct {
	Header *Header
	Body   *Body
}

func (m *Message) clone() Prototype {
	msg := *m
	return &msg
}
```



## Adapter Pattern

![adaptor pattern](https://tva1.sinaimg.cn/large/007S8ZIlgy1ghzyia5t19j315w0k0kjl.jpg)

最常用的模式之一，典型场景是系统中老的接口过时或者即将废弃，可以新增一个适配器，把老的接口适配成新的接口使用，践行了开闭原则。该模式即把一个接口adaptee，通过适配器adapter转换成client锁期望的另一个接口target，也就是adapter通过实现target接口，并在对应的方法里调用adaptee的接口实现。



继续消息处理系统的例子，目前系统的输入都来自HelloInput, 假设需要新增一个kafka消息队列中接收数据的功能，其中kafka消费者的接口如下：

```go
package kafka 

type Records struct{
  Items []string
}

type Comsumer interface{
  Poll() Records
}
```

而Pipeline的设计是通过plugin.Input接口进行消息接收，所以这个kafka的接口无法直接集成。因此需要用适配器

```go
package plugin
...
type KafkaInput struct {
	status Status
	consumer kafka.Consumer
}

func (k *KafkaInput) Receive() *msg.Message {
	records := k.consumer.Poll()
	if k.status != Started {
		fmt.Println("Kafka input plugin is not running, input nothing.")
		return nil
	}
	return msg.Builder().
		WithHeaderItem("content", "kafka").
		WithBodyItems(records.Items).
		Build()
}

// 在输入插件映射关系中加入kafka，用于通过反射创建input对象
func init() {
	inputNames["hello"] = reflect.TypeOf(HelloInput{})
	inputNames["kafka"] = reflect.TypeOf(KafkaInput{})
}
```

这里有个问题就是KafkaInput这个对象的成员构造问题，需要特别的init函数去初始化，可以考虑在Plugin接口新增一个Init方法，用于定义插件的一些初始化操作，并在工厂返回实例前调用。

```go
package plugin
...
type Plugin interface {
	Start()
	Stop()
	Status() Status
	// 新增初始化方法，在插件工厂返回实例前调用
	Init()
}

// 修改后的插件工厂实现如下
func (i *InputFactory) Create(conf Config) Plugin {
	t, _ := inputNames[conf.Name]
	p := reflect.New(t).Interface().(Plugin)
  // 返回插件实例前调用Init函数，完成相关初始化方法
	p.Init()
	return p
}

// KakkaInput的Init函数实现
func (k *KafkaInput) Init() {
	k.consumer = &kafka.MockConsumer{}
}
```

上述的MockConsumer的实现如下：

```go
package kafka
...
type MockConsumer struct {}

func (m *MockConsumer) Poll() *Records {
	records := &Records{}
	records.Items = append(records.Items, "i am mock consumer.")
	return records
}
```

### Test code

测试代码如下：

```go
package test
...
func TestKafkaInputPipeline(t *testing.T) {
	config := pipeline.Config{
		Name: "pipeline2",
		Input: plugin.Config{
			PluginType: plugin.InputType,
			Name:       "kafka",
		},
		Filter: plugin.Config{
			PluginType: plugin.FilterType,
			Name:       "upper",
		},
		Output: plugin.Config{
			PluginType: plugin.OutputType,
			Name:       "console",
		},
	}
	p := pipeline.Of(config)
	p.Start()
	p.Exec()
	p.Stop()
}
// 运行结果
=== RUN   TestKafkaInputPipeline
Console output plugin started.
Upper filter plugin started.
Kafka input plugin started.
Pipeline started.
Output:
	Header:map[content:kafka], Body:[I AM MOCK CONSUMER.]
Kafka input plugin stopped.
Upper filter plugin stopped.
Console output plugin stopped.
Pipeline stopped.
--- PASS: TestKafkaInputPipeline (0.00s)
PASS
```



## Bridge Pattern

![bridge pattern](https://tva1.sinaimg.cn/large/007S8ZIlgy1gi00awfcxcj31f20l01ky.jpg)

![bridge example](https://tva1.sinaimg.cn/large/007S8ZIlgy1gi01kwmua8j31hs0s47wj.jpg)

场景： 如果一个对象存在多个变化的方向，而且每个变化方向都需要扩展，桥接是好的选择。

实际上上述的消息处理系统就是这样，一个pipeline有三个特征，且pipeline只依赖这三个接口而非具体的实现细节。

![bridge example2](https://tva1.sinaimg.cn/large/007S8ZIlgy1gi0i6xpj9sj318a0nk4qq.jpg)





## Proxy Pattern

代理模式为一个对象提供一种代理以控制对该对象的访问，使用率非常高。