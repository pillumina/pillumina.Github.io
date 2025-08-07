---
title: "fasthttp对性能的优化压榨"
date: 2021-01-10T11:22:18+08:00
hero: /images/posts/golang2.jpg
menu:
  sidebar:
    name: fasthttp对性能的优化研究
    identifier: fasthttp
    parent: golang odyssey
    weight: 10
draft: false
---

*最近在看网络模型和go net的源码，以及各web框架例如fasthttp, weaver, gnet(更轻量)源码。fasthttp在github上已经写上了一个go开发的best practices [examples](https://github.com/valyala/fasthttp#fasthttp-best-practices),这里我也记录一些在源码中看到的一些技巧*



### `[]byte` buffer的tricks

下面的一些tricks在fasthttp中被使用，自己的代码也可以用

- 标准Go函数能够处理nil buffer

```go
var (
	// both buffers are uninitialized
	dst []byte
	src []byte
)
dst = append(dst, src...)  // is legal if dst is nil and/or src is nil
copy(dst, src)  // is legal if dst is nil and/or src is nil
(string(src) == "")  // is true if src is nil
(len(src) == 0)  // is true if src is nil
src = src[:0]  // works like a charm with nil src

// this for loop doesn't panic if src is nil
for i, ch := range src {
	doSomething(i, ch)
}
```

所以可以去掉一些对`[]byte`buffer的nil校验:

```go
srcLen := 0
if src != nil {
	srcLen = len(src)
}
```

改成

```go
srcLen := len(src)
```



- 字符串能够直接`append`到`[]byte`上

```go
dst = append(dst, "foobar"...)
```

- `[]byte`buffer能够扩展到它的cap

```go
buf := make([]byte, 100)
a := buf[:10]  // len(a) == 10, cap(a) == 100.
b := a[:100]  // is valid, since cap(a) == 100.
```

- 所有fasthtto函数都接受nil的`[]byte`buffer

```go
statusCode, body, err := fasthttp.Get(nil, "http://google.com/")
uintBuf := fasthttp.AppendUint(nil, 1234)
```



### 减少`[]byte`的分配，尽量复用

有两种方式进行复用:

1. `sync.Pool`
2. `slice = slice[:0]` 所有的类型的Reset方法，都用了这个方式。比如类型URL，Args, ByteBuffer,  Cookie, RequestHeader, ResponseHeader等

fasthttp里共有35个地方使用了sync.Pool。sync.Pool除了降低GC的压力，还能复用对象，减少内存分配，所以在自己写的`goroutine pool`中也对`worker`对象使用了`sync.Pool`。

```go
// 例如类型Server
type Server struct {
    // ...
    ctxPool        sync.Pool // 存RequestCtx对象
	readerPool     sync.Pool // 存bufio对象，用于读HTTP Request
	writerPool     sync.Pool // 存bufio对象，用于写HTTP Request
	hijackConnPool sync.Pool
	bytePool       sync.Pool
}


// 例如cookies
var cookiePool = &sync.Pool{
	New: func() interface{} {
		return &Cookie{}
	},
}

func AcquireCookie() *Cookie {
	return cookiePool.Get().(*Cookie)
}

func ReleaseCookie(c *Cookie) {
	c.Reset()
	cookiePool.Put(c)
}

// 例如workPool. 每个请求以一个新的goroutine运行。就是workpool做的调度
type workerPool struct {
  // ...
	workerChanPool sync.Pool
}

func (wp *workerPool) getCh() *workerChan {
	var ch *workerChan
	// ...

	if ch == nil {
		if !createWorker {
      // 已经达到worker数量上限，不允许创建了
			return nil
		}
    // 尝试复用旧worker
		vch := wp.workerChanPool.Get()
		if vch == nil {
			vch = &workerChan{
				ch: make(chan net.Conn, workerChanCap),
			}
		}
		ch = vch.(*workerChan)
    // 创建新的goroutine处理请求
		go func() {
			wp.workerFunc(ch)
      // 用完了返回去
			wp.workerChanPool.Put(vch)
		}()
	}
	return ch
}
```

  复用已经分配的`[]byte`。

`s = s[:0]`和`s = append(s[:0], b…)`这两种复用方式，总共出现了191次。

```go
// 清空 URI
func (u *URI) Reset() {
	u.pathOriginal = u.pathOriginal[:0]
	u.scheme = u.scheme[:0]
	u.path = u.path[:0]
    // ....
}

// 清空 ResponseHeader
func (h *ResponseHeader) resetSkipNormalize() {
	h.noHTTP11 = false
	h.connectionClose = false

	h.statusCode = 0
	h.contentLength = 0
	h.contentLengthBytes = h.contentLengthBytes[:0]

	h.contentType = h.contentType[:0]
	h.server = h.server[:0]

	h.h = h.h[:0]
	h.cookies = h.cookies[:0]
}

// 清空Cookies
func (c *Cookie) Reset() {
	c.key = c.key[:0]
	c.value = c.value[:0]
	c.expire = zeroTime
	c.maxAge = 0
	c.domain = c.domain[:0]
	c.path = c.path[:0]
	c.httpOnly = false
	c.secure = false
	c.sameSite = CookieSameSiteDisabled
}

func (c *Cookie) SetKey(key string) {
	c.key = append(c.key[:0], key...)
}
```



### 方法参数尽量用`[]byte`, write only场景可以避免用bytes.Buffer

  方法参数使用`[]byte`， 可以避免从`[]byte`到string转换时带来的内存分配和拷贝的开销。毕竟从net.Conn中读出来的数据也是[]byte类型。

  某些地方如果的确想穿string类型，fasthttp也提供XXXString()的方法。

  String方法用了`a = append(a, string…)`，这种写法不会造成string到[]byte的转换(汇编里没有用到runtime.stringtoslicebyte方法)

```go
// 例如写Response时，提供专门的String方法
func (resp *Response) SetBodyString(body string) {
	// ...
	bodyBuf.WriteString(body)
}
```

  上面的bodyBuf变量类型为ByteBuffer，来源于作者另外写的一个库，[bytebufferpool](https://link.zhihu.com/?target=https%3A//github.com/valyala/bytebufferpool)。

  正如介绍一样，库的主要目标是反对多余的内存分配行为。与标准库的bytes.Buffer类型对比，性能高30%。

  但ByteBuffer只提供了write类操作。适合高频写场景。

  先看下标准库bytes.Buffer是如何增长底层slice的。重点是bytes.Buffer没有内存复用:

```go
// 增长slice时，都会调用grow方法
func (b *Buffer) grow(n int) int {
	// ...
	if m+n <= cap(b.buf)/2 {
		copy(b.buf[:], b.buf[b.off:])
	} else {
		// 通过makeSlice获取新的slice
    buf := makeSlice(2*cap(b.buf) + n)
    // 而且还要拷贝
		copy(buf, b.buf[b.off:])
		b.buf = buf
	}
    // ...
}

func makeSlice(n int) []byte {
    // maekSlice 是直接分配出新的slice，没有复用的意思
	return make([]byte, n)
}
```

再看ByteBuffer的做法。重点是复用内存:

```go
// 通过复用减少内存分配，下次复用
func (b *ByteBuffer) Reset() {
	b.B = b.B[:0]
}

// 提供专门String方法，通过append避免string到[]byte转换带来的内存分配和拷贝
func (b *ByteBuffer) WriteString(s string) (int, error) {
	b.B = append(b.B, s...)
	return len(s), nil
}

// 如果写buffer的内容很大呢？增长的事情交给append
// 但因为Reset()做了复用，所以cap足够情况下，append速度会很快
func (b *ByteBuffer) Write(p []byte) (int, error) {
	b.B = append(b.B, p...)
	return len(p), nil
}

```

  Request和Response都是用ByteBuffer存body的。清空body是把ByteBuffer交还给pool，方便复用。

```go
var (
	responseBodyPool bytebufferpool.Pool
	requestBodyPool  bytebufferpool.Pool
)

func (req *Request) ResetBody() {
	req.RemoveMultipartFormFiles()
	req.closeBodyStream()
	if req.body != nil {
		if req.keepBodyBuffer {
			req.body.Reset()
		} else {
			requestBodyPool.Put(req.body)
			req.body = nil
		}
	}
}

func (resp *Response) ResetBody() {
	resp.bodyRaw = nil
	resp.closeBodyStream()
	if resp.body != nil {
		if resp.keepBodyBuffer {
			resp.body.Reset()
		} else {
			responseBodyPool.Put(resp.body)
			resp.body = nil
		}
	}
}
```



### 极限复用内存的地方

  有些地方需要kv型数据，一般使用map[string]string。但map不利于复用。所以fasthttp使用slice来实现了map，这个优化其实挺极限的，而且查询复杂度会降到O(n)。所以这种优化适用于**key数量不多，而且并发量大**的场景，这样slice的方式就能很好得减少内存。

```go
type argsKV struct {
	key     []byte
	value   []byte
	noValue bool
}

// 增加新的kv
func appendArg(args []argsKV, key, value string, noValue bool) []argsKV {
	var kv *argsKV
	args, kv = allocArg(args)
  // 复用原来key的内存空间
	kv.key = append(kv.key[:0], key...)
	if noValue {
		kv.value = kv.value[:0]
	} else {
    // 复用原来value的内存空间
		kv.value = append(kv.value[:0], value...)
	}
	kv.noValue = noValue
	return args
}

func allocArg(h []argsKV) ([]argsKV, *argsKV) {
	n := len(h)
	if cap(h) > n {
    // 复用底层数组空间，不用分配
		h = h[:n+1]
	} else {
    // 空间不足再分配
		h = append(h, argsKV{})
	}
	return h, &h[n]
}
```



### 避免`[]byte`与string的转化开销

  和上述提到的一样，这两种结构转化是带内存分配和拷贝开销的，这里fasthttp做了个trick避免开销。就是利用了string和slice在runtime里结构只差一个Cap字段实现：

```go
type StringHeader struct {
	Data uintptr
	Len  int
}

type SliceHeader struct {
	Data uintptr
	Len  int
	Cap  int
}

// []byte -> string
func b2s(b []byte) string {
	return *(*string)(unsafe.Pointer(&b))
}

// string -> []byte
func s2b(s string) []byte {
	sh := (*reflect.StringHeader)(unsafe.Pointer(&s))
	bh := reflect.SliceHeader{
		Data: sh.Data,
		Len:  sh.Len,
		Cap:  sh.Len,
	}
	return *(*[]byte)(unsafe.Pointer(&bh))
}
```

  不过这种trick的影响是：

1. 转换出来的[]byte不能有修改操作
2. 依赖了XXHeader结构，runtime更改结构会受到影响
3. 如果unsafe.Pointer作用被更改，也受到影响



### 总结

fasthttp [github](https://github.com/valyala/fasthttp#fasthttp-best-practices)中提到的:

- Do not allocate objects and `[]byte` buffers - just reuse them as much as possible. Fasthttp API design encourages this.
- [sync.Pool](https://golang.org/pkg/sync/#Pool) is your best friend.
- [Profile your program](http://blog.golang.org/profiling-go-programs) in production. `go tool pprof --alloc_objects your-program mem.pprof` usually gives better insights for optimization opportunities than `go tool pprof your-program cpu.pprof`.
- Write [tests and benchmarks](https://golang.org/pkg/testing/) for hot paths.
- Avoid conversion between `[]byte` and `string`, since this may result in memory allocation+copy. Fasthttp API provides functions for both `[]byte` and `string` - use these functions instead of converting manually between `[]byte` and `string`. There are some exceptions - see [this wiki page](https://github.com/golang/go/wiki/CompilerOptimizations#string-and-byte) for more details.
- Verify your tests and production code under [race detector](https://golang.org/doc/articles/race_detector.html) on a regular basis.
- Prefer [quicktemplate](https://github.com/valyala/quicktemplate) instead of [html/template](https://golang.org/pkg/html/template/) in your webserver.

总结下来一些要点就是:

