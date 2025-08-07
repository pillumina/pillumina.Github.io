---
title: "Profiling a Go Service in Production"
date: 2021-04-07T11:25:18+08:00
hero: /images/posts/golang2.jpg
menu:
  sidebar:
    name: Profling Go Service
    identifier: go-profiling
    parent: golang odyssey
    weight: 10
draft: false
---



## 参考

[Julia Evans: Profiling Go programs with pprof](https://jvns.ca/blog/2017/09/24/profiling-go-with-pprof/)

[How I investigated memory leaks in Go using pprof on a large codebase](https://www.freecodecamp.org/news/how-i-investigated-memory-leaks-in-go-using-pprof-on-a-large-codebase-4bec4325e192/)

[Memory Profiling a Go Service](https://medium.com/compass-true-north/memory-profiling-a-go-service-cd62b90619f9)

[Russ Cox: Profling Go Programs](https://blog.golang.org/pprof)

[Package pprof overview](https://golang.org/pkg/net/http/pprof/)

[github: pprof](https://github.com/google/pprof)



[Issue: Why 'Total MB' in golang heap profile is less than 'RES' in top?](https://stackoverflow.com/questions/16516189/why-total-mb-in-golang-heap-profile-is-less-than-res-in-top)

[Issue: Cannot free memory once occupied by bytes.Buffer](https://stackoverflow.com/questions/37382600/cannot-free-memory-once-occupied-by-bytes-buffer)

[Issue: FreeOSMemory() in production](https://stackoverflow.com/questions/42345060/freeosmemory-in-production)

[Issue: Is this an idiomatic worker thread pool in Go?](https://stackoverflow.com/questions/38170852/is-this-an-idiomatic-worker-thread-pool-in-go)