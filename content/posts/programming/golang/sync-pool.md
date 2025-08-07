---
title: "[源码分析]sync pool"
date: 2021-01-01T11:22:18+08:00
hero: /images/posts/golang2.jpg
menu:
  sidebar:
    name: Sync Pool源码分析
    identifier: sync-pool
    parent: golang odyssey
    weight: 10
draft: false
---

```
- 当多个goroutine都需要创建同一个对象，如果gorountine数过多，导致对象的创建数目剧增，进而导致GC压力增大，形成“并发大-占用内存大-GC缓慢-并发处理能力弱-并发更大”这样的恶性循环
- 在这个时候，需要一个对象池，每个goroutine不再自己单独创建对象，而是从对象池中取出一个对象（如果池中已有）
```

