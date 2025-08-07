---
title: "Docker Fundamentals: Cgroup"
date: 2021-04-05T11:22:18+08:00
hero: /images/posts/k8s-docker.jpg
menu:
  sidebar:
    name: Docker Fundamentals (Cgroup)
    identifier: docker-cgroup
    parent: cloud-computing
    weight: 10
draft: false
---

  Linux Namespace的技术解决了环境隔离的问题，不过这是虚拟化最基本的一步，我们另外需要解决**对计算机资源使用上的隔离**。说人话，就是虽然Namespace把我关到一个特定的环境，但是里面进程使用的CPU、内存、磁盘等计算资源实际上没有被限制。这个问题的解决，就要用到CGroup技术。

  Linux CGroup全称是Linux Control Group，也是其内核的一个功能，用于限制、控制和分离一个进程group的资源。最早这个项目是2006年由谷歌的工程师发起的，最开始名称是process containers（工程容器），后面觉得内核中容器这个名词被用烂了，就改名为cgroup。

  CGroup可以让你对系统中运行的进程的用户组分配资源-CPU时间、系统内存、网络带宽亦或者是这些的组合。同时，也可以监控你配置的cgroup，拒绝cgroup访问某些资源。主要提供的功能如下：

- `Resource Limitation`： 限制资源使用
- `Prioritization`: 优先级控制，例如CPU使用和磁盘IO吞吐
- `Accounting`：审计统计，主要用于计费
- `Control`：挂起进程，恢复执行进程

  在真正的实践当中，system admin一般会利用CGroup做以下的事：

- 对进程集合进行隔离，限制他们所消费的资源，例如绑定CPU core
- 为这组进程分配足够使用的内存
- 为这组进程分配响应的网络带宽和磁盘存储限制
- 限制访问某些设备（白名单）

  Linux实际上把CGroup实现成了一个文件系统，你可以mount。在linux环境输入下面的可以看到cgroup已经为你mount好：

```shell
derios@ubuntu:~$ mount -t cgroup
cgroup on /sys/fs/cgroup/cpuset type cgroup (rw,relatime,cpuset)
cgroup on /sys/fs/cgroup/cpu type cgroup (rw,relatime,cpu)
cgroup on /sys/fs/cgroup/cpuacct type cgroup (rw,relatime,cpuacct)
cgroup on /sys/fs/cgroup/memory type cgroup (rw,relatime,memory)
cgroup on /sys/fs/cgroup/devices type cgroup (rw,relatime,devices)
cgroup on /sys/fs/cgroup/freezer type cgroup (rw,relatime,freezer)
cgroup on /sys/fs/cgroup/blkio type cgroup (rw,relatime,blkio)
cgroup on /sys/fs/cgroup/net_prio type cgroup (rw,net_prio)
cgroup on /sys/fs/cgroup/net_cls type cgroup (rw,net_cls)
cgroup on /sys/fs/cgroup/perf_event type cgroup (rw,relatime,perf_event)
cgroup on /sys/fs/cgroup/hugetlb type cgroup (rw,relatime,hugetlb)
```

  可以看到，在`/sys/fs`下有`cgroup`目录，这个目录下面有各种子目录：`cpu`，`cpuset`，`memory`...。这些都是cgroup的子系统，分别用来干不同的事。

  如果没有看到上面的目录，你可以自己mount:

```shell
mkdir cgroup
mount -t tmpfs cgroup_root ./cgroup
mkdir cgroup/cpuset
mount -t cgroup -ocpuset cpuset ./cgroup/cpuset/
mkdir cgroup/cpu
mount -t cgroup -ocpu cpu ./cgroup/cpu/
mkdir cgroup/memory
mount -t cgroup -omemory memory ./cgroup/memory/
```

  mount成功以后，你会看见这些目录下有文件，比如下面展现的cpu和cpuset子系统:

```shell
derios@ubuntu:~$ ls /sys/fs/cgroup/cpu /sys/fs/cgroup/cpuset/ 
/sys/fs/cgroup/cpu:
cgroup.clone_children  cgroup.sane_behavior  cpu.shares         release_agent
cgroup.event_control   cpu.cfs_period_us     cpu.stat           tasks
cgroup.procs           cpu.cfs_quota_us      notify_on_release  user
/sys/fs/cgroup/cpuset/:
cgroup.clone_children  cpuset.mem_hardwall             cpuset.sched_load_balance
cgroup.event_control   cpuset.memory_migrate           cpuset.sched_relax_domain_level
cgroup.procs           cpuset.memory_pressure          notify_on_release
cgroup.sane_behavior   cpuset.memory_pressure_enabled  release_agent
cpuset.cpu_exclusive   cpuset.memory_spread_page       tasks
cpuset.cpus            cpuset.memory_spread_slab       user
cpuset.mem_exclusive   cpuset.mems
```

  你可以进入`/sys/fs/cgroup`的各个目录下去创建个dir，你会发现一旦你创建了，这个子目录下又有很多文件：

```shell
derios@ubuntu:/sys/fs/cgroup/cpu$ sudo mkdir haoel
[sudo] password for derios: 
derios@ubuntu:/sys/fs/cgroup/cpu$ ls ./haoel
cgroup.clone_children  cgroup.procs       cpu.cfs_quota_us  cpu.stat           tasks
cgroup.event_control   cpu.cfs_period_us  cpu.shares        notify_on_release
```

  

## CPU限制

  假设我们写了个吃CPU的程序：

```c++
int main(void)
{
    int i = 0;
    for(;;) i++;
    return 0;
}
```

  用sudo执行毫无疑问cpu飚到100%：

```shell
  PID USER      PR  NI    VIRT    RES    SHR S %CPU %MEM     TIME+ COMMAND     
 3529 root      20   0    4196    736    656 R 99.6  0.1   0:23.13 deadloop   
```

  在上面我们在`/sys/fs/cgroup/cpu`下面创建了`haoel`的group，我们首先设置下cpu限制

```shell
derios@ubuntu:~# cat /sys/fs/cgroup/cpu/haoel/cpu.cfs_quota_us 
-1
root@ubuntu:~# echo 20000 > /sys/fs/cgroup/cpu/haoel/cpu.cfs_quota_us
```

  我们top出来看到这个线程的pid是3529，我们加到这个cgroup中:

```shell
# echo 3529 >> /sys/fs/cgroup/cpu/haoel/tasks
```

  然后再看top，可以发现CPU使用率一下降成了20%（设置的20000代表着20%）：

```shell
  PID USER      PR  NI    VIRT    RES    SHR S %CPU %MEM     TIME+ COMMAND     
 3529 root      20   0    4196    736    656 R 19.9  0.1   8:06.11 deadloop    
```

  下面贴一个我google的代码示例，用于参考看看注释:

```c++
#define _GNU_SOURCE         /* See feature_test_macros(7) */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>


const int NUM_THREADS = 5;

void *thread_main(void *threadid)
{
    /* 把自己加入cgroup中（syscall(SYS_gettid)为得到线程的系统tid） */
    char cmd[128];
    sprintf(cmd, "echo %ld >> /sys/fs/cgroup/cpu/haoel/tasks", syscall(SYS_gettid));
    system(cmd); 
    sprintf(cmd, "echo %ld >> /sys/fs/cgroup/cpuset/haoel/tasks", syscall(SYS_gettid));
    system(cmd);

    long tid;
    tid = (long)threadid;
    printf("Hello World! It's me, thread #%ld, pid #%ld!\n", tid, syscall(SYS_gettid));
    
    int a=0; 
    while(1) {
        a++;
    }
    pthread_exit(NULL);
}
int main (int argc, char *argv[])
{
    int num_threads;
    if (argc > 1){
        num_threads = atoi(argv[1]);
    }
    if (num_threads<=0 || num_threads>=100){
        num_threads = NUM_THREADS;
    }

    /* 设置CPU利用率为50% */
    mkdir("/sys/fs/cgroup/cpu/haoel", 755);
    system("echo 50000 > /sys/fs/cgroup/cpu/haoel/cpu.cfs_quota_us");

    mkdir("/sys/fs/cgroup/cpuset/haoel", 755);
    /* 限制CPU只能使用#2核和#3核 */
    system("echo \"2,3\" > /sys/fs/cgroup/cpuset/haoel/cpuset.cpus");

    pthread_t* threads = (pthread_t*) malloc (sizeof(pthread_t)*num_threads);
    int rc;
    long t;
    for(t=0; t<num_threads; t++){
        printf("In main: creating thread %ld\n", t);
        rc = pthread_create(&threads[t], NULL, thread_main, (void *)t);
        if (rc){
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    /* Last thing that main() should do */
    pthread_exit(NULL);
    free(threads);
}
```



## 内存使用限制

我们构造一下无限分配内存的例子：

```c++
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

int main(void)
{
    int size = 0;
    int chunk_size = 512;
    void *p = NULL;

    while(1) {

        if ((p = malloc(p, chunk_size)) == NULL) {
            printf("out of memory!!\n");
            break;
        }
        memset(p, 1, chunk_size);
        size += chunk_size;
        printf("[%d] - memory is allocated [%8d] bytes \n", getpid(), size);
        sleep(1);
    }
    return 0;
}
```

  这个代码是个dead loop，每隔1秒会分配512字节的内存。然后我们再cgroup搞事情：

```shell
# 创建memory cgroup
$ mkdir /sys/fs/cgroup/memory/haoel
$ echo 64k > /sys/fs/cgroup/memory/haoel/memory.limit_in_bytes

# 把上面的进程的pid加入这个cgroup
$ echo [pid] > /sys/fs/cgroup/memory/haoel/tasks 
```

   可以看到上面的进程会因为内存oom被kill。

## 磁盘IO限制

  我们先查看下磁盘IO，模拟一下（从/dev/sda1读取数据，输出到/dev/null上）：

```shell
sudo dd if=/dev/sda1 of=/dev/null
```

  通过`iotop`命令可以看到相关的io速度为55MB/s：

```shell
  TID  PRIO  USER     DISK READ  DISK WRITE  SWAPIN     IO>    COMMAND          
 8128 be/4 root       55.74 M/s    0.00 B/s  0.00 % 85.65 % dd if=/de~=/dev/null...
```

  然后，我们先创建一个blkio（块设备IO）的cgroup:

```shell
mkdir /sys/fs/cgroup/blkio/haoel
```

  并把读IO限制到1MB/s，并把前面那个dd命令的pid放进去（注：8:0 是设备号，你可以通过ls -l /dev/sda1获得）：

```shell
root@ubuntu:~# echo '8:0 1048576'  > /sys/fs/cgroup/blkio/haoel/blkio.throttle.read_bps_device 
root@ubuntu:~# echo 8128 > /sys/fs/cgroup/blkio/haoel/tasks
```

  再用iotop命令，你马上就能看到读速度被限制到了1MB/s左右:

```shell
  TID  PRIO  USER     DISK READ  DISK WRITE  SWAPIN     IO>    COMMAND          
 8128 be/4 root      973.20 K/s    0.00 B/s  0.00 % 94.41 % dd if=/de~=/dev/null...
```

  

## CGroup的子系统

OK，有了以上比较感性的认识，我们来看看control group到底有哪些子系统：

- blkio — 这个子系统为块设备设定输入/输出限制，比如物理设备（磁盘，固态硬盘，USB 等等）。
- cpu — 这个子系统使用调度程序提供对 CPU 的 cgroup 任务访问。
- cpuacct — 这个子系统自动生成 cgroup 中任务所使用的 CPU 报告。
- cpuset — 这个子系统为 cgroup 中的任务分配独立 CPU（在多核系统）和内存节点。
- devices — 这个子系统可允许或者拒绝 cgroup 中的任务访问设备。
- freezer — 这个子系统挂起或者恢复 cgroup 中的任务。
- memory — 这个子系统设定 cgroup 中任务使用的内存限制，并自动生成内存资源使用报告。
- net_cls — 这个子系统使用等级识别符（classid）标记网络数据包，可允许 Linux 流量控制程序（tc）识别从具体 cgroup 中生成的数据包。
- net_prio — 这个子系统用来设计网络流量的优先级
- hugetlb — 这个子系统主要针对于HugeTLB系统进行限制，这是一个大页文件系统。

  在我的Ubuntu14.04虚拟机下看不到net_cls和net_prio这两个cgroup，需要手动mount：

```shell
$ sudo modprobe cls_cgroup
$ sudo mkdir /sys/fs/cgroup/net_cls
$ sudo mount -t cgroup -o net_cls none /sys/fs/cgroup/net_cls

$ sudo modprobe netprio_cgroup
$ sudo mkdir /sys/fs/cgroup/net_prio
$ sudo mount -t cgroup -o net_prio none /sys/fs/cgroup/net_prio
```

  

## CGroup术语

CGroup有下述术语：

- **任务（Tasks）**：就是系统的一个进程。
- **控制组（Control Group）**：一组按照某种标准划分的进程，比如官方文档中的Professor和Student，或是WWW和System之类的，其表示了某进程组。Cgroups中的资源控制都是以控制组为单位实现。一个进程可以加入到某个控制组。而资源的限制是定义在这个组上，就像上面示例中我用的haoel一样。简单点说，cgroup的呈现就是一个目录带一系列的可配置文件。
- **层级（Hierarchy）**：控制组可以组织成hierarchical的形式，既一颗控制组的树（目录结构）。控制组树上的子节点继承父结点的属性。简单点说，hierarchy就是在一个或多个子系统上的cgroups目录树。
- **子系统（Subsystem）**：一个子系统就是一个资源控制器，比如CPU子系统就是控制CPU时间分配的一个控制器。子系统必须附加到一个层级上才能起作用，一个子系统附加到某个层级以后，这个层级上的所有控制族群都受到这个子系统的控制。Cgroup的子系统可以有很多，也在不断增加中。