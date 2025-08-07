---
title: "Docker Fundamentals: Namespace"
date: 2021-04-01T11:22:18+08:00
hero: /images/posts/k8s-docker.jpg
menu:
  sidebar:
    name: Docker Fundamentals (Namespace)
    identifier: docker-namespace
    parent: cloud-computing
    weight: 10
draft: false
---

容器技术出现已经很久，只不过Docker容器平台的出现它变火了。Docker是第一个让容器能在不同机器之间移植的系统，它简化了打包应用的流程，也简化了打包应用的库和各种依赖。思考下整个OS的file system能直接被打包成一个简单的可移植的包，一开始的时候概念上还是很有趣的。



有时候我认为自己的阅读比较碎片化(~~short-term memory越来越少~~)，所以我想把之前学习容器知识的一些基础技术再整理出来，也算是给自己学习的反馈。这个基础系列从Linux Namespace开始，后续会陆续介绍比如cgroup、aufs、devicemapper等技术。



## 参考

[Namespace in operation](https://lwn.net/Articles/531114/)

[Linux namespace man page](https://man7.org/linux/man-pages/man7/namespaces.7.html)

[Introduction to linux namespace](https://blog.jtlebi.fr/2013/12/22/introduction-to-linux-namespaces-part-1-uts/)

## 什么是Namespace

简单来说，linux namespace是Linux提供的一种内核级别环境隔离的方法。在早期的Unix中，提供了一种叫做chroot的系统调用：通过修改root目录把用户关到一个特定的目录下面。这种就是简单的隔离方式，也就是chroot内部的file system无法访问外部的内容。Linux Namespace在此基础之上，提供了对UTS、IPC、mount、network、PID、User等隔离机制。

这里可以简单举例，比如Linux的超级父进程的PID为1，如果我们可以把用户的进程空间关到某个进程分支之下，并且像chroot那样能够让下面的进程看到那个超级父进程的PID为1，而不同PID Namespace中的进程无法看到彼此，这样就能达到进程隔离。

Linux Namespace有以下的种类，供给后续参考（刚看有个印象就行）：

| 分类                   | 系统调用参数  | 相关内核版本                                                 |
| :--------------------- | :------------ | :----------------------------------------------------------- |
| **Mount namespaces**   | CLONE_NEWNS   | [Linux 2.4.19](http://lwn.net/2001/0301/a/namespaces.php3)   |
| **UTS namespaces**     | CLONE_NEWUTS  | [Linux 2.6.19](http://lwn.net/Articles/179345/)              |
| **IPC namespaces**     | CLONE_NEWIPC  | [Linux 2.6.19](http://lwn.net/Articles/187274/)              |
| **PID namespaces**     | CLONE_NEWPID  | [Linux 2.6.24](http://lwn.net/Articles/259217/)              |
| **Network namespaces** | CLONE_NEWNET  | [始于Linux 2.6.24 完成于 Linux 2.6.29](http://lwn.net/Articles/219794/) |
| **User namespaces**    | CLONE_NEWUSER | [始于 Linux 2.6.23 完成于 Linux 3.8)](http://lwn.net/Articles/528078/) |

其主要涉及到三个系统调用：

1. `clone()`： 实现线程的系统调用，用来创建新的线程，并可通过涉及上述参数做到隔离
2. `unshare()`： 让某一个线程脱离某namespace
3. `setns()`: 把某一个线程加到某namespace

如果读者你想看具体的实例，请自己man一下(关注一下自己的linux虚拟机内核)，或者google一下，我这里贴一个`clone()`的source code：

```c++
#define _GNU_SOURCE
#include <sys/types.h>
#include <sys/wait.h>
#include <stdio.h>
#include <sched.h>
#include <signal.h>
#include <unistd.h>
/* 定义一个给 clone 用的栈，栈大小1M */
#define STACK_SIZE (1024 * 1024)
static char container_stack[STACK_SIZE];
char* const container_args[] = {
    "/bin/bash",
    NULL
};
int container_main(void* arg)
{
    printf("Container - inside the container!\n");
    /* 直接执行一个shell，以便我们观察这个进程空间里的资源是否被隔离了 */
    execv(container_args[0], container_args); 
    printf("Something's wrong!\n");
    return 1;
}
int main()
{
    printf("Parent - start a container!\n");
    /* 调用clone函数，其中传出一个函数，还有一个栈空间的（为什么传尾指针，因为栈是反着的） */
    int container_pid = clone(container_main, container_stack+STACK_SIZE, SIGCHLD, NULL);
    /* 等待子进程结束 */
    waitpid(container_pid, NULL, 0);
    printf("Parent - container stopped!\n");
    return 0;
}
```

上面的程序注释写的比较明白，和pthreads差不多。不过这个程序里，父子进程的进程空间没有什么区别，父进程能访问到的明显子进程也能访问。

我们用几个例子来看看linux的namespace到底是啥样的，运行的虚拟机为`ubuntu14.4`



## UTS Namespace

这里略去一些头文件和数据结构的定义：

```c++
int container_main(void* arg)
{
    printf("Container - inside the container!\n");
    sethostname("container",10); /* 设置hostname */
    execv(container_args[0], container_args);
    printf("Something's wrong!\n");
    return 1;
}
int main()
{
    printf("Parent - start a container!\n");
    int container_pid = clone(container_main, container_stack+STACK_SIZE, 
            CLONE_NEWUTS | SIGCHLD, NULL); /*启用CLONE_NEWUTS Namespace隔离 */
    waitpid(container_pid, NULL, 0);
    printf("Parent - container stopped!\n");
    return 0;
}
```

`执行上述的c++程序，会发现子进程的hostname变为了container`

```shell
derios@ubuntu:~$ sudo ./uts
Parent - start a container!
Container - inside the container!
root@container:~# hostname
container
root@container:~# uname -n
container
```



## IPC Namespace

IPC(Inter-Process Communication)，是Unix/Linux下的一种通信方式。IPC有共享内存、信号量、消息队列等方法。所以如果要隔离我们也要把IPC进行隔离。换句话说，这样可以保证只有在同一个namespace下的进程之间才能互相通信。目前我对IPC的原理没什么研究，查了查资料，IPC需要有个全局的ID，那么如果我们要做隔离，namespace肯定需要对这个全局ID进行隔离，不能和其他namespace中的进程共享。

要启动IPC隔离，我们需要在`clone`时加上`CLONE_NEWPIC`的参数:

```c++
int container_pid = clone(container_main, container_stack+STACK_SIZE, 
            CLONE_NEWUTS | CLONE_NEWIPC | SIGCHLD, NULL);
```

我们先创建一个IPC的queue，下面的全局ID为0：

```shell
derios@ubuntu:~$ ipcmk -Q 
Message queue id: 0
derios@ubuntu:~$ ipcs -q
------ Message Queues --------
key        msqid      owner      perms      used-bytes   messages    
0xd0d56eb2 0          hchen      644        0            0
```

如果我们不加`CLONE_NEWIPC`参数运行程序，我们可以看到在子进程中还是能看到全局的IPC queue：

```shell
derios@ubuntu:~$ sudo ./uts 
Parent - start a container!
Container - inside the container!
root@container:~# ipcs -q
------ Message Queues --------
key        msqid      owner      perms      used-bytes   messages    
0xd0d56eb2 0          hchen      644        0            0
```

如果我们运行加上了`CLONE_NEWIPC`的程序，可以有如下的结果:

```shell
root@ubuntu:~$ sudo./ipc
Parent - start a container!
Container - inside the container!
root@container:~/linux_namespace# ipcs -q
------ Message Queues --------
key        msqid      owner      perms      used-bytes   messages
```

可见IPC已经被隔离。



## PID Namespace

我们继续修改上述的程序：

```c++
int container_main(void* arg)
{
    /* 查看子进程的PID，我们可以看到其输出子进程的 pid 为 1 */
    printf("Container [%5d] - inside the container!\n", getpid());
    sethostname("container",10);
    execv(container_args[0], container_args);
    printf("Something's wrong!\n");
    return 1;
}
int main()
{
    printf("Parent [%5d] - start a container!\n", getpid());
    /*启用PID namespace - CLONE_NEWPID*/
    int container_pid = clone(container_main, container_stack+STACK_SIZE, 
            CLONE_NEWUTS | CLONE_NEWPID | SIGCHLD, NULL); 
    waitpid(container_pid, NULL, 0);
    printf("Parent - container stopped!\n");
    return 0;
}
```

运行看一下，发现子进程的PID为1：

```shell
derios@ubuntu:~$ sudo ./pid
Parent [ 3474] - start a container!
Container [ 1] - inside the container!
root@container:~# echo $$
1
```

这里的1有啥意义，你可能会问。其实在传统UNIX系统中，PID为1的进程地位比较特殊，指代`init`

，作为所有进程的父进程，有非常多的特权（信号屏蔽etc.），此外它还会检查所有进程的状态，而且如果子进程脱离了父进程（父进程没有wait它），那么`init`会负责回收资源并结束这个子进程。所以，要做到进程空间的隔离，首先要创建pid为1的进程，比如可以像chroot一样，把子进程的pid在容器内变为1。

不过，很奇怪的是，**我们在子进程的shell里执行top, ps等命令，还是可以看到所有的进程。**这意味着隔离并没有完全。因为像ps, top这些命令会读取`/proc`文件系统，而因为`/proc`文件系统在父子进程里都是一样的，所以命令的回显也都是一样的。

因此，我们还要做到**对文件系统的隔离**。



## Mount Namespace

下面的程序，我们在启用mount namespace并在子进程中重新mount了`/proc`文件系统。

```c++
int container_main(void* arg)
{
    printf("Container [%5d] - inside the container!\n", getpid());
    sethostname("container",10);
    /* 重新mount proc文件系统到 /proc下 */
    system("mount -t proc proc /proc");
    execv(container_args[0], container_args);
    printf("Something's wrong!\n");
    return 1;
}
int main()
{
    printf("Parent [%5d] - start a container!\n", getpid());
    /* 启用Mount Namespace - 增加CLONE_NEWNS参数 */
    int container_pid = clone(container_main, container_stack+STACK_SIZE, 
            CLONE_NEWUTS | CLONE_NEWPID | CLONE_NEWNS | SIGCHLD, NULL);
    waitpid(container_pid, NULL, 0);
    printf("Parent - container stopped!\n");
    return 0;
}
```

执行结果如下:

```shell
derioshen@ubuntu:~$ sudo ./pid.mnt
Parent [ 3502] - start a container!
Container [    1] - inside the container!
root@container:~# ps -elf 
F S UID        PID  PPID  C PRI  NI ADDR SZ WCHAN  STIME TTY          TIME CMD
4 S root         1     0  0  80   0 -  6917 wait   19:55 pts/2    00:00:00 /bin/bash
0 R root        14     1  0  80   0 -  5671 -      19:56 pts/2    00:00:00 ps -elf
```

我们看到只有2个进程了，`pid=1`的是我们的`/bin/bash`，同时再看看`/proc`目录，也变得比较干净:

```shell
root@container:~# ls /proc
1          dma          key-users   net            sysvipc
16         driver       kmsg        pagetypeinfo   timer_list
acpi       execdomains  kpagecount  partitions     timer_stats
asound     fb           kpageflags  sched_debug    tty
buddyinfo  filesystems  loadavg     schedstat      uptime
bus        fs           locks       scsi           version
cgroups    interrupts   mdstat      self           version_signature
cmdline    iomem        meminfo     slabinfo       vmallocinfo
consoles   ioports      misc        softirqs       vmstat
cpuinfo    irq          modules     stat           zoneinfo
crypto     kallsyms     mounts      swaps
devices    kcore        mpt         sys
diskstats  keys         mtrr        sysrq-trigger
```

通过`CLONE_NEWNS`创建mount namespace以后，父进程会把自己的文件结构复制给子进程。而子进程中新的namespace中所有mount操作都只会影响自身的文件系统，不会对外界产生任何影响，这就做到了严格的隔离。

那么我们是不是还有别的一些文件系统也要mount？答案是肯定的。



## Docker的Mount Namespace

我们可以简单搞个小的镜像，这种玩法是我google参考来的，模仿docker的mount namespace。

首先，我们需要一个rootfs， 也就是我们需要把我们要做的镜像中的那些命令什么的copy到一个rootfs的目录下，我们模仿Linux构建如下的目录：

```shell
derios@ubuntu:~/rootfs$ ls
bin  dev  etc  home  lib  lib64  mnt  opt  proc  root  run  sbin  sys  tmp  usr  var
```

然后，我们把一些我们需要的命令copy到 rootfs/bin目录中（sh命令必需要copy进去，不然我们无法 chroot ）：

```shell
derios@ubuntu:~/rootfs$ ls ./bin ./usr/bin
 
./bin:
bash   chown  gzip      less  mount       netstat  rm     tabs  tee      top       tty
cat    cp     hostname  ln    mountpoint  ping     sed    tac   test     touch     umount
chgrp  echo   ip        ls    mv          ps       sh     tail  timeout  tr        uname
chmod  grep   kill      more  nc          pwd      sleep  tar   toe      truncate  which

./usr/bin:
awk  env  groups  head  id  mesg  sort  strace  tail  top  uniq  vi  wc  xargs
```

注：你可以使用ldd命令把这些命令相关的那些so文件copy到对应的目录：

```shell
derios@ubuntu:~/rootfs/bin$ ldd bash
  linux-vdso.so.1 =>  (0x00007fffd33fc000)
  libtinfo.so.5 => /lib/x86_64-linux-gnu/libtinfo.so.5 (0x00007f4bd42c2000)
  libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f4bd40be000)
  libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f4bd3cf8000)
  /lib64/ld-linux-x86-64.so.2 (0x00007f4bd4504000)
```

下面是我的rootfs中的一些so文件：

```shell
derios@ubuntu:~/rootfs$ ls ./lib64 ./lib/x86_64-linux-gnu/

./lib64:
ld-linux-x86-64.so.2

./lib/x86_64-linux-gnu/:
libacl.so.1      libmemusage.so         libnss_files-2.19.so    libpython3.4m.so.1
libacl.so.1.1.0  libmount.so.1          libnss_files.so.2       libpython3.4m.so.1.0
libattr.so.1     libmount.so.1.1.0      libnss_hesiod-2.19.so   libresolv-2.19.so
libblkid.so.1    libm.so.6              libnss_hesiod.so.2      libresolv.so.2
libc-2.19.so     libncurses.so.5        libnss_nis-2.19.so      libselinux.so.1
libcap.a         libncurses.so.5.9      libnss_nisplus-2.19.so  libtinfo.so.5
libcap.so        libncursesw.so.5       libnss_nisplus.so.2     libtinfo.so.5.9
libcap.so.2      libncursesw.so.5.9     libnss_nis.so.2         libutil-2.19.so
libcap.so.2.24   libnsl-2.19.so         libpcre.so.3            libutil.so.1
libc.so.6        libnsl.so.1            libprocps.so.3          libuuid.so.1
libdl-2.19.so    libnss_compat-2.19.so  libpthread-2.19.so      libz.so.1
libdl.so.2       libnss_compat.so.2     libpthread.so.0
libgpm.so.2      libnss_dns-2.19.so     libpython2.7.so.1
libm-2.19.so     libnss_dns.so.2        libpython2.7.so.1.0
```

包括这些命令依赖的一些配置文件：

```shell
derios@ubuntu:~/rootfs$ ls ./etc
bash.bashrc  group  hostname  hosts  ld.so.cache  nsswitch.conf  passwd  profile  
resolv.conf  shadow
```

看到现在你可能比较懵逼，有的比较熟悉os的同学也可能会问：有的配置希望是在容器起动时给他设置的，而不是hard code在镜像中的。比如：/etc/hosts，/etc/hostname，还有DNS的/etc/resolv.conf文件。OK, 那我们在rootfs外面，我们再创建一个conf目录，把这些文件放到这个目录中:

```shell
derios@ubuntu:~$ ls ./conf
hostname     hosts     resolv.conf
```

这样，我们的父进程就可以动态地设置容器需要的这些文件的配置， 然后再把他们mount进容器，这样，容器的镜像中的配置就比较灵活了。

接下来是程序(~~Google真好~~)

```c++
#define _GNU_SOURCE
#include <sys types.h="">
#include <sys wait.h="">
#include <sys mount.h="">
#include <stdio.h>
#include <sched.h>
#include <signal.h>
#include <unistd.h>

#define STACK_SIZE (1024 * 1024)

static char container_stack[STACK_SIZE];
char* const container_args[] = {
    "/bin/bash",
    "-l",
    NULL
};

int container_main(void* arg)
{
    printf("Container [%5d] - inside the container!\n", getpid());

    //set hostname
    sethostname("container",10);

    //remount "/proc" to make sure the "top" and "ps" show container's information
    if (mount("proc", "rootfs/proc", "proc", 0, NULL) !=0 ) {
        perror("proc");
    }
    if (mount("sysfs", "rootfs/sys", "sysfs", 0, NULL)!=0) {
        perror("sys");
    }
    if (mount("none", "rootfs/tmp", "tmpfs", 0, NULL)!=0) {
        perror("tmp");
    }
    if (mount("udev", "rootfs/dev", "devtmpfs", 0, NULL)!=0) {
        perror("dev");
    }
    if (mount("devpts", "rootfs/dev/pts", "devpts", 0, NULL)!=0) {
        perror("dev/pts");
    }
    if (mount("shm", "rootfs/dev/shm", "tmpfs", 0, NULL)!=0) {
        perror("dev/shm");
    }
    if (mount("tmpfs", "rootfs/run", "tmpfs", 0, NULL)!=0) {
        perror("run");
    }
    /* 
     * 模仿Docker的从外向容器里mount相关的配置文件 
     * 你可以查看：/var/lib/docker/containers/<container_id>/目录，
     * 你会看到docker的这些文件的。
     */
    if (mount("conf/hosts", "rootfs/etc/hosts", "none", MS_BIND, NULL)!=0 ||
          mount("conf/hostname", "rootfs/etc/hostname", "none", MS_BIND, NULL)!=0 ||
          mount("conf/resolv.conf", "rootfs/etc/resolv.conf", "none", MS_BIND, NULL)!=0 ) {
        perror("conf");
    }
    /* 模仿docker run命令中的 -v, --volume=[] 参数干的事 */
    if (mount("/tmp/t1", "rootfs/mnt", "none", MS_BIND, NULL)!=0) {
        perror("mnt");
    }

    /* chroot 隔离目录 */
    if ( chdir("./rootfs") != 0 || chroot("./") != 0 ){
        perror("chdir/chroot");
    }

    execv(container_args[0], container_args);
    perror("exec");
    printf("Something's wrong!\n");
    return 1;
}

int main()
{
    printf("Parent [%5d] - start a container!\n", getpid());
    int container_pid = clone(container_main, container_stack+STACK_SIZE, 
            CLONE_NEWUTS | CLONE_NEWIPC | CLONE_NEWPID | CLONE_NEWNS | SIGCHLD, NULL);
    waitpid(container_pid, NULL, 0);
    printf("Parent - container stopped!\n");
    return 0;
}
</container_id></unistd.h></signal.h></sched.h></stdio.h></sys></sys></sys>
```

sudo运行上面的程序，你会看到下面的挂载信息以及一个所谓的“镜像”：

```shell
derios@ubuntu:~$ sudo ./mount 
Parent [ 4517] - start a container!
Container [    1] - inside the container!
root@container:/# mount
proc on /proc type proc (rw,relatime)
sysfs on /sys type sysfs (rw,relatime)
none on /tmp type tmpfs (rw,relatime)
udev on /dev type devtmpfs (rw,relatime,size=493976k,nr_inodes=123494,mode=755)
devpts on /dev/pts type devpts (rw,relatime,mode=600,ptmxmode=000)
tmpfs on /run type tmpfs (rw,relatime)
/dev/disk/by-uuid/18086e3b-d805-4515-9e91-7efb2fe5c0e2 on /etc/hosts type ext4 (rw,relatime,errors=remount-ro,data=ordered)
/dev/disk/by-uuid/18086e3b-d805-4515-9e91-7efb2fe5c0e2 on /etc/hostname type ext4 (rw,relatime,errors=remount-ro,data=ordered)
/dev/disk/by-uuid/18086e3b-d805-4515-9e91-7efb2fe5c0e2 on /etc/resolv.conf type ext4 (rw,relatime,errors=remount-ro,data=ordered)

root@container:/# ls /bin /usr/bin
/bin:
bash   chmod  echo  hostname  less  more  mv   ping  rm   sleep  tail  test    top   truncate  uname
cat    chown  grep  ip        ln    mount  nc   ps    sed  tabs   tar   timeout  touch  tty     which
chgrp  cp     gzip  kill      ls    mountpoint  netstat  pwd   sh   tac    tee   toe    tr   umount

/usr/bin:
awk  env  groups  head  id  mesg  sort  strace  tail  top  uniq  vi  wc  xargs
```



## User Namespace

User Namespace 主要用到了`CLONE_NEWUSER`参数，当我们使用这个参数以后，在内部看到的UID和GID和外部就不一样了，默认为65534。因为容器找不到其真正的UID，因此设置成了最大的UID（`/proc/sys/kernel/overflowuid`）。

要把容器中的uid和真实系统的uid给映射在一起，需要修改 **/proc/<pid>/uid_map** 和 **/proc/<pid>/gid_map** 这两个文件。这两个文件的格式为：

`ID-inside-ns ID-outside-ns length`

- 第一个字段ID-inside-ns表示在容器显示的UID或GID，
- 第二个字段ID-outside-ns表示容器外映射的真实的UID或GID。
- 第三个字段表示映射的范围，一般填1，表示一一对应。

比如，把真实的uid=1000映射成容器内的uid=0

```shell
$ cat /proc/2465/uid_map
         0       1000          1
```

再比如下面的示例：表示把namespace内部从0开始的uid映射到外部从0开始的uid，其最大范围是无符号32位整形：

```shell
$ cat /proc/$$/uid_map
         0          0          4294967295
```

- 写这两个文件的进程需要这个namespace中的CAP_SETUID (CAP_SETGID)权限（可参看[Capabilities](http://man7.org/linux/man-pages/man7/capabilities.7.html)）
- 写入的进程必须是此user namespace的父或子的user namespace进程。
- 另外需要满如下条件之一：1）父进程将effective uid/gid映射到子进程的user namespace中，2）父进程如果有CAP_SETUID/CAP_SETGID权限，那么它将可以映射到父进程中的任一uid/gid。



User Namespace是以普通用户运行，但是别的Namespace需要root权限，那么，如果我要同时使用多个Namespace，该怎么办呢？一般来说，我们先用一般用户创建User Namespace，然后把这个一般用户映射成root，在容器内用root来创建其它的Namesapce。



## Network Namespace

在linux中，一般用`ip`命令创建network namespace。不过在docker源码中并没有使用`ip`，而是自己实现了ip命令的一些内容。在这还是用`ip`命令描述一下做了啥。

首先我们来看一个图，这个图是Docker在host主机上的网络示意图

![host network](https://coolshell.cn/wp-content/uploads/2015/04/network.namespace.jpg)



实际上图还是有问题的，因为Docker也可以运行在虚拟机中，所以所谓的物理网卡其实也就是一个有能够路由的IP的网卡。

图中Docker用了一个私有的网段: 172.40.1.0，此外docker还会使用10.0.0.0以及192.168.0.0两个私有网段。如果你机器的路由表配置了(占用)所有的私有网段，那么docker就会无法启动。

启动docker以后，可以使用`ip link show`和`ip addr show`来查看目前宿主机的网络情况。这里我在`minikube`容器里执行了指令:

```shell
root@minikube:/# ip link show
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN mode DEFAULT group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
2: tunl0@NONE: <NOARP> mtu 1480 qdisc noop state DOWN mode DEFAULT group default qlen 1000
    link/ipip 0.0.0.0 brd 0.0.0.0
3: ip6tnl0@NONE: <NOARP> mtu 1452 qdisc noop state DOWN mode DEFAULT group default qlen 1000
    link/tunnel6 :: brd ::
4: docker0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP mode DEFAULT group default
    link/ether 02:42:2e:11:a4:ae brd ff:ff:ff:ff:ff:ff
6: vetha68cfee@if5: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue master docker0 state UP mode DEFAULT group default
    link/ether 46:2d:cf:22:79:9f brd ff:ff:ff:ff:ff:ff link-netnsid 1
12: veth8b65072@if11: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue master docker0 state UP mode DEFAULT group default
    link/ether 3e:fb:54:6d:ae:1e brd ff:ff:ff:ff:ff:ff link-netnsid 3
14: veth61918b0@if13: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue master docker0 state UP mode DEFAULT group default
    link/ether d2:68:38:76:73:c9 brd ff:ff:ff:ff:ff:ff link-netnsid 4
18: vethd7fa219@if17: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue master docker0 state UP mode DEFAULT group default
    link/ether 1e:14:79:68:f1:50 brd ff:ff:ff:ff:ff:ff link-netnsid 5
20: eth0@if21: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP mode DEFAULT group default
    link/ether 02:42:c0:a8:31:02 brd ff:ff:ff:ff:ff:ff link-netnsid 0
21: veth3b477c9@if19: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue master docker0 state UP mode DEFAULT group default
    link/ether 7e:b1:5d:53:00:ff brd ff:ff:ff:ff:ff:ff link-netnsid 2
23: veth265059d@if22: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue master docker0 state UP mode DEFAULT group default
    link/ether 4a:22:a4:05:79:e1 brd ff:ff:ff:ff:ff:ff link-netnsid 6
```

可以看到有`docker0`还有一些虚拟网卡。

为了能够做成这样，我又google了一段代码:

```shell
## 首先，我们先增加一个网桥lxcbr0，模仿docker0
brctl addbr lxcbr0
brctl stp lxcbr0 off
ifconfig lxcbr0 192.168.10.1/24 up #为网桥设置IP地址

## 接下来，我们要创建一个network namespace - ns1

# 增加一个namesapce 命令为 ns1 （使用ip netns add命令）
ip netns add ns1 

# 激活namespace中的loopback，即127.0.0.1（使用ip netns exec ns1来操作ns1中的命令）
ip netns exec ns1   ip link set dev lo up 

## 然后，我们需要增加一对虚拟网卡

# 增加一个pair虚拟网卡，注意其中的veth类型，其中一个网卡要按进容器中
ip link add veth-ns1 type veth peer name lxcbr0.1

# 把 veth-ns1 按到namespace ns1中，这样容器中就会有一个新的网卡了
ip link set veth-ns1 netns ns1

# 把容器里的 veth-ns1改名为 eth0 （容器外会冲突，容器内就不会了）
ip netns exec ns1  ip link set dev veth-ns1 name eth0 

# 为容器中的网卡分配一个IP地址，并激活它
ip netns exec ns1 ifconfig eth0 192.168.10.11/24 up


# 上面我们把veth-ns1这个网卡按到了容器中，然后我们要把lxcbr0.1添加上网桥上
brctl addif lxcbr0 lxcbr0.1

# 为容器增加一个路由规则，让容器可以访问外面的网络
ip netns exec ns1     ip route add default via 192.168.10.1

# 在/etc/netns下创建network namespce名称为ns1的目录，
# 然后为这个namespace设置resolv.conf，这样，容器内就可以访问域名了
mkdir -p /etc/netns/ns1
echo "nameserver 8.8.8.8" > /etc/netns/ns1/resolv.conf
```

上述就是docker网络的原理，但是有几点要注意的

1. docker的resolv.conf（配置系统DNS解析器）没有采取这样的方式，而是采取类似上述`mount namespace`的方式
2. 另外，docker采用进程的PID来做network namepspace的名称

我们原理了解了，甚至可以给正在运行的docker容器新增网卡:

```shell
ip link add peerA type veth peer name peerB 
brctl addif docker0 peerA 
ip link set peerA up 
ip link set peerB netns ${container-pid} 
ip netns exec ${container-pid} ip link set dev peerB name eth1 
ip netns exec ${container-pid} ip link set eth1 up ; 
ip netns exec ${container-pid} ip addr add ${ROUTEABLE_IP} dev eth1 ;
```

上述指令即为一个正在运行的docker容器新增一个`eth1`网卡，以及给了一个可以被外部访问到的IP静态IP地址。

这种做法，需要把外部的`物理网卡`设置为混杂模式([Promiscuous Model](https://cloud.tencent.com/developer/article/1439013))，也就是网卡接受所有流过网卡的帧(数据包)，包括那些不是发给本机的包，不验证MAC地址。这样这个`eth1`网卡就会向外部通过[ARP地址解析协议](https://zh.wikipedia.org/wiki/%E5%9C%B0%E5%9D%80%E8%A7%A3%E6%9E%90%E5%8D%8F%E8%AE%AE)发送自己的MAC地址，然后外部的交换机就会把这个IP地址的包转到`物理网卡`上。因为网卡工作在混杂模式，因此`eth1`就能收到相关的数据。如果发现是自己的数据，那么就接受，这样Docker容器的网络就与外部相通。

其实不管是Docker的NAT模式，还是混杂模式都会存在性能问题。NAT很明显转发(NAT转换)就有开销，而混杂模式下，网卡收到的负载都会完全交给所有的虚拟网卡，所以想想哪怕一个网卡没有数据，也会被其他网卡的数据影响。

因此这两种方式都不算完美，真正解决这样网络问题的是[VLAN技术](https://zh.wikipedia.org/wiki/%E8%99%9A%E6%8B%9F%E5%B1%80%E5%9F%9F%E7%BD%91)。因此Google的开发者为linux内核实现了一个[IPVLAN驱动](https://lwn.net/Articles/620087/)，基本为Docker量身定制。



## Namespace文件

整理完了linux namespace的玩法，在看一下ns的文件。

我们再运行一遍PID Namepace篇章中的`pid.mnt`程序(mount proc)，然后不退出:

```shell
$ sudo ./pid.mnt 
[sudo] password for derios: 
Parent [ 4599] - start a container!
Container [    1] - inside the container!
```

打开另外一个shell看一下父子进程的PID:

```shell
derios@ubuntu:~$ pstree -p 4599
pid.mnt(4599)───bash(4600)
```

我们可以到proc下（/proc//ns）查看进程的各个namespace的id（内核版本需要3.8以上），下面是父进程的：

```shell
derios@ubuntu:~$ sudo ls -l /proc/4599/ns
total 0
lrwxrwxrwx 1 root root 0  4月  7 22:01 ipc -> ipc:[4026531839]
lrwxrwxrwx 1 root root 0  4月  7 22:01 mnt -> mnt:[4026531840]
lrwxrwxrwx 1 root root 0  4月  7 22:01 net -> net:[4026531956]
lrwxrwxrwx 1 root root 0  4月  7 22:01 pid -> pid:[4026531836]
lrwxrwxrwx 1 root root 0  4月  7 22:01 user -> user:[4026531837]
lrwxrwxrwx 1 root root 0  4月  7 22:01 uts -> uts:[4026531838]
```

下面是子进程的:

```shell
derios@ubuntu:~$ sudo ls -l /proc/4600/ns
total 0
lrwxrwxrwx 1 root root 0  4月  7 22:01 ipc -> ipc:[4026531839]
lrwxrwxrwx 1 root root 0  4月  7 22:01 mnt -> mnt:[4026532520]
lrwxrwxrwx 1 root root 0  4月  7 22:01 net -> net:[4026531956]
lrwxrwxrwx 1 root root 0  4月  7 22:01 pid -> pid:[4026532522]
lrwxrwxrwx 1 root root 0  4月  7 22:01 user -> user:[4026531837]
lrwxrwxrwx 1 root root 0  4月  7 22:01 uts -> uts:[4026532521]
```

仔细看一下区别，发现ipc, net, user为相同ID，而mnt, pid, uts都不同。如果两个进程指向的namespace编号相同，则说明它俩在同一个namespace下，否则就不在。（如果读者你想验证，`docker exec -it <container name> bash`到一个容器里，top找两个进程，然后cat一下proc中对应PID的ns即可）

这些文件还有另一个作用，那就是，一旦这些文件被打开，只要其fd被占用着，那么就算PID所属的所有进程都已经结束，创建的namespace也会一直存在。比如：我们可以通过：mount –bind /proc/4600/ns/uts ~/uts 来hold这个namespace。

我们在最开始点了一下`setns`系统调用，函数声明如下:

```c++
int setns(int fd, int nstype);
```

其中第一个参数就是一个fd，也就是一个open()系统调用打开了上述文件后返回的fd，比如：

```c++
fd = open("/proc/4600/ns/nts", O_RDONLY);  // 获取namespace文件描述符
setns(fd, 0); // 加入新的namespace
```



