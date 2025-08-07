---
title: "Kubernetes Handbook (Start & Pod)"
date: 2021-03-31T11:22:18+08:00
hero: /images/posts/k8s-docker.jpg
menu:
  sidebar:
    name: Kubernetes Handbook (Start & Pod)
    identifier: k8s-basic
    parent: cloud-computing
    weight: 10
draft: false
---

## 使用minikube构建本地单节点k8s集群
- minikube ssh
- kubectl cluster-info
- kubectl get nodes #查看节点信息
- kubectl describe node minikube #详细信息

## 多节点k8s集群，使用Google K8s Engine

构建方式看GKE官网即可


## k8s初步使用
kubectl run kubia --image=derios/kubia --port=8080 --generator=run/v1
- `--image=derios/kubia`代表要运行的容器镜像
- 这里的`--generator`会被废弃，其含义指代的是创建一个`ReplicationController`而不是`Deployment`。
- kubectl apply -f <yaml name> 更常用
- kubectl get pods
- kubectl get pods -o wide 显示pod ip和pod的节点
- 如果使用GWE，可以访问集群的dashborad:
- kubectl clusert-info获取地址
- gcloud container clusters describe kubia | grep -E "(username|password):"获取用户名和密码
如果仅仅使用minikube，则如下不需要任何凭证即可访问:
```shell
minikube dashboard
```

### Namespace相关操作

```yaml
kubectl config set-context --current --namespace=my-namespace
```



### 创建服务对象，访问Web应用

`如果使用minikube或者kubeadm等自定义k8s，loadbalancer是没有集成的，需要AWS或者Google Cloud。最好使用NodePort或者Ingress Controller。如果真要用minikube, 可以使用minikube tunnel解决, 或者minikube service kubia-http`

- kubectl expose rc/po/svc kubia --type=LoadBalancer --name kubia-http创建出的service介于pod和node之间, kubectl get services可以查看

- 如果没有外部IP地址，因为k8s运行的云基础设施新建lb需要一段时间，lb启动以后应该会显示

  

### ReplicationController的角色
- 复制pod，replica扩缩
- `kubectl scale rc kubia --replicas=3`  可以将rc下的pods方便进行扩缩容

### 为何需要service
解决不断变化的pod IP问题，以及在一个固定的IP和端口上对外暴露多个pod。

当一个service被创建，会得到一个静态IP，在service的生命周期这个IP不会被改变，service会确保其中提个pod接受连接，而非关注这个pod运行在何处.


## k8s的pod
pod不一定要包含多个容器，单独容器的pod也是很常见的。如果pod包含多个容器，那么这些容器都工作在同一个工作节点上，pod是不能跨多个工作节点工作的。

pod概念的提出，是因为我们不能把多个进程聚集在一个单独的容器中，我们需要另一种更为高级的结构将容器绑定在一起，并把他们欧威一个单元进行管理。

一个Pod下的所有容器都在相同的额network和UTS命名空间下运行，所以共享相同的主机名和网络接口，也能够通过IPC进行通信，甚至在最新的k8s和docker版本下，能共享相同的PID命名空间。

### 容器如何共享相同IP和端口空间
由于pod中的容器运行于相同Namespace命名空间，所以他们共享相同的IP地址和端口空间。这意味着在同一个pod中的容器运行的多个线程不能绑定到相同的端口号，否则会冲突。此外，他们还有相同的loopback网络几口，所以容器可以通过localhost和同一个pod中的其他容器进行通信。

### 平坦网络
k8s集群中所有pod都在同一个共享网络地址空间中，也就是说每个pod都可以通过其他pod的IP地址来实现相互访问，表示他们之间没有NAT(网络地址转换)网管。当两个pod彼此之间发送网络数据包时，都把对方的实际IP地址看作数据包中的源IP。

因此可以看到，pod就是逻辑主机，行为和非容器世界中的物理主机或者虚拟机相似。运行在同一个pod中的进程和运行在同一物理机或者虚拟机上的进程类似，只是每个进程封装在一个容器里。

## 通过pod合理管理容器
`一个由前端应用服务器和后端数据库组成的多层应用程序，应该讲其配置为单个pod还是两个pod呢？`

### 把多层应用分散到多个pod中
一个pod下的所有容器运行在一起，而web服务器和数据库真的要在同一台计算机上运行吗？明显是否定的。而且如何k8s集群节点多了，如果只有一个单独的pod，其资源利用率也很低。

### 基于扩缩容考虑而分隔到多个pod中
pod是扩缩容的基本单位，k8s不能横向扩缩单个容器，而只能扩缩整个pod，所以明显web服务器和数据库放在一起是不对的。
而且数据库这种有状态的服务器，比无状态的web服务器更加难扩展，所以如果要单独扩缩容器，这个容器必须明确地部署在单独的pod当中

### 在pod中使用多个容器的时机
一般来说，常见场景是应用由一个主进程和多个辅助进程组成，也就是这个容器组是紧密耦合的。举例就是，pod中的主容器是仅仅服务于每个目录中文件的Web服务器，另一个sidecar容器定期从外部资源下载内容并将其存储在Web服务器目录里。这种情形需要使用k8s的Volume，把其挂在到两个容器里。
sidecar容器包括：日志轮转器、收集器、数据处理器、通信适配器等等

因此容器如何分组到pod这个问题，我们需要问以下的问题：
1. 它们是否需要一起运行还是可以在不同的主机上运行？
2. 它们代表的是一个整体还是相互独立的组件？
3. 它们必须一起扩缩容还是可以分别单独进行？

## 以`YAML`或者`JSON`创建pod
好处：`kubectl run`局限性很多，比如属性配置很有限，而资源配置的方式除了更方便定义k8s资源对象，还能把它们存储在版本控制系统中。
注意参考: [k8s api reference](https://kubernetes.io/docs/reference/)

### 检查现有pod的`YAML`描述
`kubectl get po kubia-zxzij -o yaml`
```yaml
apiVersion: v1
kind: Pod
metadata:
  creationTimestamp: "2021-03-31T03:00:40Z"
  generateName: kubia-
  labels:
    app: kubia
  managedFields:
  - apiVersion: v1
    fieldsType: FieldsV1
    fieldsV1:
      f:metadata:
        f:generateName: {}
        f:labels:
          .: {}
          f:app: {}
        f:ownerReferences:
          .: {}
          k:{"uid":"4b78e019-3753-40a1-998c-0d0ec4af9f29"}:
            .: {}
            f:apiVersion: {}
            f:blockOwnerDeletion: {}
            f:controller: {}
            f:kind: {}
            f:name: {}
            f:uid: {}
      f:spec:
        f:containers:
          k:{"name":"kubia"}:
            .: {}
            f:image: {}
            f:imagePullPolicy: {}
            f:name: {}
            f:ports:
              .: {}
              k:{"containerPort":90,"protocol":"TCP"}:
                .: {}
                f:containerPort: {}
                f:protocol: {}
            f:resources: {}
            f:terminationMessagePath: {}
            f:terminationMessagePolicy: {}
        f:dnsPolicy: {}
        f:enableServiceLinks: {}
        f:restartPolicy: {}
        f:schedulerName: {}
        f:securityContext: {}
        f:terminationGracePeriodSeconds: {}
    manager: kube-controller-manager
    operation: Update
    time: "2021-03-31T03:00:40Z"
  - apiVersion: v1
    fieldsType: FieldsV1
    fieldsV1:
      f:status:
        f:conditions:
          k:{"type":"ContainersReady"}:
            .: {}
            f:lastProbeTime: {}
            f:lastTransitionTime: {}
            f:status: {}
            f:type: {}
          k:{"type":"Initialized"}:
            .: {}
            f:lastProbeTime: {}
            f:lastTransitionTime: {}
            f:status: {}
            f:type: {}
          k:{"type":"Ready"}:
            .: {}
            f:lastProbeTime: {}
            f:lastTransitionTime: {}
            f:status: {}
            f:type: {}
        f:containerStatuses: {}
        f:hostIP: {}
        f:phase: {}
        f:podIP: {}
        f:podIPs:
          .: {}
          k:{"ip":"172.17.0.7"}:
            .: {}
            f:ip: {}
        f:startTime: {}
    manager: kubelet
    operation: Update
    time: "2021-03-31T03:00:47Z"
  name: kubia-2r2pb
  namespace: default
  ownerReferences:
  - apiVersion: v1
    blockOwnerDeletion: true
    controller: true
    kind: ReplicationController
    name: kubia
    uid: 4b78e019-3753-40a1-998c-0d0ec4af9f29
  resourceVersion: "11025"
  selfLink: /api/v1/namespaces/default/pods/kubia-2r2pb
  uid: 342307fe-f6d8-4201-ba02-0210cd75d0a8
spec:
  containers:
  - image: derios/kubia
    imagePullPolicy: Always
    name: kubia
    ports:
    - containerPort: 90
      protocol: TCP
    resources: {}
    terminationMessagePath: /dev/termination-log
    terminationMessagePolicy: File
    volumeMounts:
    - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
      name: default-token-whxzx
      readOnly: true
  dnsPolicy: ClusterFirst
  enableServiceLinks: true
  nodeName: minikube
  preemptionPolicy: PreemptLowerPriority
  priority: 0
  restartPolicy: Always
  schedulerName: default-scheduler
  securityContext: {}
  serviceAccount: default
  serviceAccountName: default
  terminationGracePeriodSeconds: 30
  tolerations:
  - effect: NoExecute
    key: node.kubernetes.io/not-ready
    operator: Exists
    tolerationSeconds: 300
  - effect: NoExecute
    key: node.kubernetes.io/unreachable
    operator: Exists
    tolerationSeconds: 300
  volumes:
  - name: default-token-whxzx
    secret:
      defaultMode: 420
      secretName: default-token-whxzx
status:
  conditions:
  - lastProbeTime: null
    lastTransitionTime: "2021-03-31T03:00:40Z"
    status: "True"
    type: Initialized
  - lastProbeTime: null
    lastTransitionTime: "2021-03-31T03:00:47Z"
    status: "True"
    type: Ready
  - lastProbeTime: null
    lastTransitionTime: "2021-03-31T03:00:47Z"
    status: "True"
    type: ContainersReady
  - lastProbeTime: null
    lastTransitionTime: "2021-03-31T03:00:40Z"
    status: "True"
    type: PodScheduled
  containerStatuses:
  - containerID: docker://1cce1d07039838f1f4d825d088fca52b54b15808efebf20d1f38cc2952a65ebd
    image: derios/kubia:latest
    imageID: docker-pullable://derios/kubia@sha256:7aa603e50514206f8d40126520409a6814001c1bae68b9f286cca9c1ca271f7a
    lastState: {}
    name: kubia
    ready: true
    restartCount: 0
    started: true
    state:
      running:
        startedAt: "2021-03-31T03:00:46Z"
  hostIP: 192.168.49.2
  phase: Running
  podIP: 172.17.0.7
  podIPs:
  - ip: 172.17.0.7
  qosClass: BestEffort
  startTime: "2021-03-31T03:00:40Z"
```

虽然看着比较复杂，主要包含了`k8s对象/资源类型`、`k8s API版本`、`pod元数据`、`pod规格和内容`、`pod和内部容器的详细状态`等。



### pod定义的主要部分

1. `metadata`: 名称、命名空间、标签和关于该容器的其他信息
2. `spec`：包含pod内容的实际说明，例如pod的容器、卷和其他数据
3. `status`: 包含运行中的pod的当前信息，比如pod所处的条件、每个容器的描述和状态，以及内部IP和其他基本信息

创建新的pod的时候，不需要提供`status`部分，这只是运行时数据。实际上深究每个属性的意义不是很大，我们需要关注创建pod的最基本信息。



### 创建一个简单的`YAML`描述

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: kubia-manual
spec:
  containers:
  - image: derios/kubia
    name: kubia
    ports:
    - containerPort: 8080
      protocol: TCP
```

可以用`kubectl explain pods`、`kubectl explain pod.spec`之类去看资源定义



### 用`kubectl create`来创建pod

```shell
$ kubectl create -f kubia-manual.yaml
```

```shell
$ kubectl get pods
```



## 查看应用程序日志

小型node.js应用把日志记录到进程的标准输出，容器化的应用程序通常把日志记录到标准输出和标准错误流，而不是写入文件。所以这就允许user通过标准的方式查看不同应用程序的日志。

Docker(或者广义的container runtime)把这些流重定向到文件，可以用以下命令获取容器的日志：

```shell
$ docker logs <container id>
```

一般可以ssh到pod运行节点然后这样查看，不过k8s提供了更为简单的方式。

**使用`kubectl logs命令获取pod日志`**：

```shell
$ kubectl logs kubia-manual
```

如果一个pod只包含一个容器，那么查看k8s应用程序的日志会变得很简单。



**获取多容器pod的日志时指定容器名称**

```shell
$ kubectl logs kubia-manual -c kubia
```

请注意，如果pod被删了，那么pod里的日志也会被删。如果希望在pod删除之后仍然可以获取其日志，需要设置中心化、集群范围的日志系统，把所有日志存储到中心存储中，这个在后续会讨论。



## 向pod发送请求

通过`端口转发`连接到pod以进行测试和调试。

### 将本地网络端口转发到pod中的端口

如果不想通过service情形下对某个特定的pod进行通信(debug)，可以通过`kubectl port-forward`命令完成端口转发:

```shell
$kubectl port-forward kubia-munual 8888:8080
```

这样就能通过本地端口连接到我们的pod。

### 通过端口转发连接到pod

另一个终端`curl localhost:8888`即可调试。实际上述指令起了一个线程进行对调试请求的处理。



## 使用Tag组织pod

对于微服务架构，部署数量能很轻松膨胀：多副本、不同版本等等。所以需要进行Pod分组去管理对象。Tag标签不仅仅可以组织pod，还包括了其他k8s资源。

标签的定义：可以附加到资源的任意键值对。只要标签的key在资源内唯一，那么一个资源便可以拥有多个标签，一般我们在创建资源的时候回把标签附加到资源上，当然后续可以新增、修改标签。举个例子：

- `app`: 指定pod属于哪个应用、组件或者微服务
- `rel`：显示在pod中运行的应用程序版本是stable、beta还是canary

利用这种组织形式，访问集群的开发或者运维能通过查看pod标签轻松看到系统结构以及每个pod的角色。



### 创建pod的时候指定标签

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: kubia-manual-v2
  labels:
    creation_method: manual
    env: prod
spec:
  containers:
  - image: derios/kubia
    name: kubia
    ports:
    - containerPort: 91
      protocol: TCP
```

可以通过以下查看标签

```shell
$ kubectl get po --show-labels
```

如果只对某些标签的pod感兴趣，可以使用`-L`选项把tag放在列上：

```shell
$ kubectl get po -L creation_method,env
```

```shell
NAME              READY   STATUS    RESTARTS   AGE     CREATION_METHOD   ENV
kubia-2r2pb       1/1     Running   0          5h7m
kubia-kt6gz       1/1     Running   0          5h7m
kubia-manual-v2   1/1     Running   0          2m24s   manual            prod
kubia-tqt52       1/1     Running   0          5h7m
nginx-r4hdr       1/1     Running   0          18h
nginx-rg76b       1/1     Running   0          18h
```



### 修改现有pod标签

新增的话，用`kubectl label`即可:

```shell
kubectl label po kubia-manual creation_method=manual
```

更改的话，需要使用`--overwrite`选项:

```shel
kubectl label po kubia-manual-v2 env=debug --overwrite
```



## 通过`标签选择器`列出pod子集

标签的强大功能体现在这里。

标签选择器根据资源的以下条件来筛选资源:

1. 包含(or not)使用特定key的标签
2. 包含具有特定key和value的标签
3. 包含具有特定key的标签，但值和我们指定的不同

### 使用`标签选择器`列出pod

```shell
$ kubectl get po -l creation_method=manual
NAME              READY   STATUS    RESTARTS   AGE
kubia-manual-v2   1/1     Running   0          9m5s
```

列出包含`env`标签的所有pod，无论其值多少：

```shell
$ kubectl get po -l env
NAME              READY   STATUS    RESTARTS   AGE
kubia-manual-v2   1/1     Running   0          10m
```

列出没有`env`标签的pod：

```shell
$ kubectl get po -l '!env'
NAME          READY   STATUS    RESTARTS   AGE
kubia-2r2pb   1/1     Running   0          5h17m
kubia-kt6gz   1/1     Running   0          5h17m
kubia-tqt52   1/1     Running   0          5h17m
nginx-r4hdr   1/1     Running   0          18h
nginx-rg76b   1/1     Running   0          18h
```

其他的选择器语法解释：

- `creation_method!=manual`：选择带有`creation_method`标签，但其值不为manual的pod
- `env in (prod, devel)`: 选择带有`env`标签且值为`prod`或者`devel`的pod
- `env notin (prod, devel)`: 差不多自己理解

### 在`标签选择器`中使用多个条件

 精确匹配：`app=pc,rel=beta`

这种列出pod子集的做法能够让我们能够进行子集的操作，简单的例如删除自己的pod



## 使用`标签`和`选择器`约束pod调度

pod的随机调度是k8s集群的工作方式，但如果我们希望对将pod调度到何处有一定的发言权，比如我们的硬件是`异构`或者`非同质`的：cpu架构区别、节点磁盘种类区别、GPU密集型运算加速特定节点等。

当然，我们不是说明pod应该调度到哪个节点，因为基础架构和应用程序的强耦合不是k8s的玩法。我们通过用`某种方式`描述对节点的需求，使得k8s选择一个符合这个需求的节点，也就是用`节点标签`和`节点标签选择器`完成。



### 用标签分类工作节点

假设集群中的一个节点刚添加完成，其包含一个用于通用GPU计算的GPU，我们希望加标签去展现这个节点特性:

```shell
$ kubectl label node <node name> gpu=true
```

```shell
$ kubectl get nodes -l gpu=true
NAME       STATUS   ROLES    AGE   VERSION
minikube   Ready    master   20h   v1.19.2
```



### 将pod调度到特定节点

假设已经给节点打上了标签，为了让调度器只在提供了确定标签的节点中进行选择，需要在pod的YAML文件中添加一个节点选择器(这里是deploy:test标签的节点):

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: kubia-manual-v2
  labels:
    creation_method: manual
    env: prod
spec:
  nodeSelector:
    deploy: "test"
  containers:
  - image: derios/kubia
    name: kubia
    ports:
    - containerPort: 91
      protocol: TCP
```



### 调度到特定一个节点(不推荐)

每个node有自己的一个唯一标签: `kubernetes.io/hostname`，但是如果节点离线了，就会导致pod无法调度，因此用标签选择器比较符合工作流程。当讨论Replication-Controllers和Service的时候，标签选择器的重要性也会彰显。



## 注解pod

这块暂时不想深究，知道用法就行



## 使用命名空间对资源进行分组

待研究



## 停止和移除pod

### 按名称删除

```shell
& kubectl delete po kubia
```

注意删除pod的时候，是k8s向进程发送`SIGTERM`信号并等待一定的秒数(30s默认)，如果还没有正常关闭，则发送`SIGKILL`终止线程。因此，为了保证线程可以正常关闭，需要正确处理`SIGTERM`信号。

### 使用`标签选择器`删除

```shell
$ kubectl delete po -l creation_method=manual
```

即删除带有此标签的所有pod。

### 通过删除整个命名空间来删除pod

```shell
$ kubectl delete ns custom-namespace
```



### 删除命名空间的所有pod，但保留命名空间

```shell
$ kubectl delete po --all
```

删除当前命名空间的所有pod。

注意如果空间内创建了replication controllers，删除所有还会新增新的pod出来，所以还需要删除rc。



### 删除命名空间中(几乎)所有资源

```shell
kubectl delete all --all
```

使用all并不意味着会删除所有的内容，比如secret还是会被保留，这些是要明确指定删除的。

注意：该命令会删除名为`kubernetes`的service，不过在几分钟之内会被重新创建。



