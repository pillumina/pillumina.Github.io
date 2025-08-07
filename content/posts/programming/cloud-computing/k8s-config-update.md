---
title: "Kubernetes ConfigMap 热更新"
date: 2021-04-24T11:22:18+08:00
hero: /images/posts/k8s-docker.jpg
menu:
  sidebar:
    name: Kubernetes ConfigMap Hot Update
    identifier: k8s-configmap-update
    parent: cloud-computing
    weight: 10
draft: false
---

  *注：如果对kubernetes的基本概念不太清楚，建议先过一下基本的资源类型再阅读此文*

  先随便给个例子:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: test-config
data:
  config.yml: |-
    start-message: 'Hello, World!'
    log-level: INFO
  bootstrap.yml:
    listen-address: '127.0.0.1:8080'
```

  我们定义了一个`ConfigMap`，data中定义了两个文件`config.yml`以及`bootstrap.yml`，当我们要引用当中的配置的时候，`kubernetes`提供了两种方案：

- 使用`configMapKeyRef`引用`configMap`中某个文件的内容作为Pod中容器的环境变量。
- 把所有`configMap`中的文件写到一个临时目录，将临时目录作为volume挂载到容器中，也就是`configmap`类型的`volume`。

  假设现在我们有一个`Deployment`，它的pod模板里引用了`configMap`，现在我们的目标是：**当`configmap`更新的时候，这个`Deployment`的业务逻辑也能随之更新**。那么有哪些方案？

- 最好的情况是，当`configMap`发生变更时，直接进行hot update，做到不影响pod的正常运行。
- 如果无法hot update或者这样完成不了需求，就要出发对应的`Deployment`做一次滚动更新。

  

## 场景一： 针对可以进行热更新的容器，进行配置热更新

  如果`configMap`由volume挂载，比如下述的投射卷，它的内容是可以更新的：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: volume-test
spec:
  containers:
  - name: container-test
    image: busybox
    volumeMounts:
    - name: all-in-one
      mountPath: "/projected-volume"
      readOnly: true
  volumes:
  - name: all-in-one
    projected:
      sources:
      - configMap:
          name: myconfigmap
          items:
            - key: config
              path: my-group/my-config
```

  为了能够比较好得理解，先说明一下`configMap`的volume挂载机制：

> 更新操作由kubelet的Pod Reconcile触发。每次Pod同步的时候（10s default），kubelet都会把Pod的所有`configMap`volume标记为`RequireRemount`，而kubelet中的volume循环控制会发现这些需要重新挂载的volume，去执行一次挂载操作。

  在`configMap`的重挂载过程中，kubelet会先比较远端的`configMap`和volume中的`configMap`是否一致，然后再做更新。需要注意的是，拿到的远端`configMap`操作可能有cache，不一定是最新版本。

  所以这样的更新方式的确可行，但是会有更新延时，最多的延时时间：

**Pod同步间隔(默认10s) + ConfigMap本地的缓存TTL**

> kubelet 上 ConfigMap 的获取是否带缓存由配置中的`ConfigMapAndSecretChangeDetectionStrategy` 决定。
>
> 注意，假如使用了 `subPath` 将 ConfigMap 中的某个文件单独挂载到其它目录下，那这个文件是无法热更新的（这是 ConfigMap 的挂载逻辑决定的）

  知道了原理，我们就明确一些概念：

1. 如果应用对`configMap`的更新有实时性要求，就需要在业务逻辑里自己到`ApiServer`去watch对应的`configMap`，或者干脆不用`configMap`而用`etcd`这样的一致性kv来存储管理配置。
2. 加入没有实时性要求，那么`configMap`本身的更新逻辑就可以做到。

  不过配置文件更新完了就不代表业务逻辑就更新了，我们还要解决如何通知应用重新读取配置，进行业务逻辑上的更新。例如对于`nginx`就需要一个`SIGHUP`信号量，这里再讨论几种做法。

### 热更新一： 应用本身监听本地配置文件

  这是最直接的方式，可以在应用里写监听的代码。一些配置相关的三方件本身就包装了这样的逻辑，比如[viper](https://github.com/spf13/viper)



### 热更新二：使用sidecar监听本地文件的变更

  `Prometheus`的`Helm Chart`中使用的就是这种方式，找到一个实用的镜像[configmap-reload](https://github.com/jimmidyson/configmap-reload)，它就会去watch本地文件的变更，并在发生变更时通过HTTP调用通知应用进行热更新。

  这种方式就有一个问题：sidecar发送信号的限制比较多，而很多开源组件比如`Fluentd`，`nginx`都是依赖`SIGHUP`信号进行热更新的。在`kubernetes` 1.10之前，并不支持pod中的容器共享同一个pid namespace，所以sidecar也就无法向业务容器发送信号。在1.10以后，虽然支持了pid共享，但是在共享之后pid namespace中的1号进程就会变成基础的`/pause`进程，我们便无法轻松定位到目标进程了。

  所以，只要k8s版本在1.10以后，并且开启了`ShareProcessNamespace`特性，多写点代码，比如通过进程名去找到pid，总是有办法的。但是1.10之前是没可能的。



### 热更新三：Fat Container

  胖容器比较`反模式`，不过可以解决sidecar的一些限制，把主进程和sidecar进程打进一个镜像里，这样就绕过了pid namespace隔离的问题。但是如果条件允许，还是用上述两个方案，因为复杂是脆弱的根源，容器本身是轻量的。



## 场景二： 无法热更新时，滚动更新Pod

无法热更新的场景举例有以下几个：

1. 应用本身没写热更新逻辑（大部分应用不会写）。
2. 使用`subPath`进程`configMap`的挂载，导致`configMap`无法自动更新。
3. 在环境变量或者`init-container`中依赖了`configMap`的内容。

  第三点，就是使用`configMapKeyRef`引用`configMap`中的信息作为环境变量时，这个操作也只会在pod创建时执行一次，所以是不会自动更新的。

  当无法进行热更新的时候，我们必须滚动去更新Pod了。一个简单的想法就是写个controller去watch`configMap`的变更，有变更就给`Deployment`资源做滚动更新。但是这样的实现是更复杂的，我们首先需要考虑有没有更简单的方式。



### 滚动更新一： 修改CI流程

  这个方式很简单，只需要写一个CI脚本，给`ConfigMap`计算一个hash，然后作为一个环境变量或者annotation加入到Deployment的Pod模板中。

  举个🌰:

```yaml
...
spec:
  template:
    metadata:
      annotations:
        com.cctoctofx.configmap/hash: ${CONFIGMAP_HASH}
...
```

  这样，如果`configMap`变化了，那么Deployment里的Pod模板自然会变化，k8s会自动帮我们做滚动更新。甚至如果`configMap`不复杂，直接转化为json放到pod模板里都行，而且还方便故障排查的时候快速知道内容是啥。



### 滚动更新二：Controller

  写个controller检测`configMap`变更并触发滚动更新，手动写之前还是看一看开源实现：

- [Reloader](https://github.com/stakater/Reloader)
- [ConfigmapController](https://github.com/fabric8io/configmapcontroller)
- [k8s-trigger-controller](https://github.com/mfojtik/k8s-trigger-controller)

  

### 滚动更新三：Liveness Probe / Readiness Probe

  这个手段需要深入一下，初步想法是用liveness调用一个脚本，脚本判断文件是否变动，如果变动，liveness得到false，重启pod，也可以同时设置readiness。



## 滚动更新需要考虑的问题

  举个例子，我们用场景二中提到的方式去更新：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  replicas: 3
  template:
    annotations:
      nginx-config-md5: d41d8cd98f00b204e9800998ecf8427e
    spec:
      containers:
      - name: nginx
        image: nginx
        volumeMounts:
        - name: nginx-config
          mountPath: /etc/config
      volumes:
      - name: config-volume
        configMap: 
          name: nginx-config
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-config
data:
  nginx.conf: |-
    ## some configurations...
```

>- 每次部署的时候，计算configMap的MD5,填入pod的template中.
>- 加入configMap发生变化，摘要也会变化，会触发一次Deployment的滚动更新。

  这个流程看起来比较美，但思考一下如果我们更新了一个配置，但这个配置是有问题的，如果pod使用了错误的配置会无法工作（比如无法通过`readinessProb`检查）。最后，滚动更新的流程就会卡住，错误的配置不会把Deployment搞崩掉。

  这个逻辑看着也挺好，但是有个问题却忽视了，如果`nginx-config`更新成了错误的值，虽然**还没有重建的Pod**暂时是健康的，但是如果Pod挂掉发生重建，或者其中的容器重新读取了一次配置，那么这些Pod就会陷入异常。所以整个集群的状态是很不稳定的。

  因此问题的本质是：**在原地更新`configMap`或者`secret`的时候，我们并没有进行滚动发布，而是一次性把新的配置更新到整个集群的所有实例当中**。而我们所说的`滚动更新`就是控制各个实例读取新的配置的时机，可是由于我们无法把控Pod挂掉的时机，我们无法准确进行过程控制。

  ### 解决方案

  上述问题的问题在于**原地更新**，要解决这个问题，只需要在每次`ConfigMap`变化的时候，重新生成一个`ConfigMap`，再更新Deployment使用这个新的`ConfigMap`就行了。而重新生成`ConfigMap`最简单的方式就是在其命名中加上`ConfigMap`的data值计算出的摘要，比如：

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-config-d41d8cd98f00b204e9800998ecf8427e
data:
  nginx.conf: |-
    ## some configurations...
```

  `ConfigMap`的Rollout在社区中也是历经很久还没有解决([#22368](https://zhuanlan.zhihu.com/p/66051135/22368))，目前为止，解决这个问题的方向也是`immutable configmap`模式。

  但是这种方案会有几个问题：

> - 如何做到每次配置文件更新时，都创建一个新的ConfigMap？
> - 目前社区的态度是把这一步放到Client解决，比如helm和kustomize。
> - 历史configMap不断积累，能怎么回收？
> - 针对这点，社区希望在服务端实现一个GC机制来清理没有任何资源引用的configMap。

  把更新逻辑放在client端虽然会有重复造轮子的问题，但是至少目前为止，configMap的新建和Deployment等对象的更新是最成熟的configMap滚动更新方案。

  ### Kustomize的实践方式

  Kustomize对这个方案有内置的支持，只需要使用`configGenerator`：

```yaml
configMapGenerator:
- name: my-configmap
  files:
  - common.properties
```

  这段yaml就能在kustomize中生成一个configMap对象，这个对象的data来自于`common.properties`文件，而且name中会加上这个文件的SHA值作为后缀。

  在kustomize的其他layer中，只要以`my-configmap`作为name引用这个configMap即可，当最终渲染的时候，kustomize会自动进行替换操作。

 

### Helm的实践方式

...



## 附录

[facilitate ConfigMap rollouts/management discussion](https://github.com/kubernetes/kubernetes/issues/22368#issuecomment-421141188)

