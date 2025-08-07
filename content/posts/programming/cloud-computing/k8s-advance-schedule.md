---
title: "Kubernetes Handbook (Schedule)"
date: 2021-04-09T10:22:18+08:00
hero: /images/posts/k8s-docker.jpg
menu:
  sidebar:
    name: Kubernetes Handbook (Schedule)
    identifier: k8s-schedule
    parent: cloud-computing
    weight: 10
draft: false
---

## 内容涵盖

- 使用节点污点和pod容忍度阻止pod调度到特定节点
- 将节点亲缘性规则作为节点选择器的一种替代
- 使用节点亲缘性进行多个pod的共同调度
- 使用节点非亲缘性来分离多个pod

## 高级调度

  在pod介绍的文章中可以看到，k8s可以通过在pod spec里面指定节点选择器，而这篇文章介绍的是后面其他逐渐加入的机制。



### 使用污点和容忍度阻止节点调度到特定节点

  新特性： `节点污点`、`pod对于污点的容忍度`

  这些特性用于限制哪些pod可以被调度到某一个节点，也就是说只有当一个pod容忍某个节点的污点，这个pod才能被调度到该节点。

  节点选择器和节点亲缘性规则，是`明确`在pod中添加的信息，来觉得一个pod可以或者不可以被调度到某个节点。而污点不一样，是在不修改已有pod信息的前提下，通过在节点上新增污点信息，来拒绝pod在这个节点的部署。

 

#### 简单介绍污点和容忍度

  我在自己的机器用minikube创建了k8s单点集群，用`kubectl describe node minikube`可以看到:

```shell
k describe node minikube
Name:               minikube
Roles:              master
Labels:             beta.kubernetes.io/arch=amd64
                    beta.kubernetes.io/os=linux
                    deploy=test
                    kubernetes.io/arch=amd64
                    kubernetes.io/hostname=minikube
                    kubernetes.io/os=linux
                    minikube.k8s.io/commit=b09ee50ec047410326a85435f4d99026f9c4f5c4
                    minikube.k8s.io/name=minikube
                    minikube.k8s.io/updated_at=2021_03_30T20_15_58_0700
                    minikube.k8s.io/version=v1.14.0
                    node-role.kubernetes.io/master=
Annotations:        kubeadm.alpha.kubernetes.io/cri-socket: /var/run/dockershim.sock
                    node.alpha.kubernetes.io/ttl: 0
                    volumes.kubernetes.io/controller-managed-attach-detach: true
CreationTimestamp:  Tue, 30 Mar 2021 20:15:55 +0800
Taints:             <none>                  # -----> 主节点暂时没有污点
Unschedulable:      false
Lease:
  HolderIdentity:  minikube
  AcquireTime:     <unset>
  RenewTime:       Fri, 09 Apr 2021 14:48:12 +0800
```

  可以看到`Taints`属性，表示目前这个主节点没有污点。不过这里可以举个例子：

```
Taints:  node-role.kubernetes.io/master:NoSchedule
```

  污点包含了一个key, value以及一个effect--> `<key>=<value>:<effect>`。上面这个例子里，key是node-role.kubernetes.io/master，空的value，effect是NoSchedule。

  这个污点能阻止pod调度到这个节点上，除非有pod能够容忍这个污点，而通过容忍这个污点的pod都是**系统级别的pod**。

```
Toleration: node-role.kubernetes.io/master:NoSchedule
```

  如果pod包含容忍度能匹配节点的污点，那么就可能被调度到这个节点上。

##### 显示pod的污点容忍度

```shell
huangyuxiao@CctoctoFX  /usr/local/var/log $ k describe po nginx-r4hdr
Name:         nginx-r4hdr
... # 省略中间其他属性
Tolerations:     node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                 node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
...
```

  可以看到，我describe了一个在集群中的pod。

##### 污点的效果

  在上述的例子中，可以看到对于这个nginx的pod，其定义了当节点状态是`not-ready`或者`unreachable`的时候，这个pod允许运行在这个节点300秒。这两个容忍度使用的是`NoExecute`而不是`NoSchedule`。

  每个污点可以关联一个效果，包含如下三种:

1. `NoSchedule`：如果pod没有容忍这些污点，pod则不能被调度到包含这些污点的节点上
2. `PreferNoSchedule`： 一个比较loose的NoSchedule，表示尽量阻止调度到这里。但是如果实在没其他地方能调度了，还是可以调度到这边的。
3. `NoExecute`：这个和上述两者不同，前两种只是影响调度。而NoExecute也会影响在节点上运行着的pod。如果在某个节点上添加了NoExecute，如果节点上运行着的pod没有容忍这个污点，就会被从这个节点删除。

#### 在节点上添加Custom污点

  一个很简单的诉求：一个k8s集群上面同时有生产环境和非生产环境的流量。最重要的一点是，非生产环境的pod不能运行在生产环境的节点上，就可以在生产环境的节点上添加污点来满足要求:

```shell
$ kubectl taint node node.k8s node-type=production:NoSchedule
```

  这里新增了一个污点，key是node-type，value是production，效果是NoSchedule。所以这个时候你再去部署常规的pod，是不会部署到添加了这些污点的节点上去的。



#### 往pod新增污点容忍度

  还是上面的诉求，现在为了把生产环境的pod部署到生产环境节点上，pod需要容忍刚才我们添加的污点，那么我们修改下pod的资源yaml：

```yaml
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: pod
spec:
  replicas: 5
  template:
    spec:
      ...
      tolerations:
      - key: node-type
        operator: Equal
        value: production
        effect: NoSchedule
```

  新增tolerations描述即可 [官方文档](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/)

  如果又不想把这个pod调度到非生产环境，则需要类似在非生产环境打上污点



#### 污点和容忍度的使用场景

  污点可以只有一个key和一个effect，而不必有value。容忍度可以通过设置`Equal` operator来指定匹配的value（default）。或者可以设置`Exists` operator来匹配污点的key。

##### 调度的时候使用污点和容忍度

  就如从最开始介绍的这样，用`NoSchedule`或者定义非优先调度节点`PreferNoSchedule`，或者把已有的pod从当前节点删除。

  比如可以把一个集群分为几个部分，部分节点可能提供了特殊影响比如GPU，TPU之类的，而且只有部分pod需要使用到这些硬件的时候，也可以通过污点和容忍度实现。

##### 节点unreachable之后pod重新调度的等待时长设置

  和前面的例子一样:

```shell
huangyuxiao@CctoctoFX  /usr/local/var/log $ k describe po nginx-r4hdr
Name:         nginx-r4hdr
... # 省略中间其他属性
Tolerations:     node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                 node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
...
```

```yaml
  tolerations:
  - effect: NoExecute
    key: node.kubernetes.io/not-ready
    operator: Exists
    tolerationSeconds: 300
  - effect: NoExecute
    key: node.kubernetes.io/unreachable
    operator: Exists
    tolerationSeconds: 300
```

  当k8s的controller检测到有节点处于not-ready或者unreachable状态的时候，会等待300秒，如果状态持续，才把pod调度到其他节点上。这两个容忍度是你没有配的时候自动加给pod的，如果觉得300太长了也可以显式得去改变。



### 使用节点亲缘性将pod调度到特定节点

 `节点亲缘性(node affinity)：允许pod尽量调度到某些节点子集。`

  早期的k8s中，初始的节点亲缘性机制就是pod描述中的`nodeSelector`字段。节点必须包含所有pod对应字段中的指定label，才能成为调度的目标节点。

  节点选择器很简单，但是不能满足所有需求，所以更强大的亲缘性机制才会被引入。

  和节点选择器类似，每个pod可以定义自己的节点亲缘性规则，这些规则可以允许你指定硬件限制或者偏好。当你指定一种偏好后，k8s会把pod尽量调度到这些节点上面，如果没法实现，则调度到其他节点。

  如果使用谷歌的k8s引擎(GKE)，可以`kubectl describe node xxxx`查到节点Labels，这里面包含了默认的和亲缘性有关的标签。



#### 指定强制性节点亲缘性规则

  在介绍Pod的文章中，我们利用节点选择器将那些需要GPU的pod只被调度到有GPU的节点上:

```yaml
apiVesion: v1
kind: Pod
metadata:
  name: kubia-gpu
spec:
  nodeSelector:
    gpu: "true"
...
```

  这样这个pod会被调度到包含`gpu=true`标签的节点。如果我们用节点亲缘性规则去替换：

```yaml
apiVesion: v1
kind: Pod
metadata:
  name: kubia-gpu
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoreDuringExecption:
        nodeSelectorTerms:
        - matchExpressions:
          - key: gpu
            operator: In
            values:
            - "true"
```

  这种写法貌似比上面的复杂了很多，但是表达能力更强了。

##### 较长的节点亲缘性属性名的意义

  上面的写法里，spec里有affinity，affinity里有nodeAffinity，下面还有个非常长的名字，我们拆解一下：

- `requiredDuringScheduling...`：说明该字段下定义的规则，为了让pod能调度到该节点上，明确指出了这个节点必须含有的标签。
- `...IgnoreDuringException`: 表明该字段下定义的规则，不会影响已经在节点上运行的pod。

  所以这个含义就是：***当前的亲缘性规则只会影响正在被调度的pod, 而不会导致正在运行的pod被删除***。所以一般来说目前的规则都是以`IgnoredDuringException`作为结尾。

*注：`RequiredDuringException`就表示如果去掉某节点上的标签，那么含有这些标签的pod会被删除，这个特性目前的k8s应该还没有。*

 

##### 了解节点选择器的条件

  ```yaml
        nodeSelectorTerms:
        - matchExpressions:
          - key: gpu
            operator: In
            values:
            - "true"
  ```

  现在这几个字段应该比较好了解。也就是表示这个pod只会被调度到gpu=true的节点上。

  更有趣的是，节点亲缘性可以在调度的时候指定节点的优先级。



#### 调度pod时的节点优先级

  `preferredDuringSchedulingIgnoredDuringException` 来实现优先考虑哪些节点。

  思考一个场景：你有一个跨越多个国家的多个数据中心，每个数据中心代表了一个单独的可用性区域。在每个区域中，你有一些特定的机器，只提供给你自己或者合作的公司使用。现在你想部署一些pod，希望吧pod优先部署在区域zone1，并且是为你公司部署预留的机器上。如果你的机器没有足够的空间给这些pod使用，或者处于其他的一些原因不希望这些pod被调度到这些机器上，那么就要调度到其他区域的其他机器上面，这种情况你也是可以接受的。那么节点亲缘性就可以实现这样的功能。

##### 给节点加标签

  每个节点需要包含两个标签:

1. `表示所在的这个节点所归属的可用性区域`
2. `表示这是一个独占的节点还是共享的节点`

  ```shell
$ kubectl label node node1.k8s availability-zone=zone1
$ kubectl label node node1.k8s share-type=dedicated
$ kubectl label node node2.k8s availability-zone=zone2
$ kubectl label node node2.k8s share-type=shared
  ```



##### 指定优先级节点亲缘性规则

  把节点的标签打好以后，创建一个Deployment，其中优先选择zone1中的的dedicated节点，下面是描述:

```yaml
apiVesion: extension/v1beta1
kind: Deployment
metadata:
  name: pref
spec:
  template:
    ...
    spec:
      affinity:
        nodeAffinity:
        preferredDuringSchedualingIgnoredDuringException:
        - weight: 80
          preference:
            matchExpressions:
            - key: availability-zone
              operator: In
              values:
              - zone1
        - wight: 20
          preference:
            matchExpressions:
            - key: share-type
              operator: In
              values:
              - dedicated
```

  可见，节点优先调度到zone1，这是最重要的偏好; 同时优先调度pod到独占(dedicated)节点，但是这个优先级是zone优先级的1/4。

 ##### 节点优先级是如何工作的

  `核心是把节点根据标签分组，然后排序。`

  比如把包含`availability-zone`以及`share-type`标签，并且匹配pod亲缘性的节点(zone1, dedicated)排在前面。然后，根据pod设置的亲缘性规则的权重，接下来是zone1和shared节点，然后是其他区域的dedicated节点，优先级最低的是其他的节点。



### 使用pod亲缘性和非亲缘性对pod进行协同部署

  上面我们了解了pod和节点间的亲缘性规则能够影响pod能够调度到哪个节点。我们再来研究研究如何制定pod自身之间的亲缘性。

  想象一下：如果你有一个前端的pod和一个后端pod，把这些节点部署得比较靠近，可以降低延时，提高应用的性能。可以使用节点亲缘性规则来确保这两个pod被调度到同一个节点、同一个rack、同一个数据中心。但是这样后续又要指定调度到确切的位置，明显是违背k8s设计哲学的。所以更好的做法应该是定义pod之间的亲缘性规则，让k8s去把pod部署在它觉得合适的地方，同时确保2个pod是靠近的。



#### 使用pod间亲缘关系将多个pod部署在同一个节点

  ...  



