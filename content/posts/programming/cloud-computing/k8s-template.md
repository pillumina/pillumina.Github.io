---
title: "Kubernetes Developement"
date: 2021-04-21T11:22:18+08:00
hero: /images/posts/k8s-docker.jpg
menu:
  sidebar:
    name: Kubernetes Development
    identifier: k8s-develope
    parent: cloud-computing
    weight: 10
draft: false
---



## 资源模板

### `statefulset`举例

```yaml
apiVersion: apps/v1beta1
kind: StatefulSet
metadata: 
  name: kubia
spec:
  serviceName: kubia
  replicas: 2
  template:
    metadata:
      labels:
        app: kubia
    spec:
      containers:
      - name: kubia
        image: derios/kubia
        ports:
        - name: http
          containerPort: 8080
        volumeMounts:
        - name: data
          mountPath: /var/data
   volumeClaimTemplates:
   - metadata:
       name: data
     spec:
       resources:
         requests:
           storage: 1Mi
       accessModes:
       - ReadWriteOnce
       
```



### `headless service`举例

```yaml
apiVersion: v1
kind: Service
metadata:
  name: kubia
spec:
  clusterIP: None
  selector:
    app: kubia
  ports:
  - name: http
    port: 80
```



### `storage class` local PV 

```yaml
kind: StorageClass
apiVersion: storage.k8s.io/v1
metadata:
  name: local-storage
provisioner: kubernetes.io/no-provisioner
volumeBindingMode: WaitForFirstConsumer
```

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: local-test
spec:
  serviceName: "local-service"
  replicas: 3
  selector:
    matchLabels:
      app: local-test
  template:
    metadata:
      labels:
        app: local-test
    spec:
      containers:
      - name: test-container
        image: k8s.gcr.io/busybox
        command:
        - "/bin/sh"
        args:
        - "-c"
        - "sleep 100000"
        volumeMounts:
        - name: local-vol
          mountPath: /usr/test-pod
  volumeClaimTemplates:
  - metadata:
      name: local-vol
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: "local-storage"
      resources:
        requests:
          storage: 368Gi
```



## Key Point

### 使用Local PV的实际场景

- 使用本地磁盘作为缓存的系统
- CI/CD中用于存储构建中的系统
- 一些允许丢失和不需要保证可靠的数据(session, token)


### Local PV与HostPath的对比

`hostpath`:

- 绑定在pod的生命周期上,pod结束,pv则被删除.
- 可以通过pvc引用,也可以直接使用pv
- 使用node的磁盘,不经过网络,开销非常小

`本地持久卷`:

- 生命周期和node绑定,Kubernetes调度程序始终确保使用本地永久卷的Pod安排到同一节点
- 无法通过storageclass动态创建.
- 使用node磁盘,不经过网络,开销非常小.

```
The biggest difference is that the Kubernetes scheduler understands which node a Local Persistent Volume belongs to. With HostPath volumes, a pod referencing a HostPath volume may be moved by the scheduler to a different node resulting in data loss. But with Local Persistent Volumes, the Kubernetes scheduler ensures that a pod using a Local Persistent Volume is always scheduled to the same node.
```



### 使用Local PV的几个注意的问题

- 本地持久卷依然会丢失数据,例如node本身出了问题.
- 本地持久卷需要提供`volumeBindingMode:WaitForFirstConsumer`支持
- 不支持动态卷配置,实际上需要一个外部的controller来控制,包括创建pv和销毁pv并清理磁盘



### 使用Local PV例子

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: example-pv
spec:
  capacity:
    storage: 2Gi
  volumeMode: Filesystem
  accessModes:	
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Delete
  storageClassName: local-storage
  local:
    path: /home/tangxu/localpv
  # 比普通的pv就多了这个亲和性调度, 也是必须加的
  nodeAffinity:
    required:
      nodeSelectorTerms:
        - matchExpressions:
            - key: kubernetes.io/hostname
              operator: In
              values:
                - tangxu-pc
---
kind: Service
apiVersion: v1
metadata:
  name: local-pv-service
spec:
  selector:
    app: local-test
  clusterIP: None
  ports:
    - port: 8090
      targetPort: 80
      protocol: tcp
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: local-test
spec:
  serviceName: "local-pv-service"
  replicas: 3
  selector:
    matchLabels:
      app: local-test
  template:
    metadata:
      labels:
        app: local-test
    spec:
      containers:
        - name: test-container
          image: nginx:latest
          ports:
            - containerPort: 80
              protocol: tcp
              name: http
          volumeMounts:
            - name: local-vol
              mountPath: /usr/test-pod
  volumeClaimTemplates:
    - metadata:
        name: local-vol
      spec:
        accessModes:
          - "ReadWriteMany"
        storageClassName: "local-storage"
        resources:
          requests:
            storage: 1Gi
```

```shell
$ kubectl create ns test
namespace/test created
$ kubectl apply -f localPV.yaml
```



### Local PV的清理

在使用了local pv之后,清理就不再是简单的使用命令删除了,因为kubernetes不会为我们管理local pv的创建和删除工作(并且删除是阻塞的,主要依靠`finalizer`机制)

```shell
#先删除使用pv的资源
$ kubectl delete -f localPV.yaml
persistentvolume "example-pv" deleted
service "local-pv-service" deleted
statefulset.apps "local-test" deleted
#此处应该阻塞...
```

再起一个终端:

```shell
$ kubectl patch pv example-pv -p '{"metadata":{"finalizers": []}}' --type=merge
```



### `k8s`服务的概念

- 相同命名空间的可以用service名称作为主机名访问，可以查看`/etc/resolve.conf`

- 业务的数据库连接，如何规划暴露的服务，db-proxy是否需要刷新endpoints？

- 需要为pod添加就绪探针，让客户端只与正常的pod交互，而不管后端是否有pod出现问题，这样在就绪探针出问题了，`Endpoints`资源会去掉这个pod。

  ```yaml
  ...
  spec:
    containers:
    - name: kubia
      image: luksa/kubia
      readinessProbe:
        exec:
          command:
          - ls
          - /var/ready
  ```

- 应该通过删除pod或者更改pod标签而不是手动更改探针来从服务中手动移除pod

  ```
  如果想要从某个服务中手动添加或者删除pod的时候，把enabled=true作为标签添加到pod，以及服务的标签选择器中。当想要从服务中移除pod中，删除标签即可。
  ```

  



### 数据库详细配置

- 安装数据库时，数据库之间的路由配置从哪里来？

- 业务访问数据库时的网络管道...

- 数据库启停，pod的增删，以及前端服务显示形式。CRD中要定义描述数据库停止的状态信息，此时数据库存储和网络标识还在。

- 数据库agent检测的状态，如何同步给CR？

  - Operator主从状态确认行为

    `unreachableTimeout`  --- `pollingInterval`

- 主备HA，是否由controller控制？

  - `oracle times ten database是放在operator进行HA操作`

- 数据库软件包的下载时机和形式？

- 以挂卷方式将进程在容器内启动的可行性？

- database metadata的定义形式？

  - 由投射卷管理所有配置

  - admin信息，密码信息
  - user信息，密码信息
  - TLS配置

- 数据库的端口问题？

- 数据库集群部署state:

  - 主从

    ```
    - Initializing
    - Normal
    - ActiveDown
    - StandbyDown
    - StandbyPartiallyDown
    - StanbyPartiallyStarting
    - BothDown
    - Failed
    ```

  - 单机

    ```
    - Intializing
    - Normal
    - ActiveDown
    - Failed
    ```

- 数据库修改

  - 以修改连接数为例（方案一：采用`configmap`保存实例的连接配置信息、db信息等）
    1. 修改`configmap`中的字段
    2. `operator`删除stanby pod，并重新创建，该创建的standby pod采用新的`configmap`字段创建
    3. standby pod正常后，删除active pod进行自动倒换，`operator`再原地创建新的stanby pod，感知`configmap`中连接数字段创建。



###  资源声明

#### operator

```yaml
# operator deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zenith-operator
spec:
  replicas: 1
  selector:
    matchLabels:
      name: zenith-operator
  template:
    metadata:
      labels:
        name: zenith-operator
    spec:
      serviceAccountName: zenith-operator
      # imagePullSecrets:
      # - name: zenith-image-pulling-secret
      packageVesion: v1.0.0
      softwarePackage: zenith-operator
      packageType: tar
      containers:
        - name: zenith-operator
          # image: ....
          command:
          - zenith-operator
          imagePullPolicy: Never
          env:
            - name: WATCH_NAMESPACE
              valueFrom:
                fieldRef:
                  fieldPath: metadata.namespace
            - name: POD_NAME
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: OPERATOR_NAME
              value: "zenith-operator"
```



#### configmap /secrets 



#### DB Object

```yaml
apiVersion: huawei.cloudb.com/v1
kind: zenithCluster
metadata: 
  annotations: {}
  labels: {}
  name: ${zenith_cluster_name}
  namespace: ${cluster_namespace}
spec:
  zenithSpec:
    replicas: 2               # master-slave 
    version: v1.0.0           # eg.: Zenith-1.0.0.tar
    packageName: Zenith       # --- | 
    packageType: tar          # --- |
    storageClassName: local   # PV
    storageSize: 5G           # PV
    replicationSSLMandatory: false
    pollingInterval: 10
    unreachableTimeout: 30
    instanceConfigMap: 
    - zenith-instance-config
    dbConfigMap:
    - zenith-db-config
    dbSecrets:
    - zenith-db-secret
    dbSpecs:            
      ... 
  agentSpec:
    version: v1.0.0
    packageName: DBAgent
    packageType: tar
    agentConfig:
    - db-agent-sample-config
  template:
    affinity: 
      ...
    spec:
      selectors:
        matchLables:
          ...
      initContainers:
      - name: notify-download-package
        image: k8s.gcr.io/busybox
        commands:
        - sh
        - "-c"
        - |
          /bin/bash << 'EOF'
          This is used for notify nodeagent to download zenith software package
          EOF
      containers:
      - name: zenith-ha
        resources:
          requests:
            memory: "2048Mi"
            cpu: "1000m"
          limits:
            memory: "4096Mi"
            cpu: "2000m"
        ports:
          - name: listen-port
            containerPort: 32080
          - name: replication-port
            containerPort: 12345
        volumeMounts:
          - name: zenith-certs
            mountPath: /etc/certificate
            readOnly: true
          - name: watch-uds
            mountPath: /etc/uds/watch-server-uds
            readOnly: false
          - name: zenith-server-uds
            mountPath: /etc/uds/zenith-server-uds
            readOnly: false
          - name: agent-config
            mountPath: /etc/DBAgent/dbagent.conf
            readOnly: true
        env:
          - name: POD_NAME
            valueFrom:
            ...
          ...
      volumes:
      - name: 
  volumeClaimTemplates:   # PVC
    ...        
  
    
  
```



### 用投射卷聚合配置信息

[proposal](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/node/all-in-one-volume.md)

```
Constraints and Assumptions
1. The volume types must remain unchanged for backward compatibility
2. There will be a new volume type for this proposed functionality, but no other API changes
3. The new volume type should support atomic updates in the event of an input change
```

```
Use Cases
1. As a user, I want to automatically populate a single volume with the keys from multiple secrets, configmaps, and with downward API information, so that I can synthesize a single directory with various sources of information
2. As a user, I want to populate a single volume with the keys from multiple secrets, configmaps, and with downward API information, explicitly specifying paths for each item, so that I can have full control over the contents of that volume
```

#### 以前的情况

要使用secrets, configmaps, downward APIs都要在volumeMounts里面声明不同的mount paths:

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
    - name: mysecret
      mountPath: "/secrets"
      readOnly: true
    - name: podInfo
      mountPath: "/podinfo"
      readOnly: true
    - name: config-volume
      mountPath: "/config"
      readOnly: true
  volumes:
  - name: mysecret
    secret:
      secretName: jpeeler-db-secret
      items:
        - key: username
          path: my-group/my-username
  - name: podInfo
    downwardAPI:
      items:
        - path: "labels"
          fieldRef:
            fieldPath: metadata.labels
        - path: "annotations"
          fieldRef:
            fieldPath: metadata.annotations
  - name: config-volume
    configMap:
      name: special-config
      items:
        - key: special.how
          path: path/to/special-key
```



#### 投射卷的情况

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
      - secret:
          name: mysecret
          items:
            - key: user
              path: my-group/my-username
      - downwardAPI:
          items:
            - path: "labels"
              fieldRef:
                fieldPath: metadata.labels
            - path: "cpu_limit"
              resourceFieldRef:
                containerName: container-test
                resource: limits.cpu
      - configMap:
          name: myconfigmap
          items:
            - key: config
              path: my-group/my-config
```



### 简单部署`TiDB Operator`以及集群

#### `crd`安装

`kubectl apply -f https://raw.githubusercontent.com/pingcap/tidb-operator/v1.1.12/manifests/crd.yaml`



#### 安装operator

```
helm repo add pingcap https://charts.pingcap.org/

kubectl create namespace tidb-admin

helm install --namespace tidb-admin tidb-operator pingcap/tidb-operator --version v1.1.12

kubectl get pods --namespace tidb-admin -l app.kubernetes.io/instance=tidb-operator
```



#### 部署`TiDB`集群和监控

```
kubectl create namespace tidb-cluster && \
    kubectl -n tidb-cluster apply -f https://raw.githubusercontent.com/pingcap/tidb-operator/master/examples/basic/tidb-cluster.yaml
    
kubectl -n tidb-cluster apply -f https://raw.githubusercontent.com/pingcap/tidb-operator/master/examples/basic/tidb-monitor.yaml
```

