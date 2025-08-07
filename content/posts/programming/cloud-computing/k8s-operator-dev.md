---
title: "Kubernetes Operator Development History"
date: 2021-04-29T11:22:18+08:00
hero: /images/posts/k8s-docker.jpg
menu:
  sidebar:
    name: Kubernetes Operator Development History
    identifier: k8s-operator-dec
    parent: cloud-computing
    weight: 10
draft: false
---

>本文旨在记录对中间件、编排组件容器化部署后，实现kubernetes扩展组件Controller的过程。



## Third-Parties

[kubernetes-client: javascript](https://github.com/kubernetes-client/javascript)

[client-go](https://github.com/kubernetes/client-go)

[kube-rs](https://github.com/clux/kube-rs)



## `client-go`源码分析

### 目录结构

>- `kubernetes`: contains the clientset to access Kubernetes API.
>- `discovery`: discover APIs supported by a Kubernetes API server.
>- `dynamic`: contains a dynamic client that can perform generic operations on arbitrary Kubernetes API objects.
>- `transport`: set up auth and start a connection.
>- `tools/cache`: useful for writing controllers.
>- `informers`:  informer group
>- `listers`:  lister group

### 代码实例

```shell
git clone https://github.com/huweihuang/client-go.git
cd client-go
#保证本地HOME目录有配置kubernetes集群的配置文件
go run client-go.go
```

`client-go.go`

```go
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
)

func main() {
	var kubeconfig *string
	if home := homeDir(); home != "" {
		kubeconfig = flag.String("kubeconfig", filepath.Join(home, ".kube", "config"), "(optional) absolute path to the kubeconfig file")
	} else {
		kubeconfig = flag.String("kubeconfig", "", "absolute path to the kubeconfig file")
	}
	flag.Parse()
	// uses the current context in kubeconfig
	config, err := clientcmd.BuildConfigFromFlags("", *kubeconfig)
	if err != nil {
		panic(err.Error())
	}
	// creates the clientset
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		panic(err.Error())
	}
	for {
		pods, err := clientset.CoreV1().Pods("").List(metav1.ListOptions{})
		if err != nil {
			panic(err.Error())
		}
		fmt.Printf("There are %d pods in the cluster\n", len(pods.Items))
		time.Sleep(10 * time.Second)
	}
}

func homeDir() string {
	if h := os.Getenv("HOME"); h != "" {
		return h
	}
	return os.Getenv("USERPROFILE") // windows
}
```

`output`

```shell
➜ go run client-go.go
There are 9 pods in the cluster
There are 7 pods in the cluster
There are 7 pods in the cluster
There are 7 pods in the cluster
There are 7 pods in the cluster
```

### `kubeconfig`

```go
kubeconfig = flag.String("kubeconfig", filepath.Join(home, ".kube", "config"), "(optional) absolute path to the kubeconfig file")
```

  为了获取k8s配置文件`kubeconfig`的绝对路径，一般路径为`$HOME/.kube/config`，这个文件主要用来配置本地连接的k8s集群。

  config的内容大概如图所示：

```yaml
apiVersion: v1
clusters:
- cluster:
    server: http://<kube-master-ip>:8080
  name: k8s
contexts:
- context:
    cluster: k8s
    namespace: default
    user: ""
  name: default
current-context: default
kind: Config
preferences: {}
users: []
```

### `rest.config`  

  通过参数和`BuildConfigFromFlags`方法获取`rest.Config`对象

```go
config, err := clientcmd.BuildConfigFromFlags("", *kubeconfig)
```

[k8s.io/client-go/tools/clientcmd/client_config.go](https://github.com/kubernetes/client-go/blob/master/tools/clientcmd/client_config.go#L522)

```go
// BuildConfigFromFlags is a helper function that builds configs from a master
// url or a kubeconfig filepath. These are passed in as command line flags for cluster
// components. Warnings should reflect this usage. If neither masterUrl or kubeconfigPath
// are passed in we fallback to inClusterConfig. If inClusterConfig fails, we fallback
// to the default config.
func BuildConfigFromFlags(masterUrl, kubeconfigPath string) (*restclient.Config, error) {
	if kubeconfigPath == "" && masterUrl == "" {
		klog.Warning("Neither --kubeconfig nor --master was specified.  Using the inClusterConfig.  This might not work.")
		kubeconfig, err := restclient.InClusterConfig()
		if err == nil {
			return kubeconfig, nil
		}
		klog.Warning("error creating inClusterConfig, falling back to default config: ", err)
	}
	return NewNonInteractiveDeferredLoadingClientConfig(
		&ClientConfigLoadingRules{ExplicitPath: kubeconfigPath},
		&ConfigOverrides{ClusterInfo: clientcmdapi.Cluster{Server: masterUrl}}).ClientConfig()
}
```

 ### `clientset`

  通过`*rest.Config`参数和`NewForConfig`方法来获取`clientset`对象，`clientset`是多个`client`集合，每个`client`可能包括不同版本的方法调用

```go
clientset, err := kubernetes.NewForConfig(config)
```

#### `NewForConfig`

  `NewForConfig`函数就是初始化`clientset`中的每个client。

[k8s.io/client-go/kubernetes/clientset.go](https://github.com/kubernetes/client-go/blob/87887458218a51f3944b2f4c553eb38173458e97/kubernetes/clientset.go#L396)

```go
// NewForConfig creates a new Clientset for the given config.
func NewForConfig(c *rest.Config) (*Clientset, error) {
	configShallowCopy := *c
	...
	var cs Clientset
	cs.appsV1beta1, err = appsv1beta1.NewForConfig(&configShallowCopy)
	...
	cs.coreV1, err = corev1.NewForConfig(&configShallowCopy)
	...
}
```

  

#### `clientset`的结构体

[k8s.io/client-go/kubernetes/clientset.go](https://github.com/kubernetes/client-go/blob/master/kubernetes/clientset.go)

```go
// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	admissionregistrationV1      *admissionregistrationv1.AdmissionregistrationV1Client
	admissionregistrationV1beta1 *admissionregistrationv1beta1.AdmissionregistrationV1beta1Client
	internalV1alpha1             *internalv1alpha1.InternalV1alpha1Client
	appsV1                       *appsv1.AppsV1Client
	appsV1beta1                  *appsv1beta1.AppsV1beta1Client
	appsV1beta2                  *appsv1beta2.AppsV1beta2Client
	authenticationV1             *authenticationv1.AuthenticationV1Client
	authenticationV1beta1        *authenticationv1beta1.AuthenticationV1beta1Client
	authorizationV1              *authorizationv1.AuthorizationV1Client
	authorizationV1beta1         *authorizationv1beta1.AuthorizationV1beta1Client
	autoscalingV1                *autoscalingv1.AutoscalingV1Client
	autoscalingV2beta1           *autoscalingv2beta1.AutoscalingV2beta1Client
	autoscalingV2beta2           *autoscalingv2beta2.AutoscalingV2beta2Client
	batchV1                      *batchv1.BatchV1Client
	batchV1beta1                 *batchv1beta1.BatchV1beta1Client
	certificatesV1               *certificatesv1.CertificatesV1Client
	certificatesV1beta1          *certificatesv1beta1.CertificatesV1beta1Client
	coordinationV1beta1          *coordinationv1beta1.CoordinationV1beta1Client
	coordinationV1               *coordinationv1.CoordinationV1Client
	coreV1                       *corev1.CoreV1Client
	discoveryV1                  *discoveryv1.DiscoveryV1Client
	discoveryV1beta1             *discoveryv1beta1.DiscoveryV1beta1Client
	eventsV1                     *eventsv1.EventsV1Client
	eventsV1beta1                *eventsv1beta1.EventsV1beta1Client
	extensionsV1beta1            *extensionsv1beta1.ExtensionsV1beta1Client
	flowcontrolV1alpha1          *flowcontrolv1alpha1.FlowcontrolV1alpha1Client
	flowcontrolV1beta1           *flowcontrolv1beta1.FlowcontrolV1beta1Client
	networkingV1                 *networkingv1.NetworkingV1Client
	networkingV1beta1            *networkingv1beta1.NetworkingV1beta1Client
	nodeV1                       *nodev1.NodeV1Client
	nodeV1alpha1                 *nodev1alpha1.NodeV1alpha1Client
	nodeV1beta1                  *nodev1beta1.NodeV1beta1Client
	policyV1                     *policyv1.PolicyV1Client
	policyV1beta1                *policyv1beta1.PolicyV1beta1Client
	rbacV1                       *rbacv1.RbacV1Client
	rbacV1beta1                  *rbacv1beta1.RbacV1beta1Client
	rbacV1alpha1                 *rbacv1alpha1.RbacV1alpha1Client
	schedulingV1alpha1           *schedulingv1alpha1.SchedulingV1alpha1Client
	schedulingV1beta1            *schedulingv1beta1.SchedulingV1beta1Client
	schedulingV1                 *schedulingv1.SchedulingV1Client
	storageV1beta1               *storagev1beta1.StorageV1beta1Client
	storageV1                    *storagev1.StorageV1Client
	storageV1alpha1              *storagev1alpha1.StorageV1alpha1Client
}
```



#### `clientset.Interface`

`clientset`实现了以下的interface，可以通过以下的方法获得具体的client，例如:

```go
pods, err := clientset.CoreV1().Pods("").List(metav1.ListOptions{})
```

```go
type Interface interface {
	Discovery() discovery.DiscoveryInterface
	AdmissionregistrationV1() admissionregistrationv1.AdmissionregistrationV1Interface
	AdmissionregistrationV1beta1() admissionregistrationv1beta1.AdmissionregistrationV1beta1Interface
	InternalV1alpha1() internalv1alpha1.InternalV1alpha1Interface
	AppsV1() appsv1.AppsV1Interface
	AppsV1beta1() appsv1beta1.AppsV1beta1Interface
	AppsV1beta2() appsv1beta2.AppsV1beta2Interface
	AuthenticationV1() authenticationv1.AuthenticationV1Interface
	AuthenticationV1beta1() authenticationv1beta1.AuthenticationV1beta1Interface
	AuthorizationV1() authorizationv1.AuthorizationV1Interface
	AuthorizationV1beta1() authorizationv1beta1.AuthorizationV1beta1Interface
	AutoscalingV1() autoscalingv1.AutoscalingV1Interface
	AutoscalingV2beta1() autoscalingv2beta1.AutoscalingV2beta1Interface
	AutoscalingV2beta2() autoscalingv2beta2.AutoscalingV2beta2Interface
	BatchV1() batchv1.BatchV1Interface
	BatchV1beta1() batchv1beta1.BatchV1beta1Interface
	CertificatesV1() certificatesv1.CertificatesV1Interface
	CertificatesV1beta1() certificatesv1beta1.CertificatesV1beta1Interface
	CoordinationV1beta1() coordinationv1beta1.CoordinationV1beta1Interface
	CoordinationV1() coordinationv1.CoordinationV1Interface
	CoreV1() corev1.CoreV1Interface
	DiscoveryV1() discoveryv1.DiscoveryV1Interface
	DiscoveryV1beta1() discoveryv1beta1.DiscoveryV1beta1Interface
	EventsV1() eventsv1.EventsV1Interface
	EventsV1beta1() eventsv1beta1.EventsV1beta1Interface
	ExtensionsV1beta1() extensionsv1beta1.ExtensionsV1beta1Interface
	FlowcontrolV1alpha1() flowcontrolv1alpha1.FlowcontrolV1alpha1Interface
	FlowcontrolV1beta1() flowcontrolv1beta1.FlowcontrolV1beta1Interface
	NetworkingV1() networkingv1.NetworkingV1Interface
	NetworkingV1beta1() networkingv1beta1.NetworkingV1beta1Interface
	NodeV1() nodev1.NodeV1Interface
	NodeV1alpha1() nodev1alpha1.NodeV1alpha1Interface
	NodeV1beta1() nodev1beta1.NodeV1beta1Interface
	PolicyV1() policyv1.PolicyV1Interface
	PolicyV1beta1() policyv1beta1.PolicyV1beta1Interface
	RbacV1() rbacv1.RbacV1Interface
	RbacV1beta1() rbacv1beta1.RbacV1beta1Interface
	RbacV1alpha1() rbacv1alpha1.RbacV1alpha1Interface
	SchedulingV1alpha1() schedulingv1alpha1.SchedulingV1alpha1Interface
	SchedulingV1beta1() schedulingv1beta1.SchedulingV1beta1Interface
	SchedulingV1() schedulingv1.SchedulingV1Interface
	StorageV1beta1() storagev1beta1.StorageV1beta1Interface
	StorageV1() storagev1.StorageV1Interface
	StorageV1alpha1() storagev1alpha1.StorageV1alpha1Interface
}
```

### `CoreV1Client`

  常用的`CoreV1Client`为例子分析

#### `corev1.NewForConfig`

```go
// NewForConfig creates a new CoreV1Client for the given config.
func NewForConfig(c *rest.Config) (*CoreV1Client, error) {
	config := *c
	if err := setConfigDefaults(&config); err != nil {
		return nil, err
	}
	client, err := rest.RESTClientFor(&config)
	if err != nil {
		return nil, err
	}
	return &CoreV1Client{client}, nil
}
```

  通过传入配置信息`rest.Config`来实例化对象，其本质是调用了`rest.RESTClientFor(&Config)`方法创建了`RESTClient`的对象，即`CoreV1Client`的本质就是一个`RESTClient`对象。



#### `CoreV1Client`结构体定义

```go
// CoreV1Client is used to interact with features provided by the  group.
type CoreV1Client struct {
	restClient rest.Interface
}
```

  `CoreV1Client`实现了`CoreV1Interface`接口，从而对k8s的资源对象进行增删改查的操作。

```go
//CoreV1Client的方法
func (c *CoreV1Client) ComponentStatuses() ComponentStatusInterface {...}
//ConfigMaps
func (c *CoreV1Client) ConfigMaps(namespace string) ConfigMapInterface {...}
//Endpoints
func (c *CoreV1Client) Endpoints(namespace string) EndpointsInterface {...}
func (c *CoreV1Client) Events(namespace string) EventInterface {...}
func (c *CoreV1Client) LimitRanges(namespace string) LimitRangeInterface {...}
//Namespaces
func (c *CoreV1Client) Namespaces() NamespaceInterface {...}
//Nodes
func (c *CoreV1Client) Nodes() NodeInterface {...}
func (c *CoreV1Client) PersistentVolumes() PersistentVolumeInterface {...}
func (c *CoreV1Client) PersistentVolumeClaims(namespace string) PersistentVolumeClaimInterface {...}
//Pods
func (c *CoreV1Client) Pods(namespace string) PodInterface {...}
func (c *CoreV1Client) PodTemplates(namespace string) PodTemplateInterface {...}
//ReplicationControllers
func (c *CoreV1Client) ReplicationControllers(namespace string) ReplicationControllerInterface {...}
func (c *CoreV1Client) ResourceQuotas(namespace string) ResourceQuotaInterface {...}
func (c *CoreV1Client) Secrets(namespace string) SecretInterface {...}
//Services
func (c *CoreV1Client) Services(namespace string) ServiceInterface {...}
func (c *CoreV1Client) ServiceAccounts(namespace string) ServiceAccountInterface {...}

```



#### `CoreV1Interface`

```go
type CoreV1Interface interface {
	RESTClient() rest.Interface
	ComponentStatusesGetter
	ConfigMapsGetter
	EndpointsGetter
	EventsGetter
	LimitRangesGetter
	NamespacesGetter
	NodesGetter
	PersistentVolumesGetter
	PersistentVolumeClaimsGetter
	PodsGetter
	PodTemplatesGetter
	ReplicationControllersGetter
	ResourceQuotasGetter
	SecretsGetter
	ServicesGetter
	ServiceAccountsGetter
}
```

  `CoreV1Interface`中包含了各种k8s对象的调用接口，例如`PodsGetter`是对k8s中pod对象增删改查的接口。`ServicesGetter`是对service对象的操作的接口。	

#### `PodsGetter`接口举例探索

 可以以`PodsGetter`接口为例来研究一下`CoreV1Client`对pod对象的增删改查调用。

 示例中的代码如下：

```go
pods, err := clientset.CoreV1().Pods("").List(metav1.ListOptions{})
```

 

**CoreV1().Pods()**

```go
// core_client.go
func (c *CoreV1Client) Pods(namespace string) PodInterface {
	return newPods(c, namespace)
}

// pod.go
// newPods returns a Pods
func newPods(c *CoreV1Client, namespace string) *pods {
	return &pods{
		client: c.RESTClient(),
		ns:     namespace,
	}
}
```

`CoreV1().Pods()`方法实际上调用了`newPods()`方法，创建了一个pod对象，该对象继承了`rest.Interface`接口，即最终的实现本质是`RESTClient`的HTTP调用。

```go
// pods implements PodInterface
type pods struct {
	client rest.Interface
	ns     string
}
```

  `pods`对象实现了`PodInterface`接口：

```go
// PodInterface has methods to work with Pod resources.
type PodInterface interface {
	Create(*v1.Pod) (*v1.Pod, error)
	Update(*v1.Pod) (*v1.Pod, error)
	UpdateStatus(*v1.Pod) (*v1.Pod, error)
	Delete(name string, options *metav1.DeleteOptions) error
	DeleteCollection(options *metav1.DeleteOptions, listOptions metav1.ListOptions) error
	Get(name string, options metav1.GetOptions) (*v1.Pod, error)
	List(opts metav1.ListOptions) (*v1.PodList, error)
	Watch(opts metav1.ListOptions) (watch.Interface, error)
	Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1.Pod, err error)
	GetEphemeralContainers(podName string, options metav1.GetOptions) (*v1.EphemeralContainers, error)
	UpdateEphemeralContainers(podName string, ephemeralContainers *v1.EphemeralContainers) (*v1.EphemeralContainers, error)

	PodExpansion

```

  这个interface定义了`pods`对象的增删改查等方法。



`PodsGetter` 继承了`PodInterface`的接口:

```go
// PodsGetter has a method to return a PodInterface.
// A group's client should implement this interface.
type PodsGetter interface {
	Pods(namespace string) PodInterface
}
```

 

`Pods().List()`方法通过`RESTClient`的HTTP调用来实现对k8s的pod资源的获取:

```go
// List takes label and field selectors, and returns the list of Pods that match those selectors.
func (c *pods) List(opts metav1.ListOptions) (result *v1.PodList, err error) {
	var timeout time.Duration
	if opts.TimeoutSeconds != nil {
		timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
	}
	result = &v1.PodList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("pods").
		VersionedParams(&opts, scheme.ParameterCodec).
		Timeout(timeout).
		Do().
		Into(result)
	return
}
```

  以上分析了`clientset.CoreV1().Pods("").List(metav1.ListOptions{})`对pod资源获取的过程，最终是调用`RESTClient`的方法实现。



### `RESTClient`

  待补充



### 总结

  `client-go`对K8s资源对象的调用，需要先获取k8s的配置信息，也就是`$HOME/.kube/config`。

  调用顺序如下：

>`kubeconfig` -> `rest.config` -> `clientset` -> 具体的client(CoreV1Client) -> 具体的资源对象(如pod) -> `RESTClient` -> `http.Client` -> HTTP 请求的发送和响应

  常用的client有`CoreV1Client`、`AppsV1betaClient`、`ExtensionsV1beta1Client`等。



## `Operator SDK` 

