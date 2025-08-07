---
title: "Python类自定义"
date: 2020-12-14T16:20:29+08:00
hero: /images/posts/python-coder.jpg
menu:
  sidebar:
    name: Python类自定义
    identifier: python-custom-class
    parent: python odyssey
    weight: 10
draft: false
---

## python类关键字

### `__init__` vs `__new__`

`__init__`为初始化方法，`__new__`为真正的构造函数。

### 描述符Descriptor

### `__contains__`

### `__slots__`

### 定制类

### type()

  python作为动态语言，和静态语言最大的不同，即函数和类的定义，不是编译的时候创建的而是动态创建的。我们常见的对类的定义:

```python
class Hello(object):
    def hello(self, name='world'):
        print('Hello, %s.' % name)
```

```
>>> from hello import Hello
>>> h = Hello()
>>> h.hello()
Hello, world.
>>> print(type(Hello))
<class 'type'>
>>> print(type(h))
<class 'hello.Hello'>
```

  type()函数可以查看一类类型或者变量的类型，`Hello`是一个class， 它的类型是个`type`，而`h`是一个instance, 它的类型就是class `Hello`。

  同时有一个概念，就是type()不仅可以返回对象的类型，还可以创建出新的类型。我们可以不用定义`class Hello() ...`而动态创建出Hello类。

```
>>> def fn(self, name='world'): # 先定义函数
...     print('Hello, %s.' % name)
...
>>> Hello = type('Hello', (object,), dict(hello=fn)) # 创建Hello class
>>> h = Hello()
>>> h.hello()
Hello, world.
>>> print(type(Hello))
<class 'type'>
>>> print(type(h))
<class '__main__.Hello'>
```

  创建一个class对象，`type()`函数传入3个参数：

1. class名称
2. 继承的父类集合，python支持多重继承，只有一个父类的话需要加上tuple的单元素写法
3. class的方法名称与参数绑定，上面的例子里就把函数`fn`绑定到方法名`hello`上

  通过`type()`函数创建的类和直接写class是一样的，python解释器遇到class定义时，也仅仅是扫描class定义的语法，然后调用type()函数创建出class。

  

### MetaClass

  除了使用`type()`进行动态类创建，如果要控制类的创建行为，还可以使用metaclass。对其简单的解释就是：当我们定义了class以后，就可以根据这个class创建出实例，也就是先定义class，再创建实例；但是，如果我们希望创建class该怎么办？这里就必须用到metaclass创建class，所以先定义metaclass，然后创建class。

  因此，metaclass允许创建类或者修改类，也就是我们可以把类看成metaclass创建出来的“实例”。metaclass在python中相对比较难理解，而且很多场景不需要用，毕竟它能够改变类创建时的行为(behaviour)，不熟容易导致一些问题。

  下面是一个简单的为自定义的MyList类增加一个`add`方法的例子：

  先定义`ListMetaClass`, 一般来说metaclass的类以Metaclass结尾。

```python
# metaclass是类的模板，所以必须从`type`类型派生：
class ListMetaclass(type):
    def __new__(cls, name, bases, attrs):
        attrs['add'] = lambda self, value: self.append(value)
        return type.__new__(cls, name, bases, attrs)
```

  有了这个定义，在定义MyList的时候传入关键字`metaclass`即可：

```python
class MyList(list, metaclass=ListMetaclass):
    pass
```

使用关键字后，python解释器在创建`MyList`的时候，要通过`ListMetaclass.__new__()`来创建。

`__new__()`方法接受到的参数为：

1. 当前准备创建的类的对象
2. 类的名字
3. 类继承的父类集合
4. 类的方法集合

测试一下是否正确加上方法：

```python
>>> L = MyList()
>>> L.add(1)
>> L
[1]
```

而普通的`liat`是没有`add`方法的。那么动态修改的意义何在？直接在`MyList`里新增`add`方法不是更简单？正常情况下的确不需要用metaclass，不过还是有一些场景需要用到，比如ORM(Object Relational Mapping) --- 把关系型数据库的一行映射成一个对象，即一个类对应一个表，这样写代码更简单而不需要SQL语句。如果要编写这样一个ORM框架，所有的类都只能动态定义了，因为只有使用者才能根据表的结构定义对应的类。

下面写一个简单的ORM框架，比如使用者想定义一个`User`类来操作对应的数据库表`user`，我们期待使用者写出如下的代码:

```python
class User(Model):
    # 定义类的属性到列的映射：
    id = IntegerField('id')
    name = StringField('username')
    email = StringField('email')
    password = StringField('password')

# 创建一个实例：
u = User(id=12345, name='Michael', email='test@orm.org', password='my-pwd')
# 保存到数据库：
u.save()
```

其中父类`Model`和属性类型`StringField`，`IntegerField`由ORM框架提供，剩下的`save()`全部由metaclass自动完成。

先定定义`Field`类，用于负责保存数据库表的字段名和字段类型：

```python
class Field(object):

    def __init__(self, name, column_type):
        self.name = name
        self.column_type = column_type

    def __str__(self):
        return '<%s:%s>' % (self.__class__.__name__, self.name)
```

基于此类，定义其他Field子类：

```python
class StringField(Field):

    def __init__(self, name):
        super(StringField, self).__init__(name, 'varchar(100)')

class IntegerField(Field):

    def __init__(self, name):
        super(IntegerField, self).__init__(name, 'bigint')
```

编写`Model`和`ModelMetaclass`：

```python
class ModelMetaclass(type):

    def __new__(cls, name, bases, attrs):
        if name=='Model':
            return type.__new__(cls, name, bases, attrs)
        print('Found model: %s' % name)
        mappings = dict()
        for k, v in attrs.items():
            if isinstance(v, Field):
                print('Found mapping: %s ==> %s' % (k, v))
                mappings[k] = v
        for k in mappings.keys():
            attrs.pop(k)
        attrs['__mappings__'] = mappings # 保存属性和列的映射关系
        attrs['__table__'] = name # 假设表名和类名一致
        return type.__new__(cls, name, bases, attrs)
```

```python
class Model(dict, metaclass=ModelMetaclass):

    def __init__(self, **kw):
        super(Model, self).__init__(**kw)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(r"'Model' object has no attribute '%s'" % key)

    def __setattr__(self, key, value):
        self[key] = value

    def save(self):
        fields = []
        params = []
        args = []
        for k, v in self.__mappings__.items():
            fields.append(v.name)
            params.append('?')
            args.append(getattr(self, k, None))
        sql = 'insert into %s (%s) values (%s)' % (self.__table__, ','.join(fields), ','.join(params))
        print('SQL: %s' % sql)
        print('ARGS: %s' % str(args))
```

**注意: 当用户定义一个`class User(Model)`的时候，python解释器首先在当前类`User`定义中查找`metaclass`，如果没找到，继续在父类`Model`中找，找到了就使用`Model`中定义的`metaclass`的`ModelMetaclass`来创建`User`类，所以metaclass可以隐式得继承到子类**

`ModelMetaclass`中的逻辑：

1. 排除掉对`Model`类的修改
2. 在当前类（如`User`）中查找定义的类的所有属性，如果找到一个`Field`类，则将其保存到一个`__mapping__`的dict中，同时从类属性中删除该Field属性，防止runtime错误。（实例同名属性对类同名属性的覆盖）
3. 把表名保存到`__table__`中

`Model`类中，就可以定义各种操作数据库的方法，比如`save`, `delete`, `update`等等。

用上述的模块，可以编写出如：

```python
u = User(id=12345, name='Michael', email='test@orm.org', password='my-pwd')
u.save()
```

获得的结果:

```python
Found model: User
Found mapping: email ==> <StringField:email>
Found mapping: password ==> <StringField:password>
Found mapping: id ==> <IntegerField:uid>
Found mapping: name ==> <StringField:username>
SQL: insert into User (password,email,username,id) values (?,?,?,?)
ARGS: ['my-pwd', 'test@orm.org', 'Michael', 12345]
```

这里只是简单打出参数列表，对backend连接没有进行真正的处理。