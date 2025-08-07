---
title: "Docker Cheat Sheet"
date: 2021-03-30T11:22:18+08:00
hero: /images/posts/k8s-docker.jpg
menu:
  sidebar:
    name: Docker Cheat Sheet
    identifier: docker-cheatsheet
    parent: cloud-computing
    weight: 10
draft: false
---



## Books

[Docker in Action (English ver.)](https://pepa.holla.cz/wp-content/uploads/2016/10/Docker-in-Action.pdf)

[Docker入门到实践(中文)](https://yeasy.gitbook.io/docker_practice/)

## 速查

[Docker Cheat Sheet](https://github.com/wsargent/docker-cheat-sheet/tree/master/zh-cn)

### 全量CLI

![docker cheat sheet](https://raw.githubusercontent.com/sangam14/dockercheatsheets/master/dockercheatsheet8.png)

### 容器管理CLI

![container management commands](https://raw.githubusercontent.com/sangam14/dockercheatsheets/master/dockercheatsheet1.png)

### 查看容器CLI

![inspect container commands](https://raw.githubusercontent.com/sangam14/dockercheatsheets/master/dockercheatsheet3.png)

### 容器交互CLI

![interact with container commands](https://raw.githubusercontent.com/sangam14/dockercheatsheets/master/dockercheatsheet4.png)

### 镜像管理CLI

![image management commands](https://raw.githubusercontent.com/sangam14/dockercheatsheets/master/dockercheatsheet5.png)

### 镜像传输CLI

![image transfer commands](https://raw.githubusercontent.com/sangam14/dockercheatsheets/master/dockercheatsheet6.png)

### DOCKERFILE主要命令

![builder main commands](https://raw.githubusercontent.com/sangam14/dockercheatsheets/master/dockercheatsheet7.png)

## Dockerfile

### 基底

```dockerfile
FROM ruby:2.2.2
```



### 变量

```dockerfile
ENV APP_HOME/myapp
RUN mkdir $APP_HOME
```



### 初始化

```dockerfile
RUN bundle install
```

```dockerfile
WORKDIR /myapp
```

```dockerfile
VOLUME ["/data"]
# Specification for mount point
```

```dockerfile
ADD file.xyz /file.xyz
COPY --chown=user:group host_file.xyz /path/container_file.xyz
```



### Onbuild

```dockerfile
ONBUILD RUN bundle install
# when used with another file
```



### 命令

```dockerfile
EXPOSE 5900
CMD ["bundle", "exec", "rails", "server"]
```



### Entrypoint

```dockerfile
ENTRYPOINT exec top -b
```



### Metadata

```dockerfile
LABEL version="1.0"
```

```dockerfile
LABEL "com.example.vendor"="ACME Incorporated"
LABEL com.example.label-with-value="foo"
```

```dockerfile
LABEL description="This text illustrates \
that label-values can span multiple lines."
```



## Docker Compose

### 基本用法

```yaml
# docker-compose.yml
version: '2'

services:
  web:
    build: .
    # build from Dockerfile
    context: ./Path
    dockerfile: Dockerfile
    ports:
     - "5000:5000"
    volumes:
     - .:/code
  redis:
    image: redis
```

### 指令

```shell
docker-compose start
docker-compose stop
```

```shell
docker-compose pause
docker-compose unpause
```

```shell
docker-compose ps
docker-compose up
docker-compose down
```



## Reference(例子)

### 构建

```yaml
web:
  # build from Dockerfile
  build: .
```

```yaml
  # build from custom Dockerfile
  build:
    context: ./dir
    dockerfile: Dockerfile.dev
```

```yaml
 # build from image
  image: ubuntu
  image: ubuntu:14.04
  image: tutum/influxdb
  image: example-registry:4000/postgresql
  image: a4bc65fd
```



### 端口

```yaml
  ports:
    - "3000"
    - "8000:80"  # guest:host
```

```yaml
  # expose ports to linked services (not to host)
  expose: ["3000"]
```



### 指令

```yaml
  # command to execute
  command: bundle exec thin -p 3000
  command: [bundle, exec, thin, -p, 3000]
```

```yaml
  # override the entrypoint
  entrypoint: /app/start.sh
  entrypoint: [php, -d, vendor/bin/phpunit]
```



### 环境变量

```yaml
  # environment vars
  environment:
    RACK_ENV: development
  environment:
    - RACK_ENV=development
```

```yaml
  # environment vars from file
  env_file: .env
  env_file: [.env, .development.env]
```



### 依赖

```yaml
  # makes the `db` service available as the hostname `database`
  # (implies depends_on)
  links:
    - db:database
    - redis
```

```yaml
  # make sure `db` is alive before starting
  depends_on:
    - db
```



### 其他选项

```yaml
  # make this service extend another
  extends:
    file: common.yml  # optional
    service: webapp
```

```yaml
  volumes:
    - /var/lib/mysql
    - ./_data:/var/lib/mysql
```



## 高级特性

### 打标签

```yaml
services:
  web:
    labels:
      com.example.description: "Accounting web app"
```



### DNS服务器

```yaml
services:
  web:
    dns: 8.8.8.8
    dns:
      - 8.8.8.8
      - 8.8.4.4
```



### 设备绑定

```yaml
services:
  web:
    devices:
    - "/dev/ttyUSB0:/dev/ttyUSB0"
```



### 外部链接

```yaml
services:
  web:
    external_links:
      - redis_1
      - project_db_1:mysql
```



### 主机设置

```yaml
services:
  web:
    extra_hosts:
      - "somehost:192.168.1.100"
```



### Services

```shell
# To view list of all the services runnning in swarm
docker service ls 

# To see all running services
docker stack services stack_name

# to see all services logs
docker service logs stack_name service_name 

# To scale services quickly across qualified node
docker service scale stack_name_service_name=replicas
```



## Clean up

```shell
# To clean or prune unused (dangling) images
docker image prune 

# To remove all images which are not in use containers , add - a
docker image prune -a 

# To Purne your entire system
docker system prune 

# To leave swarm
docker swarm leave

# To remove swarm ( deletes all volume data and database info)
docker stack rm stack_name 

# To kill all running containers
docker kill $(docekr ps -q ) 
```

