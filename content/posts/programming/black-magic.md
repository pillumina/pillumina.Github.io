---
title: "Black Magic"
date: 2020-12-20T12:40:29+08:00
hero: /images/posts/hero-google.svg
menu:
  sidebar:
    name: Black Magic
    identifier: black-magic
    weight: 10
draft: false
---

`阅读到的一些方便、有趣的技巧或者ideas的随手记录，后续考虑对相关话题专门开坑`

## 一行代码画出专业的论文图

[SciencePlots](https://github.com/garrettj403/SciencePlots)

计算机专业向来不缺少专业的绘图软件，从Excel到PPT，从最近沸沸扬扬的Matlab到Matplotlib、pyplot、ggplot，乃至其他更为专业的软件，着实丰富了我们的画图生活。

但是，这些软件或工具的背后，常常需要我们付出更多的努力：调色、统一格式、展示要高大上，等等。

现在，一款开源的软件工具包问世了：**SciencePlots**。它让你用一行代码画出天然高端且美观的论文图。

SciencePlots是一个依附于Matplotlib的扩展包，可以通过pip一键安装：

```
pip install SciencePlots
```

然后我们在画图时，只需要一句`with.plt.style.context(['science']):`，就可以画出非常美观且专业的图：

![plot1](https://pic1.zhimg.com/80/v2-90ced58bd948b48122c7c49f6dd3aeb8_1440w.jpg)

你还可以加一个选项`with.plt.style.context(['science','ieee']):`，就能画出IEEE格式的图：

![plot-ieee](https://pic3.zhimg.com/80/v2-65a94e294409928599dc91745f01662e_1440w.jpg)

甚至是超美的散点图：

![plot-scatter](https://pic4.zhimg.com/80/v2-db5c1cc749638e5bfba236fa9acdb4ff_1440w.jpg)

还有很多自定义的图像风格，保证节约我们的画图时间

> 这个包默认会调用latex来画图，如果不想用latex（也不是完全需要），可以在context里写一个属性'nolatex'即可。不然如果没有安装latex或latex路径配置有问题，则会报错。

