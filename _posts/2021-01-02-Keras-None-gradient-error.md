---
layout: post
title: "Keras:An operation has `None` for gradient. Please make sure that all of your ops have a gradient"
subtitle: ""
author: 明昊
header-style: text
tags:
  - Keras
  - 深度学习
---

Keras 报错：

ValueError: An operation has `None` for gradient. Please make sure that all of your ops have a gradient defined (i.e. are differentiable). Common ops without gradient: K.argmax, K.round, K.eval.

意思是说，构建的模型里面包含一些类似于K.argmax, K.round, K.eval.不可导的操作。
然而，模型构建的时候并没有使用这些操作。

后面查了资料才晓得：

**如果自定义的层所添加的参数（在build里面添加）或者带层数的keras.layer，并没有在call函数里面使用的话，就会导致这些参数无法执行梯度更新，并不是使用了K.argmax, K.round, K.eval.的原因。**

因此解决方法：

 - 检查call()函数，把没有使用到的层或参数从build()里面去掉
 - 检查call()函数，看是否有那些定义的参数没有被使用

https://stackoverflow.com/questions/47624843/custom-keras-layer-troubles






