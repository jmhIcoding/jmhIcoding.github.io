---
layout: post
title: Keras 载入历史模型报错： AttributeError: ‘str‘ object has no attribute ‘decode‘
subtitle: ""
author: 明昊
header-style: text
tags:
  - Keras
  - Keras踩坑
---


Keras 2.3.0 载入历史模型时报错：AttributeError: 'str' object has no attribute 'decode'

解决方法：
# 1. 降级h5py

```python
pip3 install h5py==2.10.0
```
# 2. 更换模型载入方式
上面的报错出现在调用`load_weights（）` 载入模型参数的过程中，然而载入历史模型还可以调用`keras.models.load_model`函数，按照如下载入即可：

```python
model= keras.models.load_model(model_path,custom_objects= {'Denoising_Enhancing_layer':Denoising_Enhancing_layer},compile=False)
```
其中Denoising_Enhancing_layer 是自定义的Keras层，注意如果有自定义层需要为custom_objects指定类似的参数。

**天坑： 需要指出的是，load_model是个天坑。它载入模型（尤其是复杂模型，比如带残差的，带多个输入输出的模型）的时候，似乎没有真正把该载入的参数正确载入到模型里面去。于是会出现：训练阶段，测试数据集的结果很好；保存模型后，重新载入模型，再测试，可是测试结果特别差 的诡异事情！**
当出现这个天坑的时候，请重新使用`load_weights` 去载入历史模型！







