---
layout: post
title: Keras ValueError: Unknown layer 自定义层历史参数无法载入
subtitle: Keras ValueError: Unknown layer
author: 明昊
header-style: text
tags:
  - Keras踩坑
  - Keras踩坑
---

导入模型的时候出现： File "/root/anaconda3/lib/python3.6/site-packages/keras/utils/generic_utils.py", line 140, in deserialize_keras_object
    ': ' + class_name)
ValueError: Unknown layer: Denoising_layer

Denoising_layer是我自己写的一个层，现在导入已经训练的模型时报这个错误。
解决方法：在模型载入的时候添加类似如下语句：

```python
self.model = keras.models.load_model(path,custom_objects= {'Denoising_layer':Denoising_layer} )
```
或者，使用load_weights()函数去载入模型参数。







