---
layout: post
title: Keras 自定义层以及在Summary时自定义层的参数个数为0的问题
subtitle: Keras 自定义层
author: 明昊
header-style: text
tags:
  - Keras
  
---

今天 开开心心的实现了一个带降噪功能的残差层：
自定义的层里面，使用了其他Keras自带的layers。

```python
__author__ = 'dk'
'''
    定义降噪和增强模块
'''
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import add,Flatten,Conv1D, AveragePooling1D,Dot
import keras

class Denoising_layer(Layer):
    def __init__(self, filter_type = 'non_local_mean',**kwargs):
        '''
            :param type :  降噪类型, non-local-mean 滤波, 或者是 mean 均值滤波
        '''

        assert filter_type in ['non_local_mean','mean']
        self.filter_type = filter_type
        super(Denoising_layer,self).__init__(**kwargs)

    def build(self, input_shape):

        self.average_pooling = AveragePooling1D(strides=1,padding='same')
        self.conv1d = Conv1D(kernel_size=1,padding='same',filters=input_shape[-1])
        super(Denoising_layer,self).build(input_shape)

    def call(self, inputs, **kwargs):

        if self.filter_type == 'mean' :

            delta_x = self.average_pooling(inputs)

        if self.filter_type == 'non_local_mean' :
            delta_x = Dot(axes=(2,1))([inputs , K.permute_dimensions(inputs , (0,2,1))])
            delta_x = Dot(axes=(1))([delta_x,inputs])
        delta_x = self.conv1d(delta_x)

        return add([inputs,delta_x])
```
然后开开心心的加入到现有模型上，把模型summary一下，发现这个层居然参数为0，这还得了？？？

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201118194025824.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ptaDE5OTY=,size_16,color_FFFFFF,t_70#pic_center)
而且！ 最离谱的是，这种错误不会影响模型的训练。
但是当把模型保存后，再载入模型进行预测，那么模型就直接凉凉了。 
这个错误可以说是相当之隐蔽。


最后发现，居然要这么写：

```python
__author__ = 'dk'
'''
    定义降噪和增强模块
'''
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import add,Flatten,Conv1D, AveragePooling1D,Dot
import keras

class Denoising_layer(Layer):
    def __init__(self, filter_type = 'non_local_mean',**kwargs):
        '''
            :param type :  降噪类型, non-local-mean 滤波, 或者是 mean 均值滤波
        '''

        assert filter_type in ['non_local_mean','mean']
        self.filter_type = filter_type
        super(Denoising_layer,self).__init__(**kwargs)

    def build(self, input_shape):

        self.average_pooling = AveragePooling1D(strides=1,padding='same')
        self.conv1d = Conv1D(kernel_size=1,padding='same',filters=input_shape[-1])
        self.conv1d.build(input_shape) ##关键的两行！！！
        self._trainable_weights += self.conv1d._trainable_weights #关键的两行！！！
        super(Denoising_layer,self).build(input_shape)

    def call(self, inputs, **kwargs):

        if self.filter_type == 'mean' :

            delta_x = self.average_pooling(inputs)

        if self.filter_type == 'non_local_mean' :
            delta_x = Dot(axes=(2,1))([inputs , K.permute_dimensions(inputs , (0,2,1))])
            delta_x = Dot(axes=(1))([delta_x,inputs])
        delta_x = self.conv1d(delta_x)

        return add([inputs,delta_x])
```
需要在build之后，调用中间使用其他keras层的build函数，然后把其他层的参数加入进来才行。！！！
这是妙啊妙。


再看Summary:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201118194214254.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ptaDE5OTY=,size_16,color_FFFFFF,t_70#pic_center)
这样就恢复正常了！！！！

**总结：如果在自定义layers的时候用到了Keras自带的某些layer,需要在自定义层的build函数里面把所用到的层的参数加到自定义层的参数里面去。**






