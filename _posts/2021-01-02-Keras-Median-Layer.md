---
layout: post
title: Keras获取中间层输出
subtitle: Keras获取中间层输出
author: 明昊
header-style: text
tags:
  - Keras
  - Keras高级用法
---

其中，self里面有三个对象：keras compile出来的图，self.model。
model对应的图 self.graph，model所在的会话 self.session。

```python
    def get_feature_map(self,X,layer_name='block1_conv1'):
        ''' 获取特定中间层的特征图

        :param X:           输入数据
        :param layer_name:  层的名字,str
        :return:
        '''
        with self.session.as_default():
            with self.graph.as_default():
                layer =   self.model.get_layer(layer_name)
                if layer != None:
                    value = layer.output
                    get_value= K.function(inputs=self.model.inputs,outputs=[value])
                    output = get_value([X])[0]
                    return  output
                else:
                    raise ValueError('Model{0} could not find layer named {1}.'.format(self.model_name,layer_name))
```







