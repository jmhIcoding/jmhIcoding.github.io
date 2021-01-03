---
layout: post
title: "Keras 给定输入数据，获取LOSS关于输入的梯度"
subtitle: 子标题
author: 明昊
header-style: text
category: Keras
tags:
  - Keras
  - 对抗机器学习
  - 深度学习
---

# 需求
论文需要使用对抗训练 adversarial training，里面需要拿到目标函数的loss关于输入样本的梯度。
# 方法：使用Keras.backbend的function函数

要点：
- 想办法把输入feed给模型
- Keras.backend.function就负责将实际的数据feed给相应的placeholder，接着再执行所需的计算子图流程。

Keras.backend.function的原型定义：

```python
keras.backend.function(inputs, outputs, updates=None)
inputs: 占位符张量列表。
outputs: 输出张量列表。
updates: 更新操作列表。
**kwargs: 需要传递给 tf.Session.run 的参数。
```
其中inputs指示了，为了获取运行所需计算图的结果，将需要给模型里面的那些placeholder喂数据。

inputs的实参应该是一个list或tuple 对象。一般来说，为了获取由Keras顺序模型（Sequential ）搭建的分类模型model的梯度，至少需要给如下placeholder 喂数据：
1. 模型的输入：model.inputs
2. 模型的样本的权重：model.sample_weights
3. 模型的正确lable: model.targets
4. Keras的工作模式：K.learning_phase()

因此，对于模型model，inputs的实参应该是类似于：

```python
input_tensors = [
                model.inputs[0],
                model.sample_weights[0],
                model.targets[0],
                K.learning_phase(),
            ]
```
而outputs实参，就是我们实际要执行的额外计算图，在获取loss关于输入的梯度时应该为：

```python
grads = K.gradients(model.total_loss,model.inputs)
```

把这两部分结合起来，就可以知道怎么写啦：
**注意session和graph的指定！ 否则会报找不到会话或计算图的错误。**
session和graph的定义：

```python
from tensorflow import Graph, Session
from keras import backend as K
        #1.创建空白的计算图和会话
        self.graph = Graph()
        self.session = Session(graph = self.graph)
        K.set_session(self.session)
```

示例如下：

```python
from keras import backbend as K
#model是编译好的模型，
with model.session.as_default():	#模型所在的session
     with model.graph.as_default(): #模型所在的graph 
            grads = K.gradients(model.total_loss,model.inputs)
            input_tensors = [
                model.inputs[0],
                model.sample_weights[0],
                model.targets[0],
                K.learning_phase(),
            ]
            get_gradients = K.function(inputs=input_tensors,
                                       outputs=grads)
            #最后，我们调用这个函数，然后往里面灌入数据。X_train和y_train就是我们准备好特征和标签数据。
            
            print(get_gradients([X_train,	#输入的numpy,里面是实际的数据
                                 np.ones(X_train.shape[0]),#各个样本的权值都是一样的
                                 y_train,	#输入的标签,它是个numpy,
                                 0			#默认为0，表示TEST
                                 ]))
```
示例输出：

```bash
[array([[[ 3.37047881e-04],
        [ 1.35833543e-04],
        [ 5.89624338e-04],
        ...,
        [-5.74612432e-06],
        [-6.32335559e-06],
        [-5.20324556e-06]],

       [[ 6.09082752e-04],
        [ 3.00324697e-04],
        [ 9.02293774e-04],
        ...,
        [-4.81713050e-06],
        [-5.49305832e-06],
        [-4.50423022e-06]],

       [[-1.64922003e-05],
        [-2.31035738e-05],
        [ 3.09628103e-06],
        ...,
        [-4.12806503e-06],
        [-4.30005048e-06],
        [-3.67417783e-06]],

       ...,

       [[ 2.34020845e-04],
        [-7.04993727e-06],
        [ 7.22564815e-04],
        ...,
        [ 7.31242017e-06],
        [ 8.50264314e-06],
        。。。。
```

当然这个方法也可以用于获取LOSS关于某些变量的梯度，此时只需要把K.gradients里面的第二个参数改成相应的变量就可以了。






