# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Thu Sep  7 09:25:46 2017

@author: zhuangwu
"""
import tensorflow as tf

a = tf.constant([1.0, 2.0], name = 'a')

b = tf.constant([2.0, 3.0], name = 'b')

result = a + b

"""
通过a.graph 可以查看张量所属的计算图。因为没有特意指定，
所以这个计算图应该等于当前默认的计算图。所以下面这个操作
输出值为True。
"""

print(a.graph is tf.get_default_graph())

g1 = tf.Graph()

with g1.as_default():
    # 在计算图 g1 中定义变量 “v”，并设置初始值为0.

    v = tf.get_variable("v", initializer = tf.zeros_initializer(shape = [1]))

g2 = tf.Graph()

with g2.as_default():
    # 在计算图 g2 中定义变量 “v”，并设置初始值为1.

    v = tf.get_variable("v",initializer=tf.ones_initializer(shape=[1]))
    
# 在计算图 g1 中读取变量 “v“ 的取值.

with tf.Session(graph=g1) as sess:

    tf.initialize_all_variables().run()

    with tf.variable_scope("",reuse=True):
        
        # 在计算图 g1 中，变量 ”v“ 的取值应该为0，所以下面这行会输出[0.]。

        print(sess.run(tf.get_variable("v")))
        
# 在计算图 g2 中读取变量 ”v“ 的取值。

with tf.Session(graph=g2) as sess:

    tf.initialize_all_variables().run()
    
    with tf.variable_scope("",reuse=True):

        # 在计算图 g2 中，变量 ”v“ 的取值应该为1，所以下面这行会输出[1.]。

        print(sess.run(tf.get_variable("v")))


