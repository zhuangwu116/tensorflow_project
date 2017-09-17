#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 09:05:43 2017

@author: zhuangwu
"""
import tensorflow as tf
a=tf.constant([1.0,2.0],name='a')
b=tf.constant([2.0,3.0],name='b')
result = a + b
print(result)
print(a.graph is tf.get_default_graph())