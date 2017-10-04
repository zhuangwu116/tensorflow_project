# -*- coding: utf-8 -*-
#!/usr/bin/env python3
#TensorFlow 实现进阶的卷积网络

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

sess = tf.InteractiveSession()


