#!/usr/bin/python
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : test_so.py
## Authors    : zhluo@aries
## Create Time: 2017-07-20:20:56:13
## Description:
## 
##

import tensorflow as tf
import numpy as np

lib_test = tf.load_op_library('sigmoid_diy.so')

a = np.array([[-2.2, 5.2, -1.36, 4, -2.65]])
with tf.Session():
    print(" a shape", a.shape)
    print lib_test.sigmoid_diy(a).eval()
