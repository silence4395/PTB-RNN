#!/usr/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : ptb_test.py
## Authors    : zhluo@aries
## Create Time: 2017-07-18:20:12:26
## Description:
## 
##

import numpy as np
import tensorflow as tf

flags = tf.flags

flags.DEFINE_string("graph", None, "Saved graph net")
flags.DEFINE_string("weight", None, "Pretrained weights")
flags.DEFINE_string("use_fp16", False, "Using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32

def main(_):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
        
    
    
if __main__=="__main__":
    tf.app.run()
