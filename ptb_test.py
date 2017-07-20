#!/usr/bin/python
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : ptb_test.py
## Authors    : zhluo@aries
## Create Time: 2017-07-18:20:12:26
## Description:
## 
'''
There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

run test
Before run this script make sure you had run ptb_word_lm.py generate checkpoint.
$ python ptb_test.py --scale=small/meduim/large --dataset=simple-examples/data/ --weight=model/
'''

import numpy as np
import tensorflow as tf
import time
import reader

from ptb_word_lm import PTBModel
from ptb_word_lm import PTBInput
from ptb_word_lm import run_epoch
from ptb_word_lm import SmallConfig
from ptb_word_lm import MediumConfig
from ptb_word_lm import LargeConfig

flags = tf.flags

flags.DEFINE_string("scale", "small", "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("dataset", "simple-examples/data/", "Where the training/test data is stored.")
flags.DEFINE_string("weight", "model/", "Pretrained weights")

FLAGS = flags.FLAGS

def get_config():
  if FLAGS.scale == "small":
    return SmallConfig()
  elif FLAGS.scale == "medium":
    return MediumConfig()
  elif FLAGS.scale == "large":
    return LargeConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)

def main(_):
    start_time = time.time()
    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1
    
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                 config.init_scale)
    raw_data = reader.ptb_raw_data(FLAGS.dataset)
    train_data, valid_data, test_data, _ = raw_data

    # create graph
    with tf.name_scope("Test"):
        test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            mtest = PTBModel(is_training=False, config=eval_config, input_=test_input)
    
    # constraint GPU memory use
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    # auto load checkpoint form specific path
    sv = tf.train.Supervisor(logdir=FLAGS.weight)
    with sv.managed_session() as session:
        test_perplexity = run_epoch(session, mtest)
        print("Test Perplexity: %.3f, elapsed time: %.f" % test_perplexity, time.time()-start_time)
        
if __name__ == "__main__":
    tf.app.run()
