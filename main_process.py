# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 13:18:05 2017

@author: Weiyu_Lee
"""

from model import MODEL

import tensorflow as tf
import argparse
import pprint
import os
import config

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="example",help="Configuration name")
args = parser.parse_args()

conf = config.config(args.config).config["train"]

def main(_):

    if not os.path.exists(conf["checkpoint_dir"]):
        os.makedirs(conf["checkpoint_dir"])
    if not os.path.exists(conf["output_dir"]):
        os.makedirs(conf["output_dir"])

    with tf.Session() as sess:
        srcnn = MODEL(sess, 
                      mode=conf["mode"],
                      epoch=conf["epoch"],
                      batch_size=conf["batch_size"],
                      image_size=conf["image_size"], 
                      label_size=conf["label_size"], 
                      learning_rate=conf["learning_rate"],
                      color_dim=conf["color_dim"], 
                      scale=conf["scale"],
                      train_extract_stride=conf["train_extract_stride"],
                      test_extract_stride=conf["test_extract_stride"],
                      checkpoint_dir=conf["checkpoint_dir"],
                      log_dir=conf["log_dir"],
                      output_dir=conf["output_dir"],
                      train_dir=conf["train_dir"],
                      test_dir=conf["test_dir"],
                      h5_dir=conf["h5_dir"],
                      train_h5_name=conf["train_h5_name"],
                      test_h5_name=conf["test_h5_name"],
                      ckpt_name=conf["ckpt_name"],
                      is_train=conf["is_train"],
                      model_ticket=conf["model_ticket"],
                      curr_epoch=conf["curr_epoch"])

        if conf["is_train"]:
            srcnn.train()
    
if __name__ == '__main__':
  tf.app.run()