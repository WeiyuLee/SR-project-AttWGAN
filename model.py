# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 13:20:02 2017

@author: Weiyu_Lee
"""

import os
import sys
sys.path.append('./utility')

import tensorflow as tf
from tqdm import tqdm
import numpy as np

import model_zoo
import scipy.misc as misc
import random

import time

from utils import (
  read_data, 
  batch_shuffle_rndc,
  batch_shuffle,
  log10,
  laplacian_filter
)

from tensorflow.examples.tutorials.mnist import input_data


class MODEL(object):
    def __init__(self, 
                 sess, 
                 mode=None,
                 epoch=10,
                 batch_size=128,
                 image_size=32,
                 label_size=20, 
                 learning_rate=1e-4,
                 color_dim=1, 
                 scale=4,
                 train_extract_stride=14,
                 test_extract_stride=20,
                 checkpoint_dir=None, 
                 log_dir=None,
                 output_dir=None,
                 train_dir=None,
                 test_dir=None,
                 h5_dir=None,
                 train_h5_name=None,
                 test_h5_name=None,
                 ckpt_name=None,
                 is_train=True,
                 model_ticket=None,
                 curr_epoch=0):                 
        """
        Initial function
          
        Args:
            image_size: training or testing input image size. 
                        (if scale=3, image size is [33x33].)
            label_size: label image size. 
                        (if scale=3, image size is [21x21].)
            batch_size: batch size
            color_dim: color dimension number. (only Y channel, color_dim=1)
            checkpoint_dir: checkpoint directory
            output_dir: output directory
        """  
        
        self.sess = sess
        
        self.mode = mode

        self.epoch = epoch

        self.batch_size = batch_size
        self.image_size = image_size
        self.label_size = label_size      

        self.learning_rate = learning_rate
        self.color_dim = color_dim
        self.is_grayscale = (color_dim == 1)        
        self.scale = scale
    
        self.train_extract_stride = train_extract_stride
        self.test_extract_stride = test_extract_stride
    
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.output_dir = output_dir
        
        self.train_dir = train_dir
        self.test_dir = test_dir
        
        self.h5_dir = h5_dir
        self.train_h5_name = train_h5_name
        self.test_h5_name = test_h5_name
        
        self.ckpt_name = ckpt_name
        
        self.is_train = is_train      
        
        self.model_ticket = model_ticket

        self.model_list = ["RCAN_WGAN_att_on_dis_RG3_RCAB20", "RCAN_WGAN_att_on_dis", "EDSR_WGAN_att_on_dis_RCAN", "EDSR_RaGAN", "EDSR_WGAN", "EDSR_WGAN_att", "EDSR_WGAN_MNIST", "EDSR_RaGAN_MNIST"]

        
        self.curr_epoch = curr_epoch
        
        self.build_model()        
    
    def build_model(self):### 

        print(self.model_ticket)
        print(self.model_list)             
        if self.model_ticket not in self.model_list:
            print("sorry, wrong ticket!")
            return 0
        
        else:
            fn = getattr(self, "build_" + self.model_ticket)
            model = fn()
            return model    
        
    def train(self):
        if self.model_ticket not in self.model_list:
            print("sorry, wrong ticket!")
            return 0
        
        else:
            fn = getattr(self, "train_" + self.model_ticket)
            function = fn()
            return function                 
                
    def save_ckpt(self, checkpoint_dir, ckpt_name, step):
        """
        Save the checkpoint. 
        According to the scale, use different folder to save the models.
        """          
        
        print(" [*] Saving checkpoints...step: [{}]".format(step))
        model_name = ckpt_name
        
        if ckpt_name == "":
            model_dir = "%s_%s_%s" % ("srcnn", "scale", self.scale)
        else:
            model_dir = ckpt_name
        
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def save_best_ckpt(self, checkpoint_dir, ckpt_name, loss, step):
        """
        Save the checkpoint. 
        According to the scale, use different folder to save the models.
        """          
        
        print(" [*] Saving best checkpoints...step: [{}]\n".format(step))
        model_name = ckpt_name + "_{}".format(loss)
        
        if ckpt_name == "":
            model_dir = "%s_%s_%s" % ("srcnn", "scale", self.scale)
        else:
            model_dir = ckpt_name
        
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
        checkpoint_dir = os.path.join(checkpoint_dir, "best_performance")
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        self.best_saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)     

    def load_ckpt(self, checkpoint_dir, ckpt_name=""):
        """
        Load the checkpoint. 
        According to the scale, read different folder to load the models.
        """     
        
        print(" [*] Reading checkpoints...")
        if ckpt_name == "":
            model_dir = "%s_%s_%s" % ("srcnn", "scale", self.scale)
        else:
            model_dir = ckpt_name
            
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        
        print("Current checkpoints: [{}]".format(os.path.join(checkpoint_dir, ckpt_name)))
        
        if ckpt  and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        
            return True
        
        else:
            return False          


    def load_divk(self, dataset_path, mean=0, lrtype=None, type='train'):

        #dataset_path = "/home/ubuntu/dataset/SuperResolution/DIV2K/"
        print("=====================================================================================")

        if type == "train":

            sub_path = "DIV2K_train"

            lr_subpath = []
            lr_imgs = []

            if lrtype == 'bicubic':
                lr_subpath.append(sub_path + "_LR_bicubic/" + "X" + str(self.scale))
            elif lrtype == 'unknown':
                lr_subpath.append(sub_path + "_LR_unknown/" + "X" + str(self.scale))
            elif lrtype == 'all':
                lr_subpath.append(sub_path + "_LR_bicubic/" + "X" + str(self.scale))
                lr_subpath.append(sub_path + "_LR_unknown/" + "X" + str(self.scale))
            elif lrtype == 'baseline':
                lr_subpath.append(sub_path + "_LR_bicubic_baseline/" + "X" + str(self.scale))               
            else:
                print("lrtype error: [{}]".format(lrtype))
                return  0

            HR_path = os.path.join(dataset_path, sub_path + "_HR")
#            HR_path = os.path.join(dataset_path, sub_path + "_HR_baseline_enhance")
            hr_imgs = os.listdir(HR_path)
            hr_imgs = [os.path.join(HR_path, hr_imgs[i]) for i in range(len(hr_imgs))]

            for path_idx in range(len(lr_subpath)):
                LR_path = os.path.join(dataset_path, lr_subpath[path_idx])
                file_name = [os.path.basename(hr_imgs[i]) for i in range(len(hr_imgs))]
                lr_imgs.append([os.path.join(LR_path, file_name[i].split(".")[0] + 'x' + str(self.scale)+'.' + file_name[i].split(".")[1]) for i in range(len(hr_imgs))])
            
        elif type == "test":

            lr_imgs = []
            images = os.listdir(dataset_path)
            #for i in range(len(images)//2):
            lr_imgs.append([os.path.join(dataset_path, "img_"+str(i+1)+"_SRF_" + str(self.scale)+"_LR.png") for i in range(len(images)//2)])
            hr_imgs = ([os.path.join(dataset_path, "img_"+str(i+1)+"_SRF_"+ str(self.scale)+"_HR.png")  for i in range(len(images)//2)])

        elif type == "test_baseline":

            lr_imgs = []
            HR_path = dataset_path
            images = os.listdir(dataset_path)
            #for i in range(len(images)//2):
            lr_imgs.append([os.path.join(dataset_path, "img_"+str(i+1)+"_SRF_2_LR.png") for i in range(len(images)//2)])
            hr_imgs = ([os.path.join(dataset_path, "img_"+str(i+1)+"_SRF_2_HR.png")  for i in range(len(images)//2)])       
           
        hr_list = []
        lr_list = []
        lr_list2 = []

        for i in range(len(hr_imgs)):
        #for i in range(20):
           sys.stdout.write("Load data:{}/{}".format(i,len(hr_imgs))+'\r')
           sys.stdout.flush()          
          
           hr_list.append(misc.imread(hr_imgs[i]))            
           lr_list.append(misc.imread(lr_imgs[0][i]))
           if lrtype == 'all':
            lr_list2.append(misc.imread(lr_imgs[1][i]))
#           if lrtype == 'bicubic' and i > 32: break # for tuning hyperparameters
#           if lrtype == 'bicubic' and i > 8: break # for tuning hyperparameters

        print("[load_divk] type: [{}], lrtype: [{}]".format(type, lrtype))
        
        if type == "train":           
            print("[load_divk] HR path: [{}]".format(HR_path))
        else:
            print("[load_divk] HR path: [{}]".format(dataset_path))
        print("[load_divk] HR images number: [{}]".format(len(hr_list)))
        
        if type == "train":           
            for i in range(len(lr_subpath)):
                print("[load_divk] LR path[{}]: [{}]".format(i, lr_subpath[i]))
        else:
            print("[load_divk] LR path: [{}]".format(dataset_path))
        print("[load_divk] LR bicubic images number: [{}]".format(len(lr_list)))

        if lrtype == 'all':
            print("[load_divk] LR unknown images number: [{}]".format(len(lr_list2)))
            
        if lrtype == 'all':
            return list(zip(lr_list + lr_list2, hr_list + hr_list))
        else:
            return list(zip(lr_list, hr_list))

    def build_EDSR_WGAN_att_on_dis_RCAN(self):
        self.build_EDSR_WGAN()

    def train_EDSR_WGAN_att_on_dis_RCAN(self):
        self.train_EDSR_WGAN()

    def build_RCAN_WGAN_att_on_dis(self):
        self.build_RCAN_WGAN_att()

    def train_RCAN_WGAN_att_on_dis(self):
        self.train_RCAN_WGAN_att()

    def build_RCAN_WGAN_att_on_dis_RG3_RCAB20(self):
        self.build_RCAN_WGAN_att()

    def train_RCAN_WGAN_att_on_dis_RG3_RCAB20(self):
        self.train_RCAN_WGAN_att()


    def build_EDSR_RaGAN(self):###
        """
        Build SRCNN model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.image_size*2, self.image_size*2, self.color_dim], name='images')
        self.image_target = tf.placeholder(tf.float32, [None, self.image_size*2, self.image_size*2, self.color_dim], name='labels')

        self.curr_batch_size = tf.placeholder(tf.int32, shape=[])

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        self.image_input = self.input/255.
        self.target = target = self.image_target/255.
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.image_input, self.dropout, self.is_train, self.model_ticket)
        e = 1e-14
        
        ### Build model       
        gen_f = mz.build_model({"d_inputs":None, "scale":self.scale, "feature_size" :64, "reuse":False, "is_training":True, "net":"Gen"})
        dis_t, lp_t = mz.build_model({"d_inputs":self.target, "scale":self.scale, "feature_size" :64, "reuse":False, "is_training":True, "net":"Dis", "d_model":"PatchWGAN_GP"})
        dis_f, lp_f = mz.build_model({"d_inputs":gen_f, "scale":self.scale, "feature_size" :64, "reuse":True, "is_training":True, "net":"Dis", "d_model":"PatchWGAN_GP"})           

        """
        # RaGAN
        avg_fake_logit = tf.reduce_mean(dis_f)
        avg_real_logit = tf.reduce_mean(dis_t)

        D_r_tilde = tf.nn.sigmoid(dis_t - avg_fake_logit)
        D_f_tilde = tf.nn.sigmoid(dis_f - avg_real_logit)        

        self.d_loss = - tf.reduce_mean(tf.log(D_r_tilde + epsilon)) - tf.reduce_mean(tf.log(1 - D_f_tilde + epsilon))
        self.g_loss = - tf.reduce_mean(tf.log(D_f_tilde + epsilon)) - tf.reduce_mean(tf.log(1 - D_r_tilde + epsilon))
        
        self.g_l2loss = tf.reduce_mean(tf.pow(target-gen_f, 2))
        """

        #### WGAN-GP ####
        avg_fake_logit = tf.reduce_mean(dis_f)
        avg_real_logit = tf.reduce_mean(dis_t)

        D_r_tilde = tf.nn.sigmoid(dis_t - avg_fake_logit)
        D_f_tilde = tf.nn.sigmoid(dis_f - avg_real_logit)          
        
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.curr_batch_size, 1, 1, 1], 0.0, 1.0)
        x_hat = epsilon * self.target + (1. - epsilon) * (gen_f)
        d_hat = mz.build_model({"d_inputs":x_hat, "scale":self.scale, "feature_size" :64, "reuse":True, "is_training":True, "net":"Dis", "d_model":"PatchWGAN_GP"})
        
        d_gp = tf.gradients(d_hat, [x_hat])[0]
        d_gp = tf.sqrt(tf.reduce_sum(tf.square(d_gp), axis=[1,2,3]))
        d_gp = tf.reduce_mean((d_gp - 1.0)**2) * 10

#        self.disc_ture_loss = disc_ture_loss = tf.reduce_mean(dis_t)
#        self.disc_fake_loss = disc_fake_loss = tf.reduce_mean(dis_f)
        
        perceptual_loss = tf.pow(lp_t - lp_f, 2)

        reconstucted_weight = 10.0

        lp_weight = 0.25 

        self.d_loss = - tf.reduce_mean(tf.log(D_r_tilde + e)) - tf.reduce_mean(tf.log(1 - D_f_tilde + e)) + d_gp

        self.g_l2loss = tf.reduce_mean(tf.pow(target-gen_f, 2))

        self.g_loss = - tf.reduce_mean(tf.log(D_f_tilde + e)) - tf.reduce_mean(tf.log(1 - D_r_tilde + e)) + reconstucted_weight*self.g_l2loss + lp_weight*perceptual_loss

        train_variables = tf.trainable_variables()
        generator_variables = [v for v in train_variables if v.name.startswith("EDSR_gen")]
        discriminator_variables = [v for v in train_variables if v.name.startswith("EDSR_dis")]
        self.train_d = tf.train.AdamOptimizer(self.lr, beta1=0.9, beta2=0.999).minimize(self.d_loss, var_list=discriminator_variables)
        self.train_g = tf.train.AdamOptimizer(self.lr, beta1=0.9, beta2=0.999).minimize(self.g_loss, var_list=generator_variables)

        self.g_output = gen_f
        
        mse = tf.reduce_mean(tf.squared_difference(target*255.,self.g_output*255.))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/mse
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)

        mse_ref = tf.reduce_mean(tf.squared_difference(target*255.,self.image_input*255.))    
        PSNR_ref = tf.constant(255**2,dtype=tf.float32)/mse_ref
        PSNR_ref = tf.constant(10,dtype=tf.float32)*log10(PSNR_ref)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("l2_loss", self.g_l2loss, collections=['train'])
            tf.summary.scalar("d_loss", self.d_loss, collections=['train'])

            tf.summary.scalar("MSE", mse, collections=['train'])
            tf.summary.scalar("PSNR",PSNR, collections=['train'])
            tf.summary.image("input_image",self.input , collections=['train'])
            tf.summary.image("target_image",target*255, collections=['train'])
            tf.summary.image("output_image",self.g_output*255, collections=['train'])

            tf.summary.histogram("d_false", dis_f, collections=['train']) 
            tf.summary.histogram("d_true", dis_t, collections=['train']) 

            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):

            tf.summary.scalar("l2_loss", self.g_l2loss, collections=['test'])
            tf.summary.scalar("d_loss", self.d_loss, collections=['test'])
            tf.summary.scalar("g_loss", self.g_loss, collections=['test'])
                       
            tf.summary.scalar("MSE", mse, collections=['test'])
            tf.summary.scalar("PSNR",PSNR, collections=['test'])
            tf.summary.scalar("PSNR_ref",PSNR_ref, collections=['test'])
            tf.summary.image("input_image",self.input , collections=['test'])
            tf.summary.image("target_image",target*255, collections=['test'])
            tf.summary.image("output_image",self.g_output*255, collections=['test'])             
        
            self.merged_summary_test = tf.summary.merge_all('test')                    
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()       
        
    def train_EDSR_RaGAN(self):
        """
        Training process.
        """     
        print("Training...")

        # Define dataset path
        #96X96
        test_dataset = self.load_divk("/home/wei/ML/dataset/SuperResolution/Set5/pretrain_Set5/validation/X{}/".format(self.scale), type="test_baseline")
#        dataset = self.load_divk("/home/wei/ML/dataset/SuperResolution/DIV2K/pretrain_DIV2K/", lrtype='all', type='train')
        dataset = self.load_divk("/home/wei/ML/dataset/SuperResolution/DIV2K/pretrain_DIV2K/", lrtype='bicubic', type='train') 

        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")       
        
        # Define iteration counter, learning rate...
        if(self.batch_size >= 16):
            save_ep = 5
            lr_ep = 4000
            lr_ep_2 = 8000            
        else:
            save_ep = 200       
            lr_ep = 250000
            lr_ep_2 = 500000            

        itera_counter = 0
        
        train_data, train_label  = zip(*dataset)

        learning_rate = self.learning_rate
        if self.curr_epoch >= lr_ep and self.curr_epoch < lr_ep_2:
            learning_rate = self.learning_rate / 2 # only decay once
        elif self.curr_epoch >= lr_ep and self.curr_epoch >= lr_ep_2:
            learning_rate = self.learning_rate / 4 # decay twice
            
        print("Current learning rate: [{}]".format(learning_rate))
        
        best_loss = 100
        
        epoch_pbar = tqdm(range(self.curr_epoch, self.epoch))
        for ep in epoch_pbar:            
            
            random.shuffle(dataset) 
            train_data, train_label  = zip(*dataset)
            test_data, test_label  = zip(*test_dataset)

            epoch_pbar.set_description("Epoch: [%2d], lr:%f" % ((ep+1), learning_rate))
            epoch_pbar.refresh()

            if(self.batch_size >= 16):
                batch_pbar = tqdm(range(0, len(train_data)//self.batch_size), desc="Batch: [0]")       
            else:
                batch_pbar = tqdm(range(0, 1), desc="Batch: [0]") # for tuning hyperparameters            

            if (ep % lr_ep == 0 or ep % lr_ep_2 == 0) and ep != 0 and ep <= lr_ep_2:
                learning_rate = learning_rate/2
                print("[Learning Rate Decay]")
                print("Current learning rate: [{}]".format(learning_rate))

            for idx in batch_pbar:                
                
                batch_pbar.set_description("Batch: [%2d]" % ((idx+1)))
                itera_counter += 1
                batch_index = idx*self.batch_size 

                batch_images, batch_labels = batch_shuffle_rndc(train_data, train_label, 1, self.image_size*2, batch_index, self.batch_size)                

                _ = self.sess.run([self.train_g], 
                                                   feed_dict={
                                                               self.input: batch_images,
                                                               self.image_target: batch_labels,
                                                               self.dropout: 1.,
                                                               self.lr:learning_rate 
                                                             })
                                                 
                for d_iter in range(0, 5):
                    _, d_loss, g_loss \
                    = self.sess.run([self.train_d, self.d_loss, self.g_loss],
                                                                             feed_dict={   
                                                                                         self.input: batch_images,
                                                                                         self.image_target: batch_labels,
                                                                                         self.curr_batch_size: self.batch_size,
                                                                                         self.dropout: 1.,
                                                                                         self.lr:learning_rate
                                                                                       })
                   
            if ep % save_ep == 0 and ep != 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, itera_counter)
                
                train_sum = self.sess.run(self.merged_summary_train, 
                                                                    feed_dict={
                                                                                self.input: batch_images,
                                                                                self.image_target: batch_labels,
                                                                                self.curr_batch_size: self.batch_size,
                                                                                self.dropout: 1.
                                                                              })
                
                test_sum, g_output, loss = self.sess.run([self.merged_summary_test, self.g_output, self.g_l2loss],
                                                                     feed_dict={
                                                                                 self.input: test_data,  
                                                                                 self.image_target: test_label,
                                                                                 self.curr_batch_size: len(test_label),
                                                                                 self.dropout: 1.,
                                                                               })
                                                                                                   
                
                                                                     
                print("Epoch: [{}]".format((ep+1)))       
                
                if loss < best_loss:
                    best_loss = loss
#                    self.save_best_ckpt(self.checkpoint_dir, self.ckpt_name, best_loss, itera_counter)
                    print("Current Loss: [{}]\n".format(best_loss))
                
                summary_writer.add_summary(train_sum, ep)
                summary_writer.add_summary(test_sum, ep)             

    def build_EDSR_WGAN(self):###
        """
        Build SRCNN model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.image_size*2, self.image_size*2, self.color_dim], name='images')
        self.image_target = tf.placeholder(tf.float32, [None, self.image_size*2, self.image_size*2, self.color_dim], name='labels')

        self.curr_batch_size = tf.placeholder(tf.int32, shape=[])

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        self.image_input = self.input/255.
        self.target = target = self.image_target/255.
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.image_input, self.dropout, self.is_train, self.model_ticket)
        
        ### Build model       
        gen_f = mz.build_model({"d_inputs":None, "scale":self.scale, "feature_size" :64, "reuse":False, "is_training":True, "net":"Gen"})

        dis_t, lp_t = mz.build_model({"d_inputs":self.target, "scale":self.scale, "feature_size" :64, "reuse":False, "is_training":True, "net":"Dis", "d_model":"PatchWGAN_GP"})
        dis_f, lp_f = mz.build_model({"d_inputs":gen_f, "scale":self.scale, "feature_size" :64, "reuse":True, "is_training":True, "net":"Dis", "d_model":"PatchWGAN_GP"})           

        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.curr_batch_size, 1, 1, 1], 0.0, 1.0)
        x_hat = epsilon * self.target + (1. - epsilon) * (gen_f)
        d_hat = mz.build_model({"d_inputs":x_hat, "scale":self.scale, "feature_size" :64, "reuse":True, "is_training":True, "net":"Dis", "d_model":"PatchWGAN_GP"})
        
        d_gp = tf.gradients(d_hat, [x_hat])[0]
        d_gp = tf.sqrt(tf.reduce_sum(tf.square(d_gp), axis=[1,2,3]))
        d_gp = tf.reduce_mean((d_gp - 1.0)**2) * 10

        self.disc_ture_loss = disc_ture_loss = tf.reduce_mean(dis_t)
        self.disc_fake_loss = disc_fake_loss = tf.reduce_mean(dis_f)
        
        perceptual_loss = tf.pow(lp_t - lp_f, 2)

        reconstucted_weight = 10.0

        lp_weight = 0.25 

        self.d_loss =   (disc_fake_loss - disc_ture_loss) + d_gp

        self.g_l2loss = tf.reduce_mean(tf.pow(target-gen_f, 2))

        self.g_loss = -1.0*disc_fake_loss + reconstucted_weight*self.g_l2loss + lp_weight*perceptual_loss

        train_variables = tf.trainable_variables()
        generator_variables = [v for v in train_variables if v.name.startswith("EDSR_gen")]
        discriminator_variables = [v for v in train_variables if v.name.startswith("EDSR_dis")]
        self.train_d = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=discriminator_variables)
        self.train_g = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=generator_variables)

        self.g_output = gen_f
        
        mse = tf.reduce_mean(tf.squared_difference(target*255.,gen_f*255.))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/mse
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)

        mse_ref = tf.reduce_mean(tf.squared_difference(target*255.,self.image_input*255.))    
        PSNR_ref = tf.constant(255**2,dtype=tf.float32)/mse_ref
        PSNR_ref = tf.constant(10,dtype=tf.float32)*log10(PSNR_ref)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("l2_loss", self.g_l2loss, collections=['train'])
            tf.summary.scalar("d_loss", self.d_loss, collections=['train'])
            tf.summary.scalar("d_true_loss", disc_ture_loss, collections=['train'])
            tf.summary.scalar("d_fake_loss", disc_fake_loss, collections=['train'])
            tf.summary.scalar("d_lp", perceptual_loss, collections=['train']) # for tuning

            tf.summary.scalar("MSE", mse, collections=['train'])
            tf.summary.scalar("PSNR",PSNR, collections=['train'])
            tf.summary.image("input_image",self.input , collections=['train'])
            tf.summary.image("target_image",target*255, collections=['train'])
            tf.summary.image("output_image",gen_f*255, collections=['train'])

            tf.summary.histogram("d_false", dis_f, collections=['train']) 
            tf.summary.histogram("d_true", dis_t, collections=['train']) 

            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):

            tf.summary.scalar("l2_loss", self.g_l2loss, collections=['test'])
            tf.summary.scalar("d_loss", self.d_loss, collections=['test'])
            tf.summary.scalar("g_loss", disc_fake_loss, collections=['test'])
            
            tf.summary.scalar("d_lp", perceptual_loss, collections=['test']) # for tuning
            
            tf.summary.scalar("MSE", mse, collections=['test'])
            tf.summary.scalar("PSNR",PSNR, collections=['test'])
            tf.summary.scalar("PSNR_ref",PSNR_ref, collections=['test'])
            tf.summary.image("input_image",self.input , collections=['test'])
            tf.summary.image("target_image",target*255, collections=['test'])
            tf.summary.image("output_image",gen_f*255, collections=['test'])             
        
            self.merged_summary_test = tf.summary.merge_all('test')                    
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()       
        
    def train_EDSR_WGAN(self):
        """
        Training process.
        """     
        print("Training...")

        # Define dataset path
        #96X96

        
        test_dataset = self.load_divk(self.test_dir+"/X{}".format(self.scale), type='test_baseline')
        dataset = self.load_divk(self.train_dir, lrtype='bicubic', type='train')
        #test_dataset = self.load_divk("/home/wei/ML/dataset/SuperResolution/Set5/pretrain_Set5/validation/X{}/".format(self.scale), type="test_baseline")
        #dataset = self.load_divk("/home/wei/ML/dataset/SuperResolution/DIV2K/pretrain_DIV2K/", lrtype='all', type='train')
        #dataset = self.load_divk("/home/wei/ML/dataset/SuperResolution/DIV2K/pretrain_DIV2K/", lrtype='bicubic', type='train') 

        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")       
        
        # Define iteration counter, learning rate...
        if(self.batch_size >= 16):
            save_ep = 5
            lr_ep = 4000
            lr_ep_2 = 8000            
        else:
            save_ep = 200       
            lr_ep = 250000
            lr_ep_2 = 500000            

        itera_counter = 0
        
        train_data, train_label  = zip(*dataset)

        learning_rate = self.learning_rate
        if self.curr_epoch >= lr_ep and self.curr_epoch < lr_ep_2:
            learning_rate = self.learning_rate / 2 # only decay once
        elif self.curr_epoch >= lr_ep and self.curr_epoch >= lr_ep_2:
            learning_rate = self.learning_rate / 4 # decay twice
            
        print("Current learning rate: [{}]".format(learning_rate))
        
        best_loss = 100
        
        epoch_pbar = tqdm(range(self.curr_epoch, self.epoch))
        for ep in epoch_pbar:            
            
            random.shuffle(dataset) 
            train_data, train_label  = zip(*dataset)
            test_data, test_label  = zip(*test_dataset)

            epoch_pbar.set_description("Epoch: [%2d], lr:%f" % ((ep+1), learning_rate))
            epoch_pbar.refresh()

            if(self.batch_size >= 16):
                batch_pbar = tqdm(range(0, len(train_data)//self.batch_size), desc="Batch: [0]")       
            else:
                batch_pbar = tqdm(range(0, 1), desc="Batch: [0]") # for tuning hyperparameters            

            if (ep % lr_ep == 0 or ep % lr_ep_2 == 0) and ep != 0 and ep <= lr_ep_2:
                learning_rate = learning_rate/2
                print("[Learning Rate Decay]")
                print("Current learning rate: [{}]".format(learning_rate))

            for idx in batch_pbar:                
                
                batch_pbar.set_description("Batch: [%2d]" % ((idx+1)))
                itera_counter += 1
                batch_index = idx*self.batch_size 

                batch_images, batch_labels = batch_shuffle_rndc(train_data, train_label, 1, self.image_size*2, batch_index, self.batch_size)                

                _ = self.sess.run([self.train_g], 
                                                   feed_dict={
                                                               self.input: batch_images,
                                                               self.image_target: batch_labels,
                                                               self.dropout: 1.,
                                                               self.lr:learning_rate 
                                                             })
                                                 
                for d_iter in range(0, 5):
                    _, d_loss, g_loss \
                    = self.sess.run([self.train_d, self.d_loss, self.g_loss],
                                                                             feed_dict={   
                                                                                         self.input: batch_images,
                                                                                         self.image_target: batch_labels,
                                                                                         self.curr_batch_size: self.batch_size,
                                                                                         self.dropout: 1.,
                                                                                         self.lr:learning_rate
                                                                                       })
                   
            if ep % save_ep == 0 and ep != 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, itera_counter)
                
                train_sum = self.sess.run(self.merged_summary_train, 
                                                                    feed_dict={
                                                                                self.input: batch_images,
                                                                                self.image_target: batch_labels,
                                                                                self.curr_batch_size: self.batch_size,
                                                                                self.dropout: 1.
                                                                              })
                
                test_sum, g_output, loss = self.sess.run([self.merged_summary_test, self.g_output, self.g_l2loss],
                                                                     feed_dict={
                                                                                 self.input: test_data,  
                                                                                 self.image_target: test_label,
                                                                                 self.curr_batch_size: len(test_label),
                                                                                 self.dropout: 1.,
                                                                               })
                                                                                                   
                
                                                                     
                print("Epoch: [{}]".format((ep+1)))       
                
                if loss < best_loss:
                    best_loss = loss
#                    self.save_best_ckpt(self.checkpoint_dir, self.ckpt_name, best_loss, itera_counter)
                    print("Current Loss: [{}]\n".format(best_loss))
                
                summary_writer.add_summary(train_sum, ep)
                summary_writer.add_summary(test_sum, ep)             

    def build_EDSR_WGAN_att(self):###
        """
        Build SRCNN model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.image_size*2, self.image_size*2, self.color_dim], name='images')
        self.image_target = tf.placeholder(tf.float32, [None, self.image_size*2, self.image_size*2, self.color_dim], name='labels')

        self.curr_batch_size = tf.placeholder(tf.int32, shape=[])

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        self.image_input = self.input/255.
        self.target = target = self.image_target/255.
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.image_input, self.dropout, self.is_train, self.model_ticket)
        
        ### Build model       
        gen_f = mz.build_model({"d_inputs":None, "scale":self.scale, "feature_size" :64, "reuse":False, "is_training":True, "net":"Gen"})

        dis_t, lp_t = mz.build_model({"d_inputs":self.target, "scale":self.scale, "feature_size" :64, "reuse":False, "is_training":True, "net":"Dis", "d_model":"PatchWGAN_GP"})
        dis_f, lp_f = mz.build_model({"d_inputs":gen_f, "scale":self.scale, "feature_size" :64, "reuse":True, "is_training":True, "net":"Dis", "d_model":"PatchWGAN_GP"})           

        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.curr_batch_size, 1, 1, 1], 0.0, 1.0)
        x_hat = epsilon * self.target + (1. - epsilon) * (gen_f)
        d_hat = mz.build_model({"d_inputs":x_hat, "scale":self.scale, "feature_size" :64, "reuse":True, "is_training":True, "net":"Dis", "d_model":"PatchWGAN_GP"})
        
        d_gp = tf.gradients(d_hat, [x_hat])[0]
        d_gp = tf.sqrt(tf.reduce_sum(tf.square(d_gp), axis=[1,2,3]))
        d_gp = tf.reduce_mean((d_gp - 1.0)**2) * 10

        self.disc_ture_loss = disc_ture_loss = tf.reduce_mean(dis_t)
        self.disc_fake_loss = disc_fake_loss = tf.reduce_mean(dis_f)
        
        perceptual_loss = tf.pow(lp_t - lp_f, 2)

        reconstucted_weight = 10.0

        lp_weight = 0.25 

        self.d_loss =   (disc_fake_loss - disc_ture_loss) + d_gp

        self.g_l2loss = tf.reduce_mean(tf.pow(target-gen_f, 2))

        self.g_loss = -1.0*disc_fake_loss + reconstucted_weight*self.g_l2loss + lp_weight*perceptual_loss

        train_variables = tf.trainable_variables()
        generator_variables = [v for v in train_variables if v.name.startswith("EDSR_gen")]
        discriminator_variables = [v for v in train_variables if v.name.startswith("EDSR_dis")]
        self.train_d = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=discriminator_variables)
        self.train_g = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=generator_variables)

        self.g_output = gen_f
        
        mse = tf.reduce_mean(tf.squared_difference(target*255.,gen_f*255.))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/mse
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)

        mse_ref = tf.reduce_mean(tf.squared_difference(target*255.,self.image_input*255.))    
        PSNR_ref = tf.constant(255**2,dtype=tf.float32)/mse_ref
        PSNR_ref = tf.constant(10,dtype=tf.float32)*log10(PSNR_ref)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("l1_loss", self.g_l2loss, collections=['train'])
            tf.summary.scalar("d_loss", self.d_loss, collections=['train'])
            tf.summary.scalar("d_true_loss", disc_ture_loss, collections=['train'])
            tf.summary.scalar("d_fake_loss", disc_fake_loss, collections=['train'])
            tf.summary.scalar("d_lp", perceptual_loss, collections=['train']) # for tuning

            tf.summary.scalar("MSE", mse, collections=['train'])
            tf.summary.scalar("PSNR",PSNR, collections=['train'])
            tf.summary.image("input_image",self.input , collections=['train'])
            tf.summary.image("target_image",target*255, collections=['train'])
            tf.summary.image("output_image",gen_f*255, collections=['train'])

            tf.summary.histogram("d_false", dis_f, collections=['train']) 
            tf.summary.histogram("d_true", dis_t, collections=['train']) 

            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):

            tf.summary.scalar("loss", self.g_l2loss, collections=['test'])
            tf.summary.scalar("d_loss", self.d_loss, collections=['test'])
            tf.summary.scalar("g_loss", disc_fake_loss, collections=['test'])
            
            tf.summary.scalar("d_lp", perceptual_loss, collections=['test']) # for tuning
            
            tf.summary.scalar("MSE", mse, collections=['test'])
            tf.summary.scalar("PSNR",PSNR, collections=['test'])
            tf.summary.scalar("PSNR_ref",PSNR_ref, collections=['test'])
            tf.summary.image("input_image",self.input , collections=['test'])
            tf.summary.image("target_image",target*255, collections=['test'])
            tf.summary.image("output_image",gen_f*255, collections=['test'])             
        
            self.merged_summary_test = tf.summary.merge_all('test')                    
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()       
        
    def train_EDSR_WGAN_att(self):
        """
        Training process.
        """     
        print("Training...")

        # Define dataset path
        #96X96
        test_dataset = self.load_divk("/home/wei/ML/dataset/SuperResolution/Set5/pretrain_Set5/validation/X{}/".format(self.scale), type="test_baseline")
#        dataset = self.load_divk("/home/wei/ML/dataset/SuperResolution/DIV2K/pretrain_DIV2K/", lrtype='bicubic', type='train') 
        dataset = self.load_divk("/home/wei/ML/dataset/SuperResolution/DIV2K/pretrain_DIV2K_RCAN/", lrtype='bicubic', type='train') 
        
        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")       
        
        # Define iteration counter, learning rate...
        if(self.batch_size >= 16):
            save_ep = 5
            lr_ep = 4000
            lr_ep_2 = 8000            
        else:
            save_ep = 200       
            lr_ep = 250000
            lr_ep_2 = 500000            

        itera_counter = 0
        
        train_data, train_label  = zip(*dataset)

        learning_rate = self.learning_rate
        if self.curr_epoch >= lr_ep and self.curr_epoch < lr_ep_2:
            learning_rate = self.learning_rate / 2 # only decay once
        elif self.curr_epoch >= lr_ep and self.curr_epoch >= lr_ep_2:
            learning_rate = self.learning_rate / 4 # decay twice
            
        print("Current learning rate: [{}]".format(learning_rate))
        
        best_loss = 100
        
        epoch_pbar = tqdm(range(self.curr_epoch, self.epoch))
        for ep in epoch_pbar:            
            
            random.shuffle(dataset) 
            train_data, train_label  = zip(*dataset)
            test_data, test_label  = zip(*test_dataset)

            epoch_pbar.set_description("Epoch: [%2d], lr:%f" % ((ep+1), learning_rate))
            epoch_pbar.refresh()

            if(self.batch_size >= 16):
                batch_pbar = tqdm(range(0, len(train_data)//self.batch_size), desc="Batch: [0]")       
            else:
                batch_pbar = tqdm(range(0, 1), desc="Batch: [0]") # for tuning hyperparameters            

            if (ep % lr_ep == 0 or ep % lr_ep_2 == 0) and ep != 0 and ep <= lr_ep_2:
                learning_rate = learning_rate/2
                print("[Learning Rate Decay]")
                print("Current learning rate: [{}]".format(learning_rate))

            for idx in batch_pbar:                
                
                batch_pbar.set_description("Batch: [%2d]" % ((idx+1)))
                itera_counter += 1
                batch_index = idx*self.batch_size 

                batch_images, batch_labels = batch_shuffle_rndc(train_data, train_label, 1, self.image_size*2, batch_index, self.batch_size)                

                _ = self.sess.run([self.train_g], 
                                                   feed_dict={
                                                               self.input: batch_images,
                                                               self.image_target: batch_labels,
                                                               self.dropout: 1.,
                                                               self.lr:learning_rate 
                                                             })
                                                 
                for d_iter in range(0, 5):
                    _, d_loss, g_loss \
                    = self.sess.run([self.train_d, self.d_loss, self.g_loss],
                                                                             feed_dict={   
                                                                                         self.input: batch_images,
                                                                                         self.image_target: batch_labels,
                                                                                         self.curr_batch_size: self.batch_size,
                                                                                         self.dropout: 1.,
                                                                                         self.lr:learning_rate
                                                                                       })
                   
            if ep % save_ep == 0 and ep != 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, itera_counter)
                
                train_sum = self.sess.run(self.merged_summary_train, 
                                                                    feed_dict={
                                                                                self.input: batch_images,
                                                                                self.image_target: batch_labels,
                                                                                self.curr_batch_size: self.batch_size,
                                                                                self.dropout: 1.
                                                                              })
                
                test_sum, g_output, loss = self.sess.run([self.merged_summary_test, self.g_output, self.g_l2loss],
                                                                     feed_dict={
                                                                                 self.input: test_data,  
                                                                                 self.image_target: test_label,
                                                                                 self.curr_batch_size: len(test_label),
                                                                                 self.dropout: 1.,
                                                                               })
                                                                                                   
                
                                                                     
                print("Epoch: [{}]".format((ep+1)))       
                
                if loss < best_loss:
                    best_loss = loss
#                    self.save_best_ckpt(self.checkpoint_dir, self.ckpt_name, best_loss, itera_counter)
                    print("Current Loss: [{}]\n".format(best_loss))
                
                summary_writer.add_summary(train_sum, ep)
                summary_writer.add_summary(test_sum, ep) 

                
    def build_EDSR_WGAN_MNIST(self):###
        """
        Build SRCNN model
        """        
        # Define input and label images
        self.image_input = tf.random_normal([self.batch_size, 128], name='input')  
        self.image_target = tf.placeholder(tf.float32, [self.batch_size, self.image_size*self.image_size*self.color_dim], name='labels')

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.image_input, self.dropout, self.is_train, self.model_ticket)
        
        ### Build model       
        gen_f = mz.build_model({"d_inputs":None, "scale":self.scale, "feature_size" :64, "reuse":False, "is_training":True, "net":"Gen"})
        dis_t = mz.build_model({"d_inputs":self.image_target, "scale":self.scale, "feature_size" :64, "reuse":False, "is_training":True, "net":"Dis"})
        dis_f = mz.build_model({"d_inputs":gen_f, "scale":self.scale, "feature_size" :64, "reuse":True, "is_training":True, "net":"Dis"})

        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.batch_size, 1], 0.0, 1.0)
        x_hat = epsilon * self.image_target + (1. - epsilon) * (gen_f)
        d_hat = mz.build_model({"d_inputs":x_hat, "scale":self.scale,"feature_size" :64, "reuse":True, "is_training":True, "net":"Dis"})
        
        d_gp = tf.gradients(d_hat, [x_hat])[0]
        d_gp = tf.sqrt(tf.reduce_sum(tf.square(d_gp), reduction_indices=[1]))
        d_gp = tf.reduce_mean((d_gp - 1.0)**2) * 10

        self.disc_ture_loss = disc_ture_loss = tf.reduce_mean(dis_t)
        self.disc_fake_loss = disc_fake_loss = tf.reduce_mean(dis_f)
        
        # W(P_data, P_G) = min{ E_x~P_G[D(x)] - E_x~P_data[D(x)] + lamda*E_x~P_penalty[(D'(x)-1)^2] } ~ max{ V(G,D) }
        self.d_loss =   disc_fake_loss - disc_ture_loss + d_gp
        
        # Generator loss
        self.g_loss = -1.0*disc_fake_loss

        train_variables = tf.trainable_variables()
        generator_variables = [v for v in train_variables if v.name.startswith("EDSR_gen")]
        discriminator_variables = [v for v in train_variables if v.name.startswith("EDSR_dis")]
        self.train_d = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=discriminator_variables)
        self.train_g = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=generator_variables)
        
        self.g_output = gen_f
        
        with tf.name_scope('train_summary'):
            
            tf.summary.scalar("d_loss", self.d_loss, collections=['train'])
            tf.summary.scalar("g_loss", self.g_loss, collections=['train'])
            tf.summary.scalar("d_true_loss", disc_ture_loss, collections=['train'])
            tf.summary.scalar("d_fake_loss", disc_fake_loss, collections=['train'])
            tf.summary.scalar("grad_loss", d_gp, collections=['train'])
            tf.summary.scalar("dis_f_mean", tf.reduce_mean(dis_f), collections=['train'])
            tf.summary.scalar("dis_t_mean", tf.reduce_mean(dis_t), collections=['train'])
            tf.summary.image("target_image", tf.reshape(self.image_target, [self.batch_size, self.image_size, self.image_size, self.color_dim]), collections=['train'])
            tf.summary.image("output_image", tf.reshape(gen_f, [self.batch_size, self.image_size, self.image_size, self.color_dim]), collections=['train'])
    
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):

            tf.summary.scalar("d_loss", self.d_loss, collections=['test'])
            tf.summary.scalar("g_loss", self.g_loss, collections=['test'])
            tf.summary.image("target_image", tf.reshape(self.image_target, [self.batch_size, self.image_size, self.image_size, self.color_dim]), collections=['test'])
            tf.summary.image("output_image", tf.reshape(gen_f, [self.batch_size, self.image_size, self.image_size, self.color_dim]), collections=['test'])
        
            self.merged_summary_test = tf.summary.merge_all('test')                 
        
        self.saver = tf.train.Saver()

        
    def train_EDSR_WGAN_MNIST(self):
        """
        Training process.
        """     
        print("Training...")

        # Define dataset path
        mnist = input_data.read_data_sets("/home/wei/ML/dataset/MNIST_data/", one_hot=True)
        print("...")
        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")       
        
        # Define iteration counter, learning rate...
        itera_counter = 0
        learning_rate = self.learning_rate

        epoch_pbar = tqdm(range(self.curr_epoch, self.epoch))
        for ep in epoch_pbar:            
            # Run by batch images
            idxs = 1000

            epoch_pbar.set_description("Epoch: [%2d], lr:%f" % ((ep+1), learning_rate))
            epoch_pbar.refresh()
        
            batch_pbar = tqdm(range(0, idxs), desc="Batch: [0]")

            for idx in batch_pbar:                
                batch_pbar.set_description("Batch: [%2d]" % ((idx+1)))
                itera_counter += 1
                
                mnist_img, _ = mnist.train.next_batch(self.batch_size)
                                  
                for d_iter in range(0, 5):
                    _, d_loss, g_loss \
                    = self.sess.run([self.train_d, self.d_loss, self.g_loss],
                                                                             feed_dict={   
                                                                                         self.image_target: mnist_img,
                                                                                         self.dropout: 1.,
                                                                                         self.lr:learning_rate
                                                                                       })
                
                _ = self.sess.run([self.train_g], 
                                                   feed_dict={
                                                               self.image_target: mnist_img,
                                                               self.dropout: 1.,
                                                               self.lr:learning_rate 
                                                             })
            
            print("EP:[{}], d_loss = [{}], g_loss = [{}]\n".format(ep, d_loss, g_loss))
    
            if ep % 5 == 0 and ep != 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, itera_counter)
                
                train_sum = self.sess.run(self.merged_summary_train, 
                                                                    feed_dict={
                                                                                self.image_target: mnist_img,
                                                                                self.dropout: 1.
                                                                              })
                
                test_sum, g_output = self.sess.run([self.merged_summary_test, self.g_output],
                                                                     feed_dict={
                                                                                 self.image_target: mnist_img,
                                                                                 self.dropout: 1.,
                                                                               })
                                                                                                   
                
                                                                     
                print("Epoch: [{}]".format((ep+1)))       
                
                
                summary_writer.add_summary(train_sum, ep)
                summary_writer.add_summary(test_sum, ep)                

    def build_EDSR_RaGAN_MNIST(self):###
        """
        Build SRCNN model
        """        
        # Define input and label images
        self.image_input = tf.random_normal([self.batch_size, 128], name='input')  
        self.image_target = tf.placeholder(tf.float32, [self.batch_size, self.image_size*self.image_size*self.color_dim], name='labels')

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.image_input, self.dropout, self.is_train, self.model_ticket)
        epsilon = 1e-14
        
        ### Build model       
        gen_f = mz.build_model({"d_inputs":None, "scale":self.scale, "feature_size" :64, "reuse":False, "is_training":True, "net":"Gen"})
        dis_t = mz.build_model({"d_inputs":self.image_target, "scale":self.scale, "feature_size" :64, "reuse":False, "is_training":True, "net":"Dis"})
        dis_f = mz.build_model({"d_inputs":gen_f, "scale":self.scale, "feature_size" :64, "reuse":True, "is_training":True, "net":"Dis"})

        avg_fake_logit = tf.reduce_mean(dis_f)
        avg_real_logit = tf.reduce_mean(dis_t)

        D_r_tilde = tf.nn.sigmoid(dis_t - avg_fake_logit)
        D_f_tilde = tf.nn.sigmoid(dis_f - avg_real_logit)        

        self.d_loss = - tf.reduce_mean(tf.log(D_r_tilde + epsilon)) - tf.reduce_mean(tf.log(1 - D_f_tilde + epsilon))
        self.g_loss = - tf.reduce_mean(tf.log(D_f_tilde + epsilon)) - tf.reduce_mean(tf.log(1 - D_r_tilde + epsilon))

        train_variables = tf.trainable_variables()
        generator_variables = [v for v in train_variables if v.name.startswith("EDSR_gen")]
        discriminator_variables = [v for v in train_variables if v.name.startswith("EDSR_dis")]
        self.train_d = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=discriminator_variables)
        self.train_g = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=generator_variables)
        
        self.g_output = gen_f
        
        with tf.name_scope('train_summary'):
            
            tf.summary.scalar("d_loss", self.d_loss, collections=['train'])
            tf.summary.scalar("g_loss", self.g_loss, collections=['train'])
            tf.summary.scalar("dis_f_mean", tf.reduce_mean(dis_f), collections=['train'])
            tf.summary.scalar("dis_t_mean", tf.reduce_mean(dis_t), collections=['train'])
            tf.summary.image("target_image", tf.reshape(self.image_target, [self.batch_size, self.image_size, self.image_size, self.color_dim]), collections=['train'])
            tf.summary.image("output_image", tf.reshape(gen_f, [self.batch_size, self.image_size, self.image_size, self.color_dim]), collections=['train'])
    
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):

            tf.summary.scalar("d_loss", self.d_loss, collections=['test'])
            tf.summary.scalar("g_loss", self.g_loss, collections=['test'])
            tf.summary.image("target_image", tf.reshape(self.image_target, [self.batch_size, self.image_size, self.image_size, self.color_dim]), collections=['test'])
            tf.summary.image("output_image", tf.reshape(gen_f, [self.batch_size, self.image_size, self.image_size, self.color_dim]), collections=['test'])
        
            self.merged_summary_test = tf.summary.merge_all('test')                 
        
        self.saver = tf.train.Saver()

        
    def train_EDSR_RaGAN_MNIST(self):
        """
        Training process.
        """     
        print("Training...")

        # Define dataset path
        mnist = input_data.read_data_sets("/home/wei/ML/dataset/MNIST_data/", one_hot=True)
        print("...")
        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")       
        
        # Define iteration counter, learning rate...
        itera_counter = 0
        learning_rate = self.learning_rate

        epoch_pbar = tqdm(range(self.curr_epoch, self.epoch))
        for ep in epoch_pbar:            
            # Run by batch images
            idxs = 1000

            epoch_pbar.set_description("Epoch: [%2d], lr:%f" % ((ep+1), learning_rate))
            epoch_pbar.refresh()
        
            batch_pbar = tqdm(range(0, idxs), desc="Batch: [0]")

            for idx in batch_pbar:                
                batch_pbar.set_description("Batch: [%2d]" % ((idx+1)))
                itera_counter += 1
                
                mnist_img, _ = mnist.train.next_batch(self.batch_size)
                                  
                for d_iter in range(0, 5):
                    _, d_loss, g_loss \
                    = self.sess.run([self.train_d, self.d_loss, self.g_loss],
                                                                             feed_dict={   
                                                                                         self.image_target: mnist_img,
                                                                                         self.dropout: 1.,
                                                                                         self.lr:learning_rate
                                                                                       })
                
                _ = self.sess.run([self.train_g], 
                                                   feed_dict={
                                                               self.image_target: mnist_img,
                                                               self.dropout: 1.,
                                                               self.lr:learning_rate 
                                                             })
            
            print("EP:[{}], d_loss = [{}], g_loss = [{}]\n".format(ep, d_loss, g_loss))
    
            if ep % 5 == 0 and ep != 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, itera_counter)
                
                train_sum = self.sess.run(self.merged_summary_train, 
                                                                    feed_dict={
                                                                                self.image_target: mnist_img,
                                                                                self.dropout: 1.
                                                                              })
                
                test_sum, g_output = self.sess.run([self.merged_summary_test, self.g_output],
                                                                     feed_dict={
                                                                                 self.image_target: mnist_img,
                                                                                 self.dropout: 1.,
                                                                               })
                                                                                                   
                
                                                                     
                print("Epoch: [{}]".format((ep+1)))       
                
                
                summary_writer.add_summary(train_sum, ep)
                summary_writer.add_summary(test_sum, ep)                

    def build_RCAN_WGAN_att(self):###
        """
        Build SRCNN model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.image_size*2, self.image_size*2, self.color_dim], name='images')
        self.image_target = tf.placeholder(tf.float32, [None, self.image_size*2, self.image_size*2, self.color_dim], name='labels')

        self.curr_batch_size = tf.placeholder(tf.int32, shape=[])

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        self.image_input = self.input/255.
        self.target = target = self.image_target/255.
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.image_input, self.dropout, self.is_train, self.model_ticket)
        
        ### Build model       
        gen_f = mz.build_model({"d_inputs":None, "scale":self.scale, "feature_size" :64, "reuse":False, "is_training":True, "net":"Gen"})

        dis_t, lp_t = mz.build_model({"d_inputs":self.target, "scale":self.scale, "feature_size" :64, "reuse":False, "is_training":True, "net":"Dis", "d_model":"PatchWGAN_GP"})
        dis_f, lp_f = mz.build_model({"d_inputs":gen_f, "scale":self.scale, "feature_size" :64, "reuse":True, "is_training":True, "net":"Dis", "d_model":"PatchWGAN_GP"})           

        #### WGAN-GP ####
        # Calculate gradient penalty
        self.epsilon = epsilon = tf.random_uniform([self.curr_batch_size, 1, 1, 1], 0.0, 1.0)
        x_hat = epsilon * self.target + (1. - epsilon) * (gen_f)
        d_hat = mz.build_model({"d_inputs":x_hat, "scale":self.scale, "feature_size" :64, "reuse":True, "is_training":True, "net":"Dis", "d_model":"PatchWGAN_GP"})
        
        d_gp = tf.gradients(d_hat, [x_hat])[0]
        d_gp = tf.sqrt(tf.reduce_sum(tf.square(d_gp), axis=[1,2,3]))
        d_gp = tf.reduce_mean((d_gp - 1.0)**2) * 10

        self.disc_ture_loss = disc_ture_loss = tf.reduce_mean(dis_t)
        self.disc_fake_loss = disc_fake_loss = tf.reduce_mean(dis_f)
        
        perceptual_loss = tf.pow(lp_t - lp_f, 2)

        reconstucted_weight = 10.0

        lp_weight = 0.25 

        self.d_loss =   (disc_fake_loss - disc_ture_loss) + d_gp
        self.g_l2loss = tf.reduce_mean(tf.pow(target-gen_f, 2))
        self.g_l1loss = tf.reduce_mean(tf.losses.absolute_difference(target,gen_f))
        self.g_loss = -1.0*disc_fake_loss + reconstucted_weight*self.g_l2loss + reconstucted_weight*self.g_l1loss + lp_weight*perceptual_loss

        train_variables = tf.trainable_variables()
        generator_variables = [v for v in train_variables if v.name.startswith("RCAN_gen")]
        discriminator_variables = [v for v in train_variables if v.name.startswith("RCAN_dis")]
        self.train_d = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=discriminator_variables)
        self.train_g = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=generator_variables)

        self.g_output = gen_f
        
        mse = tf.reduce_mean(tf.squared_difference(target*255.,gen_f*255.))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/mse
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)

        mse_ref = tf.reduce_mean(tf.squared_difference(target*255.,self.image_input*255.))    
        PSNR_ref = tf.constant(255**2,dtype=tf.float32)/mse_ref
        PSNR_ref = tf.constant(10,dtype=tf.float32)*log10(PSNR_ref)
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("l2_loss", self.g_l2loss, collections=['train'])
            tf.summary.scalar("d_loss", self.d_loss, collections=['train'])
            tf.summary.scalar("d_true_loss", disc_ture_loss, collections=['train'])
            tf.summary.scalar("d_fake_loss", disc_fake_loss, collections=['train'])
            tf.summary.scalar("d_lp", perceptual_loss, collections=['train']) # for tuning

            tf.summary.scalar("MSE", mse, collections=['train'])
            tf.summary.scalar("PSNR",PSNR, collections=['train'])
            tf.summary.image("input_image",self.input , collections=['train'])
            tf.summary.image("target_image",target*255, collections=['train'])
            tf.summary.image("output_image",gen_f*255, collections=['train'])

            tf.summary.histogram("d_false", dis_f, collections=['train']) 
            tf.summary.histogram("d_true", dis_t, collections=['train']) 

            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):

            tf.summary.scalar("l2_loss", self.g_l2loss, collections=['test'])
            tf.summary.scalar("d_loss", self.d_loss, collections=['test'])
            tf.summary.scalar("g_loss", disc_fake_loss, collections=['test'])
            
            tf.summary.scalar("d_lp", perceptual_loss, collections=['test']) # for tuning
            
            tf.summary.scalar("MSE", mse, collections=['test'])
            tf.summary.scalar("PSNR",PSNR, collections=['test'])
            tf.summary.scalar("PSNR_ref",PSNR_ref, collections=['test'])
            tf.summary.image("input_image",self.input , collections=['test'])
            tf.summary.image("target_image",target*255, collections=['test'])
            tf.summary.image("output_image",gen_f*255, collections=['test'])             
        
            self.merged_summary_test = tf.summary.merge_all('test')                    
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()       
        
    def train_RCAN_WGAN_att(self):
        """
        Training process.
        """     
        print("Training...")

        # Define dataset path
        #96X96
        test_dataset = self.load_divk(self.test_dir+"/X{}".format(self.scale), type='test_baseline')
        dataset = self.load_divk(self.train_dir, lrtype='bicubic', type='train')
        # test_dataset = self.load_divk("/home/sdc1/dataset/SuperResolution/Set5/pretrain_Set5_RCAN/validation/X{}/".format(self.scale), type="test_baseline")
        # dataset = self.load_divk("/home/sdc1/dataset/SuperResolution/DIV2K/pretrain_DIV2K_RCAN/", lrtype='bicubic', type='train')   

        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")       
        
        # Define iteration counter, learning rate...
        if(self.batch_size >= 16):
            save_ep = 5
            lr_ep = 4000
            lr_ep_2 = 8000            
        else:
            save_ep = 200       
            lr_ep = 250000
            lr_ep_2 = 500000            

        itera_counter = 0
        
        train_data, train_label  = zip(*dataset)

        learning_rate = self.learning_rate
        if self.curr_epoch >= lr_ep and self.curr_epoch < lr_ep_2:
            learning_rate = self.learning_rate / 2 # only decay once
        elif self.curr_epoch >= lr_ep and self.curr_epoch >= lr_ep_2:
            learning_rate = self.learning_rate / 4 # decay twice
            
        print("Current learning rate: [{}]".format(learning_rate))
        
        best_loss = 100
        
        epoch_pbar = tqdm(range(self.curr_epoch, self.epoch))
        for ep in epoch_pbar:            
            
            random.shuffle(dataset) 
            train_data, train_label  = zip(*dataset)
            test_data, test_label  = zip(*test_dataset)

            epoch_pbar.set_description("Epoch: [%2d], lr:%f" % ((ep+1), learning_rate))
            epoch_pbar.refresh()

            if(self.batch_size >= 16):
                batch_pbar = tqdm(range(0, len(train_data)//self.batch_size), desc="Batch: [0]")       
            else:
                batch_pbar = tqdm(range(0, 1), desc="Batch: [0]") # for tuning hyperparameters            

            if (ep % lr_ep == 0 or ep % lr_ep_2 == 0) and ep != 0 and ep <= lr_ep_2:
                learning_rate = learning_rate/2
                print("[Learning Rate Decay]")
                print("Current learning rate: [{}]".format(learning_rate))

            for idx in batch_pbar:                
                
                batch_pbar.set_description("Batch: [%2d]" % ((idx+1)))
                itera_counter += 1
                batch_index = idx*self.batch_size 

                batch_images, batch_labels = batch_shuffle_rndc(train_data, train_label, 1, self.image_size*2, batch_index, self.batch_size)                

                _ = self.sess.run([self.train_g], 
                                                   feed_dict={
                                                               self.input: batch_images,
                                                               self.image_target: batch_labels,
                                                               self.dropout: 1.,
                                                               self.lr:learning_rate 
                                                             })
                                                 
                for d_iter in range(0, 5):
                    _, d_loss, g_loss \
                    = self.sess.run([self.train_d, self.d_loss, self.g_loss],
                                                                             feed_dict={   
                                                                                         self.input: batch_images,
                                                                                         self.image_target: batch_labels,
                                                                                         self.curr_batch_size: self.batch_size,
                                                                                         self.dropout: 1.,
                                                                                         self.lr:learning_rate
                                                                                       })
                   
            if ep % save_ep == 0 and ep != 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, itera_counter)
                
                train_sum = self.sess.run(self.merged_summary_train, 
                                                                    feed_dict={
                                                                                self.input: batch_images,
                                                                                self.image_target: batch_labels,
                                                                                self.curr_batch_size: self.batch_size,
                                                                                self.dropout: 1.
                                                                              })
                
                test_sum, g_output, loss = self.sess.run([self.merged_summary_test, self.g_output, self.g_l2loss],
                                                                     feed_dict={
                                                                                 self.input: test_data,  
                                                                                 self.image_target: test_label,
                                                                                 self.curr_batch_size: len(test_label),
                                                                                 self.dropout: 1.,
                                                                               })
                                                                                                   
                
                                                                     
                print("Epoch: [{}]".format((ep+1)))       
                
                if loss < best_loss:
                    best_loss = loss
#                    self.save_best_ckpt(self.checkpoint_dir, self.ckpt_name, best_loss, itera_counter)
                    print("Current Loss: [{}]\n".format(best_loss))
                
                summary_writer.add_summary(train_sum, ep)
                summary_writer.add_summary(test_sum, ep)                                                                      