import cv2
import numpy as np
import os
import argparse
import tensorflow as tf
import sys
import scipy.misc
import matplotlib.pyplot as plt

sys.path.append('./utility')
import utils as ut
import model_zoo
import config
import unittest

FULL_IMAGE = False

class evaluation:

    def __init__(self, config):

        eval_conf = config["evaluation"]
        model_conf = [mkey for mkey in eval_conf['models'][0]]
        model_ticket = model_conf[0]

        #Set evaluation configuration
        self.dataset_path = eval_conf["dataroot"]
        self.padding_size = eval_conf['models'][0][model_ticket]["padding"]
        self.subimg_size = eval_conf['models'][0][model_ticket]["subimages"]
        self.model_ticket = model_ticket
        self.ckpt_file = eval_conf['models'][0][model_ticket]["ckpt_file"]
        self.model_config = eval_conf['models'][0][model_ticket]["model_config"]
        self.scale = eval_conf['models'][0][model_ticket]["scale"][0]
        self.save_path = eval_conf['save_path']
        print("checkpoint: ", self.ckpt_file)

    def input_setup(self):

        #List image in evaluation folder
        #Split image

        image_list = os.listdir(self.dataset_path)

        images_grid = {}

        for img_name in image_list:

            image_path = os.path.join(self.dataset_path, img_name)
            img = scipy.misc.imread(image_path, mode="RGB")

            if FULL_IMAGE:
                self.subimg_size = img.shape

            grid_imgs = self.split_img(img, img_name)

            images_grid[image_path] = {"grids":grid_imgs,
                                       "img_size":img.shape
                                      }
        return images_grid

    def split_img(self, image, imgname):

        #Split image into grids

        img_shape = image.shape
        shaved_size = [img_shape[0] - 2*self.padding_size,
                       img_shape[1] - 2*self.padding_size,
                       img_shape[2]
                      ]

        grid_imgs = {} # Container for saved splited image
        grid_step_size = self.subimg_size # Use same step for rows and columns
        grid_size = (self.subimg_size[0] + 2*self.padding_size,
                     self.subimg_size[1] + 2*self.padding_size,
                     self.subimg_size[2])

        for r in range(0, shaved_size[0] - self.subimg_size[0] + grid_step_size[0], grid_step_size[0]):
            for c in range(0, shaved_size[1] - self.subimg_size[1] + grid_step_size[1], grid_step_size[1]):

                grid_r = r + self.padding_size  # grid location after shave padding  
                grid_c = c + self.padding_size  # grid location after shave padding

                subimg_rloc = [grid_r - self.padding_size, grid_r + self.subimg_size[0] + self.padding_size]
                subimg_cloc = [grid_c - self.padding_size, grid_c + self.subimg_size[1] + self.padding_size]


                # Switch Cases for grid size is not exact fit image sizes
                if subimg_rloc[1] < img_shape[0] and subimg_cloc[1] < img_shape[1]:

                    subimg_rloc = subimg_rloc
                    subimg_cloc = subimg_cloc

                elif subimg_rloc[1] >= img_shape[0] and subimg_cloc[1] < img_shape[1]:

                    subimg_rloc = [grid_r - self.padding_size, img_shape[0]]
                    
                elif subimg_rloc[1] < img_shape[0] and subimg_cloc[1] >= img_shape[1]:
            
                    subimg_cloc = [grid_c - self.padding_size, img_shape[1]]

                else:
                    subimg_rloc = [grid_r - self.padding_size, img_shape[0]]
                    subimg_cloc = [grid_c - self.padding_size, img_shape[1]]


                subimg = np.zeros(grid_size)

                subimg[ 0 : subimg_rloc[1] - subimg_rloc[0],
                        0 : subimg_cloc[1] - subimg_cloc[0],
                        :]\
                = image[    subimg_rloc[0]: subimg_rloc[1],
                            subimg_cloc[0]: subimg_cloc[1],
                            :
                        ]
                
                grid_imgs[str(subimg_rloc[0]) + "_" + str(subimg_rloc[1]) + \
                            "_" +  str(subimg_cloc[0]) + "_" +  str(subimg_cloc[1])] = subimg

                
        return grid_imgs
    
    def merge_img(self, img_size, grid_imgs):

        #Merge grids into image
        # Create an empty array for merging image

        shaved_size =   [   (img_size[0] - 2*self.padding_size)*self.scale,
                            (img_size[1] - 2*self.padding_size)*self.scale,
                            img_size[2]
                        ]

        merged_image = np.zeros([shaved_size[0], shaved_size[1], shaved_size[2]])

        for k in grid_imgs:

            key = k.split("_")
            
            grid_r0 = int(key[0])
            grid_r1 = int(key[1])
            grid_c0 = int(key[2])
            grid_c1 = int(key[3])

            mergeimg_loc_r = [grid_r0*self.scale, (grid_r1 - 2*self.padding_size)*self.scale]
            mergeimg_loc_c = [grid_c0*self.scale, (grid_c1 - 2*self.padding_size)*self.scale]
            gridimg_loc_r = [self.padding_size*self.scale , (grid_r1 - grid_r0 - self.padding_size)*self.scale]
            gridimg_loc_c = [self.padding_size*self.scale , (grid_c1 - grid_c0 - self.padding_size)*self.scale]

            merged_image[mergeimg_loc_r[0]:mergeimg_loc_r[1],
                         mergeimg_loc_c[0]:mergeimg_loc_c[1],
                         :]\
            = grid_imgs[k][gridimg_loc_r[0]:gridimg_loc_r[1],
                             gridimg_loc_c[0]:gridimg_loc_c[1],
                             :]

        return merged_image


    #Load Tensorflow Model from check points
    def load_model(self, config = {}, isNormallized=True):

        tf.reset_default_graph() 

        if FULL_IMAGE:
           self.inputs = tf.placeholder(tf.float32, [  None,
                                                       None,
                                                       None,
                                                       3])

        else:
           self.inputs = tf.placeholder(tf.float32, [ 1,
                                                      int((self.subimg_size[0]+self.padding_size*2)),
                                                      int((self.subimg_size[1]+self.padding_size*2)),
                                                      3])
                
        if isNormallized: inputs_n = self.inputs/255.
        else: inputs_n = self.inputs

        mz = model_zoo.model_zoo(inputs_n, None, False, self.model_ticket)
        predict_op = mz.build_model(self.model_config)
        
        if type(predict_op) is tuple:
            self.predict_op = predict_op[0]
        else:
            self.predict_op = predict_op

        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, self.ckpt_file)
        print("Session created and Model restored")


    def prediction(self, image):
     
        #run model in model_ticket_list and return prediction       
        predicted = self.sess.run(self.predict_op, feed_dict = {self.inputs:[image]})
        return predicted
    
    def run_evaluation(self):

        #run evaluation for images in input folder and save as image

        grid_imgs = self.input_setup()
        self.load_model()
        progress = 1

        for img in grid_imgs:

            output_grids = {}

            for grid_key in grid_imgs[img]["grids"]:
               
                pred = self.prediction(grid_imgs[img]["grids"][grid_key])
                output_grids[grid_key] = pred[0]

            merged_img = self.merge_img(grid_imgs[img]["img_size"], output_grids)
            merged_img = merged_img*255.
            merged_img = np.round(merged_img, 0)
            merged_img = np.clip(merged_img, 0, 255).astype('uint8')

            test_img = scipy.misc.toimage(merged_img, high=np.max(merged_img), low=np.min(merged_img))
            test_img = scipy.misc.toimage(merged_img, high=np.max(merged_img), low=np.min(merged_img))   

            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)         
            test_img.save(self.save_path+"/{}".format(img.split('/')[-1]))
            print(self.save_path)
            #target_img.save("./evaluation/target/target_{}.png".format(progress))
            print("Process:{}/{}".format(progress, len(grid_imgs)))
            progress += 1
            
        
def  main_process():

    #Parsing argumet(configuration name) from shell, 
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="example",help="Configuration name")
    args = parser.parse_args()
    conf = config.config(args.config).config

    eval = evaluation(conf)
    eval.run_evaluation()

#Unit Test
class test_evaluation(unittest.TestCase):

    def setUp(self):

        self.dataset = '/home/ubuntu/dataset/SuperResolution/Set5/preprocessed_scale_1'
        model_ticket = "edsr_attention_v3"
        ckpt_file = "/home/ubuntu/model/model/SR_project/edsr_attention_v3_noatt/edsr_attention_v3_noatt-111496"
        model_config  = {"d_inputs":None, "d_target":None,"scale":2,"feature_size" : 64,"dropout" : 1.0,"feature_size" : 64, "is_training":False, "reuse":False, "net":"Gen"}
                                        
        self.eval = evaluation(self.dataset, model_ticket, ckpt_file, model_config)

    def test_split_and_merge(self):

        grid_imgs = self.eval.input_setup()

        for imgname in grid_imgs:

            img = scipy.misc.imread(imgname, mode="RGB")
            print("Source size", img.shape)
            #plt.imshow(img.astype(np.uint8))
            #plt.show()

            merged_img = self.eval.merge_img(grid_imgs[imgname]["img_size"], grid_imgs[imgname]["grids"])
            print("Merged size",merged_img.shape)
            

            shaved_shape = (img.shape[0] - 2*self.eval.padding_size,
                            img.shape[1] - 2*self.eval.padding_size,
                            img.shape[2])

            self.assertEqual(shaved_shape, merged_img.shape, msg="Must equal")
           

    def test_run_evaluation(self):

        self.eval.run_evaluation()

            

if __name__ == '__main__':

    print("Start Evaluation")
    #unittest.main()
    #test_eval = test_evaluation()
    #test_eval.test_run_evaluation()
    main_process()
    print("Done Evaluation")