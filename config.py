class config:

	def __init__(self, configuration):
		
		self.configuration = configuration
		self.config = {
						"common":{},
						"train":{},
						"evaluation":{
								"dataroot":None,
								"test_set":["Set5", "Set14", "BSD100"],
								"models":{},
								
							}
						}
		self.get_config()


	def get_config(self):

		try:
			conf = getattr(self, self.configuration)
			conf()

		except: 
			print("Can not find configuration")
			raise
			
			flags.DEFINE_string("mode", "normal", "operation mode: normal or freq [normal]")

	def EDSR_RaGAN(self):
        
		train_config = self.config["train"]

		train_config["mode"] = "small" # Operation mode: normal or freq [normal]
		train_config["epoch"] = 40000  # Number of epoch [10]
		train_config["batch_size"] = 16 # The size of batch images [128]
		train_config["image_size"] = 48 # The size of image to use [33]
		train_config["label_size"] = 96 # The size of label to produce [21]
		train_config["learning_rate"] = 1e-4 #The learning rate of gradient descent algorithm [1e-4]
		train_config["color_dim"] = 3 # Dimension of image color. [1]
		train_config["scale"] = 4 # The size of scale factor for preprocessing input image [3]
		train_config["train_extract_stride"] = 14 #The size of stride to apply input image [14]
		train_config["test_extract_stride"] = train_config["label_size"] #The size of stride to apply input image [14]
		train_config["checkpoint_dir"] = "/home/wei/ML/model/SuperResolution/SR-project-prototype/" #Name of checkpoint directory [checkpoint]
		train_config["log_dir"] = "/home/wei/ML/model/SuperResolution/SR-project-prototype/log/" #Name of checkpoint directory [checkpoint]
		train_config["output_dir"] = "output" # Name of sample directory [output]
		train_config["train_dir"] =  "Train" # Name of train dataset directory
		train_config["test_dir"] = "Test/Set5" # Name of test dataset directory [Test/Set5]
		train_config["h5_dir"] = "/home/wei/ML/dataset/SuperResolution/train" # Name of train dataset .h5 file
		train_config["train_h5_name"] = "train" # Name of train dataset .h5 file
		train_config["test_h5_name"] = "test" # Name of test dataset .h5 file
      
		train_config["ckpt_name"] = "025_10_wo_att_RSGAN-GP" # Name of checkpoints 0.1 [1,1,1,1] ******************************               
                                   
		train_config["is_train"] = True # True for training, False for testing [True]
		train_config["model_ticket"] = "EDSR_RaGAN" # Name of checkpoints
		train_config["curr_epoch"] = 0 # Name of checkpoints        
        
		def EDSR_RaGAN(self):
						
			mconfig = {}
			
			mconfig["EDSR_WGAN_att"] = {

										"scale":[1],
										"subimages":(80, 80, 3), #V1:[96,96]
										"padding":8,
										"ckpt_file":"/home/wei/ML/model/SuperResolution/SR-project-prototype/025_10_full_PatchWGAN-GP_v4_ep0_MSE_lp_28/best_performance/025_10_full_PatchWGAN-GP_v4_ep0_MSE_lp_28_0.0010903702350333333-70400",
										"isGray": False,
										"isNormallized":True,
										"upsample": False,
										"sub_mean":False,
										"model_config" :{"d_inputs":None, "d_target":None, "scale":2, "feature_size":64, "reuse":False, "is_training":False, "net":"Gen"}
										}
			
			
			return mconfig

		eval_config = self.config["evaluation"]
		eval_config["dataroot"] = '/data/wei/dataset/SuperResolution/eval_input/'        
		eval_config["models"] = [EDSR_RaGAN(self)]
		eval_config["summary_file"] = "example_summary.txt" 

	def EDSR_WGAN(self):
        
		train_config = self.config["train"]

		train_config["mode"] = "small" # Operation mode: normal or freq [normal]
		train_config["epoch"] = 40000  # Number of epoch [10]
		train_config["batch_size"] = 16 # The size of batch images [128]
		train_config["image_size"] = 48 # The size of image to use [33]
		train_config["label_size"] = 96 # The size of label to produce [21]
		train_config["learning_rate"] = 1e-4 #The learning rate of gradient descent algorithm [1e-4]
		train_config["color_dim"] = 3 # Dimension of image color. [1]
		train_config["scale"] = 4 # The size of scale factor for preprocessing input image [3]
		train_config["train_extract_stride"] = 14 #The size of stride to apply input image [14]
		train_config["test_extract_stride"] = train_config["label_size"] #The size of stride to apply input image [14]
		train_config["checkpoint_dir"] = "/home/wei/ML/model/SuperResolution/SR-project-prototype/" #Name of checkpoint directory [checkpoint]
		train_config["log_dir"] = "/home/wei/ML/model/SuperResolution/SR-project-prototype/log/" #Name of checkpoint directory [checkpoint]
		train_config["output_dir"] = "output" # Name of sample directory [output]
		train_config["train_dir"] =  "Train" # Name of train dataset directory
		train_config["test_dir"] = "Test/Set5" # Name of test dataset directory [Test/Set5]
		train_config["h5_dir"] = "/home/wei/ML/dataset/SuperResolution/train" # Name of train dataset .h5 file
		train_config["train_h5_name"] = "train" # Name of train dataset .h5 file
		train_config["test_h5_name"] = "test" # Name of test dataset .h5 file
      
		train_config["ckpt_name"] = "025_10_wo_att" # Name of checkpoints 0.1 [1,1,1,1] ******************************               
                                   
		train_config["is_train"] = True # True for training, False for testing [True]
		train_config["model_ticket"] = "EDSR_WGAN" # Name of checkpoints
		train_config["curr_epoch"] = 0 # Name of checkpoints        
        
		def EDSR_WGAN(self):
						
			mconfig = {}
			
			mconfig["EDSR_WGAN_att"] = {

										"scale":[1],
										"subimages":(80, 80, 3), #V1:[96,96]
										"padding":8,
										"ckpt_file":"/home/wei/ML/model/SuperResolution/SR-project-prototype/025_10_full_PatchWGAN-GP_v4_ep0_MSE_lp_28/best_performance/025_10_full_PatchWGAN-GP_v4_ep0_MSE_lp_28_0.0010903702350333333-70400",
										"isGray": False,
										"isNormallized":True,
										"upsample": False,
										"sub_mean":False,
										"model_config" :{"d_inputs":None, "d_target":None, "scale":2, "feature_size":64, "reuse":False, "is_training":False, "net":"Gen"}
										}
			
			
			return mconfig

		eval_config = self.config["evaluation"]
		eval_config["dataroot"] = '/data/wei/dataset/SuperResolution/eval_input/'        
		eval_config["models"] = [EDSR_WGAN(self)]
		eval_config["summary_file"] = "example_summary.txt"      

	def EDSR_WGAN_att_v1(self):

		train_config = self.config["train"]

		train_config["mode"] = "small" # Operation mode: normal or freq [normal]
		train_config["epoch"] = 40000  # Number of epoch [10]
		train_config["batch_size"] = 16 # The size of batch images [128]
		train_config["image_size"] = 48 # The size of image to use [33]
		train_config["label_size"] = 96 # The size of label to produce [21]
		train_config["learning_rate"] = 1e-4 #The learning rate of gradient descent algorithm [1e-4]
		train_config["color_dim"] = 3 # Dimension of image color. [1]
		train_config["scale"] = 4 # The size of scale factor for preprocessing input image [3]
		train_config["train_extract_stride"] = 14 #The size of stride to apply input image [14]
		train_config["test_extract_stride"] = train_config["label_size"] #The size of stride to apply input image [14]
		train_config["checkpoint_dir"] = "/home/wei/ML/model/SuperResolution/SR-project-prototype/" #Name of checkpoint directory [checkpoint]
		train_config["log_dir"] = "/home/wei/ML/model/SuperResolution/SR-project-prototype/log/" #Name of checkpoint directory [checkpoint]
		train_config["output_dir"] = "output" # Name of sample directory [output]
		train_config["train_dir"] =  "Train" # Name of train dataset directory
		train_config["test_dir"] = "Test/Set5" # Name of test dataset directory [Test/Set5]
		train_config["h5_dir"] = "/home/wei/ML/dataset/SuperResolution/train" # Name of train dataset .h5 file
		train_config["train_h5_name"] = "train" # Name of train dataset .h5 file
		train_config["test_h5_name"] = "test" # Name of test dataset .h5 file
      
#		train_config["ckpt_name"] = "025_10_full_PatchWGAN-GP_v4_ep0_MSE_lp_64_layer1_3" # Name of checkpoints 0.1 [1,1,1,1] ******************************               
		train_config["ckpt_name"] = "025_10_w_att_RCAN" # Name of checkpoints 0.1 [1,1,1,1] ******************************                       
                                   
		train_config["is_train"] = True # True for training, False for testing [True]
		train_config["model_ticket"] = "EDSR_WGAN_att" # Name of checkpoints
		train_config["curr_epoch"] = 0 # Name of checkpoints        
        
		def EDSR_WGAN_att_v1(self):
						
			mconfig = {}
			
			mconfig["EDSR_WGAN_att"] = {

										"scale":[1],
										"subimages":(80, 80, 3), #V1:[96,96]
										"padding":8,
										"ckpt_file":"/home/wei/ML/model/SuperResolution/SR-project-prototype/025_10_full_PatchWGAN-GP_v4_ep0_MSE_lp_28/best_performance/025_10_full_PatchWGAN-GP_v4_ep0_MSE_lp_28_0.0010903702350333333-70400",
#										"ckpt_file":"/home/wei/ML/model/SuperResolution/SR-project-prototype/025_10_full_PatchWGAN-GP_v4_ep0_MSE_lp_28/025_10_full_PatchWGAN-GP_v4_ep0_MSE_lp_28-81950",
#										"ckpt_file":"/mnt/GPU_Server/ML/model/SuperResolution/SR-project-prototype/025_10_full_PatchWGAN-GP_v4_ep0_MSE_lp_28_layer1_3_pre/025_10_full_PatchWGAN-GP_v4_ep0_MSE_lp_28_layer1_3_pre-272525",                                        
#										"ckpt_file":"/home/wei/ML/model/SuperResolution/SR-project-prototype/training_data/EDSR_WGAN_att_v1_x2_95_full/best_performance/EDSR_WGAN_att_v1_x2_95_full_0.009988665580749512-5225",
										"isGray": False,
										"isNormallized":True,
										"upsample": False,
										"sub_mean":False,
										"model_config" :{"d_inputs":None, "d_target":None, "scale":2, "feature_size":64, "reuse":False, "is_training":False, "net":"Gen"}
										}
			
			
			return mconfig

		eval_config = self.config["evaluation"]
		eval_config["dataroot"] = '/data/wei/dataset/SuperResolution/eval_input/'        
		eval_config["models"] = [EDSR_WGAN_att_v1(self)]
		eval_config["summary_file"] = "example_summary.txt"

	def RCAN_WGAN_att(self):

		train_config = self.config["train"]

		train_config["mode"] = "small" # Operation mode: normal or freq [normal]
		train_config["epoch"] = 5000  # Number of epoch [10]
		train_config["batch_size"] = 16 # The size of batch images [128]
		train_config["image_size"] = 24 # The size of image to use [33]
		train_config["label_size"] = 96 # The size of label to produce [21]
		train_config["learning_rate"] = 1e-4 #The learning rate of gradient descent algorithm [1e-4]
		train_config["color_dim"] = 3 # Dimension of image color. [1]
		train_config["scale"] = 4 # The size of scale factor for preprocessing input image [3]
		train_config["train_extract_stride"] = 14 #The size of stride to apply input image [14]
		train_config["test_extract_stride"] = train_config["label_size"] #The size of stride to apply input image [14]
		train_config["checkpoint_dir"] = "/data/wei/model/SuperResolution/SR-project-AttWGAN/" #Name of checkpoint directory [checkpoint]
		train_config["log_dir"] = "/data/wei/model/SuperResolution/SR-project-AttWGAN/log/" #Name of checkpoint directory [checkpoint]
		train_config["output_dir"] = "output" # Name of sample directory [output]
		train_config["train_dir"] =  "Train" # Name of train dataset directory
		train_config["test_dir"] = "Test/Set5" # Name of test dataset directory [Test/Set5]
		train_config["h5_dir"] = "/home/wei/ML/dataset/SuperResolution/train" # Name of train dataset .h5 file
		train_config["train_h5_name"] = "train" # Name of train dataset .h5 file
		train_config["test_h5_name"] = "test" # Name of test dataset .h5 file
                     
#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1" # Name of checkpoints 0.1 [1,1,1,1] ******************************                       
#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_3_RCAB_5" # Name of checkpoints 0.1 [1,1,1,1] ******************************                               
#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_3_RCAB_5_025_20" # Name of checkpoints 0.1 [1,1,1,1] ******************************                  
#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_3_RCAB_5_025_50" # Name of checkpoints 0.1 [1,1,1,1] ******************************                 

#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_3_RCAB_5_L1_DisAtt_025_05" # Name of checkpoints 0.1 [1,1,1,1] ******************************                 
#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_3_RCAB_5_L1_DisAtt_025_1" # Name of checkpoints 0.1 [1,1,1,1] ******************************        

#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_3_RCAB_5_L1_DisAtt_025_1_3x3" # Name of checkpoints 0.1 [1,1,1,1] ******************************                                                  
#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_3_RCAB_5_L1_DisAtt_025_10" # Name of checkpoints 0.1 [1,1,1,1] ******************************                                 

#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_3_RCAB_5_L1_DisAtt_001_1_001" # [GAN, Lp, L1] 
#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_3_RCAB_5_L1_DisAtt_0005_1_0025" # [GAN, Lp, L1]         
#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_3_RCAB_5_L1_DisAtt_0005_1_005" # [GAN, Lp, L1]         
#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_3_RCAB_5_L1_DisAtt_0005_1_1" # [GAN, Lp, L1] ** GAN=1        
#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_3_RCAB_5_L1_DisAtt_0005_1_5" # [GAN, Lp, L1] ** GAN=1            
#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_3_RCAB_5_L1_DisAtt_005_1_0025" # [GAN, Lp, L1] ** GAN=1                    
#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_3_RCAB_5_L1_DisAtt_01_1_0025" # [GAN, Lp, L1] ** GAN=1    
        
#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_3_RCAB_5_L2_DisAtt_interp_Add_sigmoid_1_025_10" # [GAN, Lp, L1]          
#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_3_RCAB_5_L2_DisAtt_interp_sigmoid_1_025_10" # [GAN, Lp, L1]                  

#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_3_RCAB_5_L2_DisAtt_interp_Add_relu_1_025_10" # [GAN, Lp, L1]             
#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_3_RCAB_5_L2_DisAtt_interp_relu_1_025_10" # [GAN, Lp, L1]                     

# Compact model
#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_1_RCAB_3_L2_DisAtt_interp_Add_1_025_10" # [GAN, Lp, L1]                     
#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_1_RCAB_3_L2_DisAtt_interp_Add_1_025_20" # [GAN, Lp, L1]               
		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_1_RCAB_3_L2_DisAtt_interp_Add_1_025_50" # [GAN, Lp, L1]            
#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_1_RCAB_3_L2_DisAtt_interp_Add_1_025_75" # [GAN, Lp, L1]             
        

#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_1_RCAB_3_L2_DisAtt_interp_Add_1_05_10" # [GAN, Lp, L1]                     
#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_1_RCAB_3_L2_DisAtt_interp_Add_1_1_10" # [GAN, Lp, L1]         

#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_1_RCAB_3_L2_DisAtt_interp_Add_1_025_10_before" # [GAN, Lp, L1]                     

# without interp
#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_1_RCAB_3_L2_DisAtt_wo_interp_1_025_10" # [GAN, Lp, L1]    
# without attention
#		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_1_RCAB_3_L2_DisAtt_wo_att_1_025_10" # [GAN, Lp, L1]            
# Stage1: DBPN
#		train_config["ckpt_name"] = "DBPN_WGAN_att_v1_RG_1_RCAB_3_L2_DisAtt_interp_Add_1_025_10" # [GAN, Lp, L1]            
# Stage1: EDSR
#		train_config["ckpt_name"] = "EDSR_WGAN_att_v1_RG_1_RCAB_3_L2_DisAtt_interp_Add_1_025_10" # [GAN, Lp, L1]                  
                                     
		train_config["is_train"] = True # True for training, False for testing [True]
		train_config["model_ticket"] = "RCAN_WGAN_att" # Name of checkpoints
		train_config["curr_epoch"] = 4591 # Name of checkpoints        
        
		def RCAN_WGAN_att(self):
						
			mconfig = {}
			
			mconfig["RCAN_WGAN_att"] = {

										"scale":[1],
										#"subimages":(80, 80, 3), #V1:[96,96]
										"subimages":(40, 40, 3), #V1:[96,96]                                        
										"padding":8,
										"ckpt_file":"/data/wei/model/SuperResolution/SR-project-AttWGAN/RCAN_WGAN_att_v1_RG_1_RCAB_3_L2_DisAtt_interp_Add_1_025_50/RCAN_WGAN_att_v1_RG_1_RCAB_3_L2_DisAtt_interp_Add_1_025_50-158180",
										"isGray": False,
										"isNormallized":True,
										"upsample": False,
										"sub_mean":False,
										"model_config" :{"d_inputs":None, "d_target":None, "scale":2, "feature_size":64, "reuse":False, "is_training":False, "net":"Gen", "net_size": [1, 3]}
										}
			
			
			return mconfig

		eval_config = self.config["evaluation"]
#		eval_config["dataroot"] = '/data/wei/dataset/SuperResolution/eval_input/'        
#		eval_config["dataroot"] = '/data/wei/dataset/SuperResolution/RCAN/SR/BI/RCAN/B100/x4'                
		eval_config["dataroot"] = '/data/wei/dataset/SuperResolution/RCAN/SR/BI/RCAN/PIRM/Validation/x4'                        
		eval_config["models"] = [RCAN_WGAN_att(self)]
		eval_config["summary_file"] = "example_summary.txt"

	def RCAN_WGAN_att_viz(self):
    
		train_config = self.config["train"]

		train_config["mode"] = "small" # Operation mode: normal or freq [normal]
		train_config["epoch"] = 5000  # Number of epoch [10]
		train_config["batch_size"] = 16 # The size of batch images [128]
		train_config["image_size"] = 24 # The size of image to use [33]
		train_config["label_size"] = 96 # The size of label to produce [21]
		train_config["learning_rate"] = 1e-4 #The learning rate of gradient descent algorithm [1e-4]
		train_config["color_dim"] = 3 # Dimension of image color. [1]
		train_config["scale"] = 4 # The size of scale factor for preprocessing input image [3]
		train_config["train_extract_stride"] = 14 #The size of stride to apply input image [14]
		train_config["test_extract_stride"] = train_config["label_size"] #The size of stride to apply input image [14]
		train_config["checkpoint_dir"] = "/data/wei/model/SuperResolution/SR-project-AttWGAN/" #Name of checkpoint directory [checkpoint]
		train_config["log_dir"] = "/data/wei/model/SuperResolution/SR-project-AttWGAN/log/" #Name of checkpoint directory [checkpoint]
		train_config["output_dir"] = "output" # Name of sample directory [output]
		train_config["train_dir"] =  "Train" # Name of train dataset directory
		train_config["test_dir"] = "Test/Set5" # Name of test dataset directory [Test/Set5]
		train_config["h5_dir"] = "/home/wei/ML/dataset/SuperResolution/train" # Name of train dataset .h5 file
		train_config["train_h5_name"] = "train" # Name of train dataset .h5 file
		train_config["test_h5_name"] = "test" # Name of test dataset .h5 file
   
# Compact model
            
		train_config["ckpt_name"] = "RCAN_WGAN_att_v1_RG_1_RCAB_3_L2_DisAtt_interp_Add_1_025_50" # [GAN, Lp, L1]            
		train_config["is_train"] = True # True for training, False for testing [True]
		train_config["model_ticket"] = "RCAN_WGAN_att" # Name of checkpoints
		train_config["curr_epoch"] = 4591 # Name of checkpoints        
        
		def RCAN_WGAN_att_viz(self):
						
			mconfig = {}
			
			mconfig["RCAN_WGAN_att_viz"] = {

										"scale":[1],
										#"subimages":(80, 80, 3), #V1:[96,96]
										"subimages":(40, 40, 3), #V1:[96,96]                                        
										"padding":8,
										"ckpt_file":"/home/ubuntu/model/model/SR_project/RCAN_WGAN_Final/RCAN_WGAN_att_v1_RG_1_RCAB_3_L2_DisAtt_interp_Add_1_025_50/RCAN_WGAN_att_v1_RG_1_RCAB_3_L2_DisAtt_interp_Add_1_025_50-158180",
										"isGray": False,
										"isNormallized":True,
										"upsample": False,
										"sub_mean":False,
										"model_config" :{"d_inputs":None, "d_target":None, "scale":2, "feature_size":64, "reuse":False, "is_training":False, "net":"Gen", "net_size": [1, 3]}
										}
			
			
			return mconfig

		eval_config = self.config["evaluation"]
#		eval_config["dataroot"] = '/data/wei/dataset/SuperResolution/eval_input/'        
#		eval_config["dataroot"] = '/data/wei/dataset/SuperResolution/RCAN/SR/BI/RCAN/B100/x4'                
		eval_config["dataroot"] = '/home/ubuntu/dataset/SuperResolution/Attviz/'                        
		eval_config["models"] = [RCAN_WGAN_att_viz(self)]
		eval_config["summary_file"] = "example_summary.txt"

	def RCAN_RSGAN_GP_att(self):

		train_config = self.config["train"]

		train_config["mode"] = "small" # Operation mode: normal or freq [normal]
		train_config["epoch"] = 40000  # Number of epoch [10]
		train_config["batch_size"] = 16 # The size of batch images [128]
		train_config["image_size"] = 24 # The size of image to use [33]
		train_config["label_size"] = 96 # The size of label to produce [21]
		train_config["learning_rate"] = 1e-4 #The learning rate of gradient descent algorithm [1e-4]
		train_config["color_dim"] = 3 # Dimension of image color. [1]
		train_config["scale"] = 4 # The size of scale factor for preprocessing input image [3]
		train_config["train_extract_stride"] = 14 #The size of stride to apply input image [14]
		train_config["test_extract_stride"] = train_config["label_size"] #The size of stride to apply input image [14]
		train_config["checkpoint_dir"] = "/home/wei/ML/model/SuperResolution/SR-project-AttWGAN/" #Name of checkpoint directory [checkpoint]
		train_config["log_dir"] = "/home/wei/ML/model/SuperResolution/SR-project-AttWGAN/log/" #Name of checkpoint directory [checkpoint]
		train_config["output_dir"] = "output" # Name of sample directory [output]
		train_config["train_dir"] =  "Train" # Name of train dataset directory
		train_config["test_dir"] = "Test/Set5" # Name of test dataset directory [Test/Set5]
		train_config["h5_dir"] = "/home/wei/ML/dataset/SuperResolution/train" # Name of train dataset .h5 file
		train_config["train_h5_name"] = "train" # Name of train dataset .h5 file
		train_config["test_h5_name"] = "test" # Name of test dataset .h5 file
                     
#		train_config["ckpt_name"] = "RCAN_RSGAN_GP_att_v1_RG_3_RCAB_5_025_10" # Name of checkpoints 0.1 [1,1,1,1] ******************************                               
#		train_config["ckpt_name"] = "RCAN_RSGAN_GP_att_v1_RG_3_RCAB_5_025_20" # Name of checkpoints 0.1 [1,1,1,1] ******************************                                       
		train_config["ckpt_name"] = "RCAN_RSGAN_GP_att_v1_RG_3_RCAB_5_025_50" # Name of checkpoints 0.1 [1,1,1,1] ******************************                                               
                                   
		train_config["is_train"] = True # True for training, False for testing [True]
		train_config["model_ticket"] = "RCAN_RSGAN_GP_att" # Name of checkpoints
		train_config["curr_epoch"] = 0 # Name of checkpoints        
        
		def RCAN_RSGAN_GP_att(self):
						
			mconfig = {}
			
			mconfig["RCAN_RSGAN_GP_att"] = {

										"scale":[1],
										#"subimages":(80, 80, 3), #V1:[96,96]
										"subimages":(40, 40, 3), #V1:[96,96]                                        
										"padding":8,
										"ckpt_file":"/home/wei/ML/model/SuperResolution/SR-project-AttWGAN/Temp/RCAN_RSGAN_GP_att_v1_RG_3_RCAB_5_025_10-10505",
										"isGray": False,
										"isNormallized":True,
										"upsample": False,
										"sub_mean":False,
										"model_config" :{"d_inputs":None, "d_target":None, "scale":2, "feature_size":64, "reuse":False, "is_training":False, "net":"Gen"}
										}
			
			
			return mconfig

		eval_config = self.config["evaluation"]
		eval_config["dataroot"] = '/data/wei/dataset/SuperResolution/eval_input/'        
		eval_config["models"] = [RCAN_RSGAN_GP_att(self)]
		eval_config["summary_file"] = "example_summary.txt"
        
	def EDSR_WGAN_MNIST(self):

		train_config = self.config["train"]

		train_config["mode"] = "small" # Operation mode: normal or freq [normal]
		train_config["epoch"] = 40  # Number of epoch [10]
		train_config["batch_size"] = 64 # The size of batch images [128]
		train_config["image_size"] = 28 # The size of image to use [33]
		train_config["label_size"] = 28 # The size of label to produce [21]
		train_config["learning_rate"] = 1e-4 #The learning rate of gradient descent algorithm [1e-4]
		train_config["color_dim"] = 1 # Dimension of image color. [1]
		train_config["scale"] = 2 # The size of scale factor for preprocessing input image [3]
		train_config["train_extract_stride"] = 14 #The size of stride to apply input image [14]
		train_config["test_extract_stride"] = train_config["label_size"] #The size of stride to apply input image [14]
		train_config["checkpoint_dir"] = "/home/wei/ML/model/SuperResolution/SR-project-prototype/" #Name of checkpoint directory [checkpoint]
		train_config["log_dir"] = "/home/wei/ML/model/SuperResolution/SR-project-prototype/log/" #Name of checkpoint directory [checkpoint]
		train_config["output_dir"] = "output" # Name of sample directory [output]
		train_config["train_dir"] =  "Train" # Name of train dataset directory
		train_config["test_dir"] = "Test/Set5" # Name of test dataset directory [Test/Set5]
		train_config["h5_dir"] = "/home/wei/ML/dataset/SuperResolution/train" # Name of train dataset .h5 file
		train_config["train_h5_name"] = "train" # Name of train dataset .h5 file
		train_config["test_h5_name"] = "test" # Name of test dataset .h5 file
		train_config["ckpt_name"] = "EDSR_WGAN_MNIST_v1" # Name of checkpoints       
		train_config["is_train"] = True # True for training, False for testing [True]
		train_config["model_ticket"] = "EDSR_WGAN_MNIST" # Name of checkpoints
		train_config["curr_epoch"] = 0 # Name of checkpoints        

		def edsr_lsgan(self):
						
			mconfig = {}
			
			mconfig["edsr_lsgan"] = {

										"scale":[1],
										"subimages":[80,80],
										"padding":[8,8],
										#"ckpt_file":"/home/ubuntu/model/model/SR_project/edsr_base_attention_v2_oh/edsr_base_attention_v2_oh-719656",
										"ckpt_file":"/home/wei/ML/model/SR_project/edsr_ls_gan_res4/edsr_ls_gan_res4-103712",
										"isGray": False,
										"isNormallized":True,
										"upsample": False,
										"sub_mean":False,
										"model_config" :{"d_inputs":None, "d_target":None,"scale":2,"feature_size" : 64,"dropout" : 1.0,"feature_size" : 64, "is_training":False, "reuse":False}
										}
			
			
			return mconfig

		

		eval_config = self.config["evaluation"]
		eval_config["dataroot"] = '/home/ubuntu/dataset/SuperResolution/'
		eval_config["test_set"] = ["Set5"]
		eval_config["models"] = [edsr_lsgan(self)]
		eval_config["summary_file"] = "example_summary.txt"        
        
	def EDSR_RaGAN_MNIST(self):

		train_config = self.config["train"]

		train_config["mode"] = "small" # Operation mode: normal or freq [normal]
		train_config["epoch"] = 40  # Number of epoch [10]
		train_config["batch_size"] = 64 # The size of batch images [128]
		train_config["image_size"] = 28 # The size of image to use [33]
		train_config["label_size"] = 28 # The size of label to produce [21]
		train_config["learning_rate"] = 1e-4 #The learning rate of gradient descent algorithm [1e-4]
		train_config["color_dim"] = 1 # Dimension of image color. [1]
		train_config["scale"] = 2 # The size of scale factor for preprocessing input image [3]
		train_config["train_extract_stride"] = 14 #The size of stride to apply input image [14]
		train_config["test_extract_stride"] = train_config["label_size"] #The size of stride to apply input image [14]
		train_config["checkpoint_dir"] = "/home/wei/ML/model/SuperResolution/SR-project-prototype/" #Name of checkpoint directory [checkpoint]
		train_config["log_dir"] = "/home/wei/ML/model/SuperResolution/SR-project-prototype/log/" #Name of checkpoint directory [checkpoint]
		train_config["output_dir"] = "output" # Name of sample directory [output]
		train_config["train_dir"] =  "Train" # Name of train dataset directory
		train_config["test_dir"] = "Test/Set5" # Name of test dataset directory [Test/Set5]
		train_config["h5_dir"] = "/home/wei/ML/dataset/SuperResolution/train" # Name of train dataset .h5 file
		train_config["train_h5_name"] = "train" # Name of train dataset .h5 file
		train_config["test_h5_name"] = "test" # Name of test dataset .h5 file
		train_config["ckpt_name"] = "EDSR_RaGAN_MNIST_v1" # Name of checkpoints       
		train_config["is_train"] = True # True for training, False for testing [True]
		train_config["model_ticket"] = "EDSR_RaGAN_MNIST" # Name of checkpoints
		train_config["curr_epoch"] = 0 # Name of checkpoints        

		def edsr_lsgan(self):
						
			mconfig = {}
			
			mconfig["edsr_lsgan"] = {

										"scale":[1],
										"subimages":[80,80],
										"padding":[8,8],
										#"ckpt_file":"/home/ubuntu/model/model/SR_project/edsr_base_attention_v2_oh/edsr_base_attention_v2_oh-719656",
										"ckpt_file":"/home/wei/ML/model/SR_project/edsr_ls_gan_res4/edsr_ls_gan_res4-103712",
										"isGray": False,
										"isNormallized":True,
										"upsample": False,
										"sub_mean":False,
										"model_config" :{"d_inputs":None, "d_target":None,"scale":2,"feature_size" : 64,"dropout" : 1.0,"feature_size" : 64, "is_training":False, "reuse":False}
										}
			
			
			return mconfig

		

		eval_config = self.config["evaluation"]
		eval_config["dataroot"] = '/home/ubuntu/dataset/SuperResolution/'
		eval_config["test_set"] = ["Set5"]
		eval_config["models"] = [edsr_lsgan(self)]
		eval_config["summary_file"] = "example_summary.txt"           