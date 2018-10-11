import tensorflow as tf
import netfactory as nf
import numpy as np

class model_zoo:
    
    def __init__(self, inputs, dropout, is_training, model_ticket):
        
        self.model_ticket = model_ticket
        self.inputs = inputs
        self.dropout = dropout
        self.is_training = is_training

    def EDSR_RaGAN(self, kwargs):

        reuse = kwargs["reuse"]
        d_inputs = kwargs["d_inputs"]
        net = kwargs["net"]
        
        init = tf.random_normal_initializer(stddev=0.01)

        feature_size = 64
        scaling_factor = 1

        DEPTH = 64

        model_params = {

                        # Generator                        
                        'conv1': [3,3,feature_size],
                        'resblock': [3,3,feature_size],
                        'conv2': [3,3,feature_size],
                        'conv3': [3,3,3],
                        #'d_output': [3,3,3*feature_size],
                        'd_output': [3,3,3], #********************************* without Attention, the output depth should be 3

                        # Discriminator                        
                        'conv1_wgan': [5,5,DEPTH],
                        'conv2_wgan': [5,5,DEPTH*2], 
                        'conv3_wgan': [5,5,DEPTH*4], 
                        'conv4_wgan': [5,5,DEPTH*8], 
                        'd_output_wgan': [5,5,1],                                       

                        }

        if net is "Gen":
        
            ### Generator
            num_resblock = 16
                       
            g_input = self.inputs
            
            with tf.variable_scope("EDSR_gen", reuse=reuse):  
               
                x = nf.convolution_layer(g_input, model_params["conv1"], [1,1,1,1], name="conv1", activat_fn=None, initializer=init)
                conv_1 = x
                with tf.variable_scope("resblock",reuse=reuse): 
                
                        #Add the residual blocks to the model
                        for i in range(num_resblock):
                            x = nf.resBlock(x,feature_size,scale=scaling_factor, reuse=reuse, idx = i, initializer=init)
                        x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2", activat_fn=None, initializer=init)
                        x += conv_1
                x = nf.convolution_layer(x, model_params["conv1"], [1,1,1,1], name="conv3",  activat_fn=None, initializer=init)
                g_network = nf.convolution_layer(x, model_params["d_output"], [1,1,1,1], name="conv4", activat_fn=None, initializer=init)

                g_output = tf.nn.sigmoid(g_network)
                           
            return g_output

        elif net is "Dis":
            d_model = kwargs["d_model"]            
            
            ### Discriminator
            num_resblock = 2
            
            input_gan = d_inputs 
            
            with tf.variable_scope("EDSR_dis", reuse=reuse):     

                if d_model is "PatchWGAN_GP":    

                    layer1_1 = nf.convolution_layer(input_gan,    model_params["conv1_wgan"],    [1,1,1,1], name="conv1_wgan_1",     activat_fn=nf.lrelu, initializer=init)
                    layer1_2 = nf.convolution_layer(layer1_1,    model_params["conv1_wgan"],    [1,1,1,1], name="conv1_wgan_2",     activat_fn=nf.lrelu, initializer=init)
                    layer1_3 = nf.convolution_layer(layer1_1 + layer1_2,       model_params["conv1_wgan"],    [1,2,2,1], name="conv1_wgan_3",     activat_fn=nf.lrelu, initializer=init)
#                    layer1_3 = nf.convolution_layer(layer1_1 + layer1_2,       model_params["conv1_wgan"],    [1,1,1,1], name="conv1_wgan_3",     activat_fn=nf.lrelu, initializer=init)
                    
                    layer2_1 = nf.convolution_layer(layer1_3,       model_params["conv2_wgan"],    [1,1,1,1], name="conv2_wgan_1",     activat_fn=nf.lrelu, initializer=init)
                    layer2_2 = nf.convolution_layer(layer2_1,       model_params["conv2_wgan"],    [1,1,1,1], name="conv2_wgan_2",     activat_fn=nf.lrelu, initializer=init)
                    layer2_3 = nf.convolution_layer(layer2_1 + layer2_2,       model_params["conv2_wgan"],    [1,2,2,1], name="conv2_wgan_3",     activat_fn=nf.lrelu, initializer=init)
#                    layer2_3 = nf.convolution_layer(layer2_1 + layer2_2,       model_params["conv2_wgan"],    [1,1,1,1], name="conv2_wgan_3",     activat_fn=nf.lrelu, initializer=init)
                            
                    layer3_1 = nf.convolution_layer(layer2_3,       model_params["conv3_wgan"],    [1,1,1,1], name="conv3_wgan_1",     activat_fn=nf.lrelu, initializer=init)
                    layer3_2 = nf.convolution_layer(layer3_1,       model_params["conv3_wgan"],    [1,1,1,1], name="conv3_wgan_2",     activat_fn=nf.lrelu, initializer=init)
                    layer3_3 = nf.convolution_layer(layer3_1 + layer3_2,       model_params["conv3_wgan"],    [1,2,2,1], name="conv3_wgan_3",     activat_fn=nf.lrelu, initializer=init)
#                    layer3_3 = nf.convolution_layer(layer3_1 + layer3_2,       model_params["conv3_wgan"],    [1,1,1,1], name="conv3_wgan_3",     activat_fn=nf.lrelu, initializer=init)

                    layer4_1 = nf.convolution_layer(layer3_3,       model_params["d_output_wgan"], [1,1,1,1], name="d_output_wgan_1",  activat_fn=nf.lrelu, initializer=init)
                    output = nf.convolution_layer(layer4_1,       model_params["d_output_wgan"], [1,1,1,1], name="d_output_wgan_2",  activat_fn=nf.lrelu, initializer=init)
                            
                    d_logits = output

                    return [d_logits, tf.reduce_mean(layer1_3)]
                
                else:
                    print("d_model parameter error!")
                    
            return d_logits

    def EDSR_WGAN(self, kwargs):

        reuse = kwargs["reuse"]
        d_inputs = kwargs["d_inputs"]
        net = kwargs["net"]
        
        init = tf.random_normal_initializer(stddev=0.01)

        feature_size = 64
        scaling_factor = 1

        DEPTH = 64

        model_params = {

                        # Generator                        
                        'conv1': [3,3,feature_size],
                        'resblock': [3,3,feature_size],
                        'conv2': [3,3,feature_size],
                        'conv3': [3,3,3],
                        #'d_output': [3,3,3*feature_size],
                        'd_output': [3,3,3], #********************************* without Attention, the output depth should be 3

                        # Discriminator                        
                        'conv1_wgan': [5,5,DEPTH],
                        'conv2_wgan': [5,5,DEPTH*2], 
                        'conv3_wgan': [5,5,DEPTH*4], 
                        'conv4_wgan': [5,5,DEPTH*8], 
                        'd_output_wgan': [5,5,1],                                       

                        }

        if net is "Gen":
        
            ### Generator
            num_resblock = 16
                       
            g_input = self.inputs
            
            with tf.variable_scope("EDSR_gen", reuse=reuse):  
               
                x = nf.convolution_layer(g_input, model_params["conv1"], [1,1,1,1], name="conv1", activat_fn=None, initializer=init)
                conv_1 = x
                with tf.variable_scope("resblock",reuse=reuse): 
                
                        #Add the residual blocks to the model
                        for i in range(num_resblock):
                            x = nf.resBlock(x,feature_size,scale=scaling_factor, reuse=reuse, idx = i, initializer=init)
                        x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2", activat_fn=None, initializer=init)
                        x += conv_1
                x = nf.convolution_layer(x, model_params["conv1"], [1,1,1,1], name="conv3",  activat_fn=None, initializer=init)
                g_network = nf.convolution_layer(x, model_params["d_output"], [1,1,1,1], name="conv4", activat_fn=None, initializer=init)

                g_output = tf.nn.sigmoid(g_network)
                           
            return g_output

        elif net is "Dis":
            d_model = kwargs["d_model"]            
            
            ### Discriminator
            num_resblock = 2
            
            input_gan = d_inputs 
            
            with tf.variable_scope("EDSR_dis", reuse=reuse):     

                if d_model is "PatchWGAN_GP":    

                    layer1_1 = nf.convolution_layer(input_gan,    model_params["conv1_wgan"],    [1,1,1,1], name="conv1_wgan_1",     activat_fn=nf.lrelu, initializer=init)
                    layer1_2 = nf.convolution_layer(layer1_1,    model_params["conv1_wgan"],    [1,1,1,1], name="conv1_wgan_2",     activat_fn=nf.lrelu, initializer=init)
                    layer1_3 = nf.convolution_layer(layer1_1 + layer1_2,       model_params["conv1_wgan"],    [1,2,2,1], name="conv1_wgan_3",     activat_fn=nf.lrelu, initializer=init)
#                    layer1_3 = nf.convolution_layer(layer1_1 + layer1_2,       model_params["conv1_wgan"],    [1,1,1,1], name="conv1_wgan_3",     activat_fn=nf.lrelu, initializer=init)
                    
                    layer2_1 = nf.convolution_layer(layer1_3,       model_params["conv2_wgan"],    [1,1,1,1], name="conv2_wgan_1",     activat_fn=nf.lrelu, initializer=init)
                    layer2_2 = nf.convolution_layer(layer2_1,       model_params["conv2_wgan"],    [1,1,1,1], name="conv2_wgan_2",     activat_fn=nf.lrelu, initializer=init)
                    layer2_3 = nf.convolution_layer(layer2_1 + layer2_2,       model_params["conv2_wgan"],    [1,2,2,1], name="conv2_wgan_3",     activat_fn=nf.lrelu, initializer=init)
#                    layer2_3 = nf.convolution_layer(layer2_1 + layer2_2,       model_params["conv2_wgan"],    [1,1,1,1], name="conv2_wgan_3",     activat_fn=nf.lrelu, initializer=init)
                            
                    layer3_1 = nf.convolution_layer(layer2_3,       model_params["conv3_wgan"],    [1,1,1,1], name="conv3_wgan_1",     activat_fn=nf.lrelu, initializer=init)
                    layer3_2 = nf.convolution_layer(layer3_1,       model_params["conv3_wgan"],    [1,1,1,1], name="conv3_wgan_2",     activat_fn=nf.lrelu, initializer=init)
                    layer3_3 = nf.convolution_layer(layer3_1 + layer3_2,       model_params["conv3_wgan"],    [1,2,2,1], name="conv3_wgan_3",     activat_fn=nf.lrelu, initializer=init)
#                    layer3_3 = nf.convolution_layer(layer3_1 + layer3_2,       model_params["conv3_wgan"],    [1,1,1,1], name="conv3_wgan_3",     activat_fn=nf.lrelu, initializer=init)

                    layer4_1 = nf.convolution_layer(layer3_3,       model_params["d_output_wgan"], [1,1,1,1], name="d_output_wgan_1",  activat_fn=nf.lrelu, initializer=init)
                    output = nf.convolution_layer(layer4_1,       model_params["d_output_wgan"], [1,1,1,1], name="d_output_wgan_2",  activat_fn=nf.lrelu, initializer=init)
                            
                    d_logits = output

                    return [d_logits, tf.reduce_mean(layer1_3)]
                
                else:
                    print("d_model parameter error!")
                    
            return d_logits

    def EDSR_WGAN_att(self, kwargs):


        def attention_network(image_input, layers, channels, is_training):

            with tf.variable_scope("attention"):
                    
                    att_net = nf.convolution_layer(image_input, [3,3,64], [1,2,2,1],name="conv1-1")
                    att_net = nf.convolution_layer(att_net, [3,3,64], [1,1,1,1],name="conv1-2")
                    att_net = tf.nn.max_pool(att_net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
                    att_net = nf.convolution_layer(att_net, [3,3,128], [1,1,1,1],name="conv2-1")
                    att_net = nf.convolution_layer(att_net, [3,3,128], [1,1,1,1],name="conv2-2")
                    att_net = tf.nn.max_pool(att_net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
                    att_net = nf.convolution_layer(att_net, [3,3,256], [1,1,1,1],name="conv3-1")
                    att_net = nf.convolution_layer(att_net, [3,3,256], [1,1,1,1],name="conv3-2")
                    att_net = tf.nn.max_pool(att_net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
                    att_net = nf.convolution_layer(att_net, [3,3,512], [1,1,1,1],name="conv4-1")
                    #att_net = nf.convolution_layer(att_net, [3,3,512], [1,1,1,1],name="conv4-2")
                    att_net = tf.nn.max_pool(att_net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
                    bsize, a, b, c = att_net.get_shape().as_list()
                    if bsize == None:
                        bsize = -1
                    att_net = tf.reshape(att_net, [bsize, int(np.prod(att_net.get_shape()[1:]))])                    
                    att_net = nf.fc_layer(att_net, 2048, name="fc1")
                    att_net = nf.fc_layer(att_net, 2048, name="fc2")
                    #att_net = tf.layers.dropout(att_net, rate=dropout, training=is_training, name='dropout1')
                    logits = nf.fc_layer(att_net, channels*layers, name="logits", activat_fn=None)
                    
                    bsize = tf.shape(logits)[0]
                    #logits = tf.reshape(logits, (bsize,1,1,channels*layers))
                    logits = tf.reshape(logits, (bsize,1,1,channels, layers))
                    weighting = tf.nn.softmax(logits)
                    
                    """
                    max_index = tf.argmax(tf.nn.softmax(logits),4) 
                    weighting = tf.one_hot(max_index, 
                                        depth=layers, 
                                        on_value=1.0,
                                        axis = -1)
                    """
                  

            return weighting

        reuse = kwargs["reuse"]
        d_inputs = kwargs["d_inputs"]
        is_training = kwargs["is_training"]
        net = kwargs["net"]
        
        init = tf.random_normal_initializer(stddev=0.01)

        feature_size = 64
        scaling_factor = 1

        DEPTH = 64

        model_params = {

                        # Generator                        
                        'conv1': [3,3,feature_size],
                        'resblock': [3,3,feature_size],
                        'conv2': [3,3,feature_size],
                        'conv3': [3,3,3],
                        'd_output': [3,3,3*feature_size],

                        # Discriminator                        
                        'conv1_wgan': [5,5,DEPTH],
                        'conv2_wgan': [5,5,DEPTH*2], 
                        'conv3_wgan': [5,5,DEPTH*4], 
                        'conv4_wgan': [5,5,DEPTH*8], 
                        'd_output_wgan': [5,5,1],                                       

                        }

        if net is "Gen":
        
            ### Generator
            num_resblock = 16
                       
            g_input = self.inputs
            
            with tf.variable_scope("EDSR_gen", reuse=reuse):  

                with tf.name_scope("attention_x2"):
                    att_layers = feature_size
                    arr = attention_network(self.inputs, att_layers,3, is_training)
               
                x = nf.convolution_layer(g_input, model_params["conv1"], [1,1,1,1], name="conv1", activat_fn=None, initializer=init)
                conv_1 = x
                with tf.variable_scope("resblock",reuse=reuse): 
                
                        #Add the residual blocks to the model
                        for i in range(num_resblock):
                            x = nf.resBlock(x,feature_size,scale=scaling_factor, reuse=reuse, idx = i, initializer=init)
                        x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2", activat_fn=None, initializer=init)
                        x += conv_1
                x = nf.convolution_layer(x, model_params["conv1"], [1,1,1,1], name="conv3",  activat_fn=None, initializer=init)
                g_network = nf.convolution_layer(x, model_params["d_output"], [1,1,1,1], name="conv4", activat_fn=None, initializer=init)
                
                #Attention
                bsize, a, b, c = g_network.get_shape().as_list()
                g_network = tf.reshape(g_network, (-1, a, b, 3, att_layers))
                g_network = tf.multiply(g_network, arr)
                g_network = tf.reduce_sum(g_network,4)

                g_output = tf.nn.sigmoid(g_network)
                           
            return g_output

        elif net is "Dis":
            d_model = kwargs["d_model"]            
            
            ### Discriminator
            num_resblock = 2
            
            input_gan = d_inputs 
            
            with tf.variable_scope("EDSR_dis", reuse=reuse):     

                if d_model is "PatchWGAN_GP":    

                    layer1_1 = nf.convolution_layer(input_gan,    model_params["conv1_wgan"],    [1,1,1,1], name="conv1_wgan_1",     activat_fn=nf.lrelu, initializer=init)
                    layer1_2 = nf.convolution_layer(layer1_1,    model_params["conv1_wgan"],    [1,1,1,1], name="conv1_wgan_2",     activat_fn=nf.lrelu, initializer=init)
                    layer1_3 = nf.convolution_layer(layer1_1 + layer1_2,       model_params["conv1_wgan"],    [1,2,2,1], name="conv1_wgan_3",     activat_fn=nf.lrelu, initializer=init)
#                    layer1_3 = nf.convolution_layer(layer1_1 + layer1_2,       model_params["conv1_wgan"],    [1,1,1,1], name="conv1_wgan_3",     activat_fn=nf.lrelu, initializer=init)
                    
                    layer2_1 = nf.convolution_layer(layer1_3,       model_params["conv2_wgan"],    [1,1,1,1], name="conv2_wgan_1",     activat_fn=nf.lrelu, initializer=init)
                    layer2_2 = nf.convolution_layer(layer2_1,       model_params["conv2_wgan"],    [1,1,1,1], name="conv2_wgan_2",     activat_fn=nf.lrelu, initializer=init)
                    layer2_3 = nf.convolution_layer(layer2_1 + layer2_2,       model_params["conv2_wgan"],    [1,2,2,1], name="conv2_wgan_3",     activat_fn=nf.lrelu, initializer=init)
#                    layer2_3 = nf.convolution_layer(layer2_1 + layer2_2,       model_params["conv2_wgan"],    [1,1,1,1], name="conv2_wgan_3",     activat_fn=nf.lrelu, initializer=init)
                            
                    layer3_1 = nf.convolution_layer(layer2_3,       model_params["conv3_wgan"],    [1,1,1,1], name="conv3_wgan_1",     activat_fn=nf.lrelu, initializer=init)
                    layer3_2 = nf.convolution_layer(layer3_1,       model_params["conv3_wgan"],    [1,1,1,1], name="conv3_wgan_2",     activat_fn=nf.lrelu, initializer=init)
                    layer3_3 = nf.convolution_layer(layer3_1 + layer3_2,       model_params["conv3_wgan"],    [1,2,2,1], name="conv3_wgan_3",     activat_fn=nf.lrelu, initializer=init)
#                    layer3_3 = nf.convolution_layer(layer3_1 + layer3_2,       model_params["conv3_wgan"],    [1,1,1,1], name="conv3_wgan_3",     activat_fn=nf.lrelu, initializer=init)

                    layer4_1 = nf.convolution_layer(layer3_3,       model_params["d_output_wgan"], [1,1,1,1], name="d_output_wgan_1",  activat_fn=nf.lrelu, initializer=init)
                    output = nf.convolution_layer(layer4_1,       model_params["d_output_wgan"], [1,1,1,1], name="d_output_wgan_2",  activat_fn=nf.lrelu, initializer=init)
                            
                    d_logits = output

                    return [d_logits, tf.reduce_mean(layer1_3)]
                
                else:
                    print("d_model parameter error!")
                    
            return d_logits

    def RCAN_WGAN_att(self, kwargs):

        reuse = kwargs["reuse"]
        d_inputs = kwargs["d_inputs"]
        net = kwargs["net"]
        
        init = tf.random_normal_initializer(stddev=0.01)

        feature_size = 64

        DEPTH = 64

        model_params = {

                        # Generator                        
                        'conv1': [3,3,feature_size],
                        'output_conv': [3,3,3],

                        # Discriminator                        
                        'conv1_wgan': [5,5,DEPTH],
                        'conv2_wgan': [5,5,DEPTH*2], 
                        'conv3_wgan': [5,5,DEPTH*4], 
                        'conv4_wgan': [5,5,DEPTH*8], 
                        'd_output_wgan': [5,5,1],                                       

                        }
        
        # CA
        def channel_attention(image_input, initializer, name, shrink_ratio=0.25):

            _,_ , _, c = image_input.get_shape().as_list()
        
            with tf.variable_scope("CA"):
               att_net = tf.reduce_mean(image_input, axis=[1,2], keep_dims=True)
               att_net = nf.convolution_layer(att_net, [1,1,int(c*shrink_ratio)], [1,1,1,1], name=name+"_down_scaling", activat_fn=tf.nn.relu, initializer=initializer)
               att_net = nf.convolution_layer(att_net, [1,1,c], [1,1,1,1], name=name+"_up_scaling", activat_fn=tf.nn.sigmoid, initializer=initializer)
               layer_output = tf.multiply(image_input, att_net, name=name+"output")
               return layer_output                        
            
        # RCAB
        def residual_channel_attention_block(image_input, initializer, name):
            
            with tf.variable_scope("RCAB_"+name):
                x = nf.convolution_layer(image_input, model_params["conv1"], [1,1,1,1], name=name+"_conv1", activat_fn=tf.nn.relu, initializer=initializer)
                x = nf.convolution_layer(x,           model_params["conv1"], [1,1,1,1], name=name+"_conv2", activat_fn=None, initializer=initializer)

                CA_output = channel_attention(x, initializer=initializer, name="CA")

                RCAB_output = tf.add(image_input, CA_output)
                
                return RCAB_output

        # RG
        def residual_group(image_input, initializer, name, RCAB_num=5):
            
            with tf.variable_scope("RG_"+name):
                
                x = image_input
                
                for i in range(RCAB_num):
                    x = residual_channel_attention_block(x, initializer=initializer, name=str(i))

                x = nf.convolution_layer(x, model_params["conv1"], [1,1,1,1], name=name+"_conv", activat_fn=None, initializer=initializer)    
                RG_output = tf.add(x, image_input)
                
                return RG_output

        # RIR
        def residual_in_residual(image_input, initializer, name, RG_num=3):
            
            with tf.variable_scope("RIR"):
                
                x = image_input
                
                for i in range(RG_num):
                    x = residual_group(x, initializer=initializer, name=str(i))

                x = nf.convolution_layer(x, model_params["conv1"], [1,1,1,1], name=name+"_conv", activat_fn=None, initializer=initializer)    
                RIR_output = tf.add(x, image_input)
                
                return RIR_output
                
        if net is "Gen":
        
            ### Generator
            with tf.variable_scope("RCAN_gen", reuse=reuse):     
                input_conv_output = nf.convolution_layer(self.inputs, model_params["conv1"], [1,1,1,1], name="input_conv", activat_fn=None, initializer=init)    

                RIR_output = residual_in_residual(input_conv_output, initializer=init, name="RIR")

                output_conv_output = nf.convolution_layer(RIR_output, model_params["output_conv"], [1,1,1,1], name="output_conv", activat_fn=None, initializer=init)                

                return output_conv_output

        elif net is "Dis":
            d_model = kwargs["d_model"]            
            
            ### Discriminator
            
            input_gan = d_inputs 
            
            with tf.variable_scope("RCAN_dis", reuse=reuse):     

                if d_model is "PatchWGAN_GP":    

                    layer1_1 = nf.convolution_layer(input_gan,    model_params["conv1_wgan"],    [1,1,1,1], name="conv1_wgan_1",     activat_fn=nf.lrelu, initializer=init)
                    layer1_2 = nf.convolution_layer(layer1_1,    model_params["conv1_wgan"],    [1,1,1,1], name="conv1_wgan_2",     activat_fn=nf.lrelu, initializer=init)
                    layer1_3_before = nf.convolution_layer(layer1_1 + layer1_2,       model_params["conv1_wgan"],    [1,2,2,1], name="conv1_wgan_3",     activat_fn=None, initializer=init)
                    layer1_3 = nf.lrelu(layer1_3_before)
                    
                    layer2_1 = nf.convolution_layer(layer1_3,       model_params["conv2_wgan"],    [1,1,1,1], name="conv2_wgan_1",     activat_fn=nf.lrelu, initializer=init)
                    layer2_2 = nf.convolution_layer(layer2_1,       model_params["conv2_wgan"],    [1,1,1,1], name="conv2_wgan_2",     activat_fn=nf.lrelu, initializer=init)
                    layer2_3 = nf.convolution_layer(layer2_1 + layer2_2,       model_params["conv2_wgan"],    [1,2,2,1], name="conv2_wgan_3",     activat_fn=nf.lrelu, initializer=init)
#                    layer2_3 = nf.convolution_layer(layer2_1 + layer2_2,       model_params["conv2_wgan"],    [1,1,1,1], name="conv2_wgan_3",     activat_fn=nf.lrelu, initializer=init)
                            
                    layer3_1 = nf.convolution_layer(layer2_3,       model_params["conv3_wgan"],    [1,1,1,1], name="conv3_wgan_1",     activat_fn=nf.lrelu, initializer=init)
                    layer3_2 = nf.convolution_layer(layer3_1,       model_params["conv3_wgan"],    [1,1,1,1], name="conv3_wgan_2",     activat_fn=nf.lrelu, initializer=init)
                    layer3_3 = nf.convolution_layer(layer3_1 + layer3_2,       model_params["conv3_wgan"],    [1,2,2,1], name="conv3_wgan_3",     activat_fn=nf.lrelu, initializer=init)
#                    layer3_3 = nf.convolution_layer(layer3_1 + layer3_2,       model_params["conv3_wgan"],    [1,1,1,1], name="conv3_wgan_3",     activat_fn=nf.lrelu, initializer=init)

                    layer4_1 = nf.convolution_layer(layer3_3,       model_params["d_output_wgan"], [1,1,1,1], name="d_output_wgan_1",  activat_fn=nf.lrelu, initializer=init)
                    output = nf.convolution_layer(layer4_1,       model_params["d_output_wgan"], [1,1,1,1], name="d_output_wgan_2",  activat_fn=nf.lrelu, initializer=init)
                            
                    d_logits = output

                    return [d_logits, tf.reduce_mean(layer1_3_before)]

                elif d_model is "PatchWGAN_GP_att":

                    layer1_1 = nf.convolution_layer(input_gan,    model_params["conv1_wgan"],    [1,1,1,1], name="conv1_wgan_1",     activat_fn=nf.lrelu, initializer=init)
                    layer1_2 = nf.convolution_layer(layer1_1,    model_params["conv1_wgan"],    [1,1,1,1], name="conv1_wgan_2",     activat_fn=nf.lrelu, initializer=init)
                    layer1_3_before = nf.convolution_layer(layer1_1 + layer1_2,       model_params["conv1_wgan"],    [1,2,2,1], name="conv1_wgan_3",     activat_fn=None, initializer=init)
                    layer1_3 = nf.lrelu(layer1_3_before)
                    layer1_att = nf.attention_RCAN(layer1_3, init, name="conv1_att")

                    layer2_1 = nf.convolution_layer(layer1_att,       model_params["conv2_wgan"],    [1,1,1,1], name="conv2_wgan_1",     activat_fn=nf.lrelu, initializer=init)
                    layer2_2 = nf.convolution_layer(layer2_1,       model_params["conv2_wgan"],    [1,1,1,1], name="conv2_wgan_2",     activat_fn=nf.lrelu, initializer=init)
                    layer2_3 = nf.convolution_layer(layer2_1 + layer2_2,       model_params["conv2_wgan"],    [1,2,2,1], name="conv2_wgan_3",     activat_fn=nf.lrelu, initializer=init)
                    layer2_att = nf.attention_RCAN(layer2_3, init, name="conv2_att")

                    layer3_1 = nf.convolution_layer(layer2_att,       model_params["conv3_wgan"],    [1,1,1,1], name="conv3_wgan_1",     activat_fn=nf.lrelu, initializer=init)
                    layer3_2 = nf.convolution_layer(layer3_1,       model_params["conv3_wgan"],    [1,1,1,1], name="conv3_wgan_2",     activat_fn=nf.lrelu, initializer=init)
                    layer3_3 = nf.convolution_layer(layer3_1 + layer3_2,       model_params["conv3_wgan"],    [1,2,2,1], name="conv3_wgan_3",     activat_fn=nf.lrelu, initializer=init)
                    layer3_att = nf.attention_RCAN(layer3_3, init, name="conv3_att")

                    layer4_1 = nf.convolution_layer(layer3_att,       model_params["d_output_wgan"], [1,1,1,1], name="d_output_wgan_1",  activat_fn=nf.lrelu, initializer=init)
                    output = nf.convolution_layer(layer4_1,       model_params["d_output_wgan"], [1,1,1,1], name="d_output_wgan_2",  activat_fn=nf.lrelu, initializer=init)

                    d_logits = output

                    return [d_logits, tf.reduce_mean(layer1_3_before)]
                
                else:
                    print("d_model parameter error!")
                    
            return d_logits

    def RCAN_RSGAN_GP_att(self, kwargs):

        reuse = kwargs["reuse"]
        d_inputs = kwargs["d_inputs"]
        net = kwargs["net"]
        
        init = tf.random_normal_initializer(stddev=0.01)

        feature_size = 64

        DEPTH = 64

        model_params = {

                        # Generator                        
                        'conv1': [3,3,feature_size],
                        'output_conv': [3,3,3],

                        # Discriminator                        
                        'conv1_wgan': [5,5,DEPTH],
                        'conv2_wgan': [5,5,DEPTH*2], 
                        'conv3_wgan': [5,5,DEPTH*4], 
                        'conv4_wgan': [5,5,DEPTH*8], 
                        'd_output_wgan': [5,5,1],                                       

                        }
        
        # CA
        def channel_attention(image_input, initializer, name, shrink_ratio=0.25):

            _,_ , _, c = image_input.get_shape().as_list()
        
            with tf.variable_scope("CA"):
               att_net = tf.reduce_mean(image_input, axis=[1,2], keep_dims=True)
               att_net = nf.convolution_layer(att_net, [1,1,int(c*shrink_ratio)], [1,1,1,1], name=name+"_down_scaling", activat_fn=tf.nn.relu, initializer=initializer)
               att_net = nf.convolution_layer(att_net, [1,1,c], [1,1,1,1], name=name+"_up_scaling", activat_fn=tf.nn.sigmoid, initializer=initializer)
               layer_output = tf.multiply(image_input, att_net, name=name+"output")
               return layer_output                        
            
        # RCAB
        def residual_channel_attention_block(image_input, initializer, name):
            
            with tf.variable_scope("RCAB_"+name):
                x = nf.convolution_layer(image_input, model_params["conv1"], [1,1,1,1], name=name+"_conv1", activat_fn=tf.nn.relu, initializer=initializer)
                x = nf.convolution_layer(x,           model_params["conv1"], [1,1,1,1], name=name+"_conv2", activat_fn=None, initializer=initializer)

                CA_output = channel_attention(x, initializer=initializer, name="CA")

                RCAB_output = tf.add(image_input, CA_output)
                
                return RCAB_output

        # RG
        def residual_group(image_input, initializer, name, RCAB_num=5):
            
            with tf.variable_scope("RG_"+name):
                
                x = image_input
                
                for i in range(RCAB_num):
                    x = residual_channel_attention_block(x, initializer=initializer, name=str(i))

                x = nf.convolution_layer(x, model_params["conv1"], [1,1,1,1], name=name+"_conv", activat_fn=None, initializer=initializer)    
                RG_output = tf.add(x, image_input)
                
                return RG_output

        # RIR
        def residual_in_residual(image_input, initializer, name, RG_num=3):
            
            with tf.variable_scope("RIR"):
                
                x = image_input
                
                for i in range(RG_num):
                    x = residual_group(x, initializer=initializer, name=str(i))

                x = nf.convolution_layer(x, model_params["conv1"], [1,1,1,1], name=name+"_conv", activat_fn=None, initializer=initializer)    
                RIR_output = tf.add(x, image_input)
                
                return RIR_output
                
        if net is "Gen":
        
            ### Generator
            with tf.variable_scope("RCAN_gen", reuse=reuse):     
                input_conv_output = nf.convolution_layer(self.inputs, model_params["conv1"], [1,1,1,1], name="input_conv", activat_fn=None, initializer=init)    

                RIR_output = residual_in_residual(input_conv_output, initializer=init, name="RIR")

                output_conv_output = nf.convolution_layer(RIR_output, model_params["output_conv"], [1,1,1,1], name="output_conv", activat_fn=None, initializer=init)                

                return output_conv_output

        elif net is "Dis":
            d_model = kwargs["d_model"]            
            
            ### Discriminator
            num_resblock = 2
            
            input_gan = d_inputs 
            
            with tf.variable_scope("RCAN_dis", reuse=reuse):     

                if d_model is "PatchWGAN_GP":    

                    layer1_1 = nf.convolution_layer(input_gan,    model_params["conv1_wgan"],    [1,1,1,1], name="conv1_wgan_1",     activat_fn=nf.lrelu, initializer=init)
                    layer1_2 = nf.convolution_layer(layer1_1,    model_params["conv1_wgan"],    [1,1,1,1], name="conv1_wgan_2",     activat_fn=nf.lrelu, initializer=init)
                    layer1_3 = nf.convolution_layer(layer1_1 + layer1_2,       model_params["conv1_wgan"],    [1,2,2,1], name="conv1_wgan_3",     activat_fn=nf.lrelu, initializer=init)
#                    layer1_3 = nf.convolution_layer(layer1_1 + layer1_2,       model_params["conv1_wgan"],    [1,1,1,1], name="conv1_wgan_3",     activat_fn=nf.lrelu, initializer=init)
                    
                    layer2_1 = nf.convolution_layer(layer1_3,       model_params["conv2_wgan"],    [1,1,1,1], name="conv2_wgan_1",     activat_fn=nf.lrelu, initializer=init)
                    layer2_2 = nf.convolution_layer(layer2_1,       model_params["conv2_wgan"],    [1,1,1,1], name="conv2_wgan_2",     activat_fn=nf.lrelu, initializer=init)
                    layer2_3 = nf.convolution_layer(layer2_1 + layer2_2,       model_params["conv2_wgan"],    [1,2,2,1], name="conv2_wgan_3",     activat_fn=nf.lrelu, initializer=init)
#                    layer2_3 = nf.convolution_layer(layer2_1 + layer2_2,       model_params["conv2_wgan"],    [1,1,1,1], name="conv2_wgan_3",     activat_fn=nf.lrelu, initializer=init)
                            
                    layer3_1 = nf.convolution_layer(layer2_3,       model_params["conv3_wgan"],    [1,1,1,1], name="conv3_wgan_1",     activat_fn=nf.lrelu, initializer=init)
                    layer3_2 = nf.convolution_layer(layer3_1,       model_params["conv3_wgan"],    [1,1,1,1], name="conv3_wgan_2",     activat_fn=nf.lrelu, initializer=init)
                    layer3_3 = nf.convolution_layer(layer3_1 + layer3_2,       model_params["conv3_wgan"],    [1,2,2,1], name="conv3_wgan_3",     activat_fn=nf.lrelu, initializer=init)
#                    layer3_3 = nf.convolution_layer(layer3_1 + layer3_2,       model_params["conv3_wgan"],    [1,1,1,1], name="conv3_wgan_3",     activat_fn=nf.lrelu, initializer=init)

                    layer4_1 = nf.convolution_layer(layer3_3,       model_params["d_output_wgan"], [1,1,1,1], name="d_output_wgan_1",  activat_fn=nf.lrelu, initializer=init)
                    output = nf.convolution_layer(layer4_1,       model_params["d_output_wgan"], [1,1,1,1], name="d_output_wgan_2",  activat_fn=nf.lrelu, initializer=init)
                            
                    d_logits = output

                    return [d_logits, tf.reduce_mean(layer1_3)]

                else:
                    print("d_model parameter error!")
                    
            return d_logits

    def EDSR_WGAN_MNIST(self, kwargs):

        reuse = kwargs["reuse"]
        d_inputs = kwargs["d_inputs"]
        net = kwargs["net"]
        
        DEPTH = 28
        OUTPUT_SIZE = 28
        batch_size = 64

        if net is "Gen":
        
            ###Generator
            g_input = self.inputs
            with tf.variable_scope("EDSR_gen", reuse=reuse):     

                noise = g_input

                output = self.fully_connected('g_fc_1', noise, 2*2*8*DEPTH)
                output = tf.reshape(output, [batch_size, 2, 2, 8*DEPTH], 'g_conv')
        
                output = self.deconv2d('g_deconv_1', output, ksize=5, outshape=[batch_size, 4, 4, 4*DEPTH])
                output = tf.nn.relu(output)
                output = tf.reshape(output, [batch_size, 4, 4, 4*DEPTH])
        
                output = self.deconv2d('g_deconv_2', output, ksize=5, outshape=[batch_size, 7, 7, 2* DEPTH])
                output = tf.nn.relu(output)
        
                output = self.deconv2d('g_deconv_3', output, ksize=5, outshape=[batch_size, 14, 14, DEPTH])
                output = tf.nn.relu(output)
        
                output = self.deconv2d('g_deconv_4', output, ksize=5, outshape=[batch_size, OUTPUT_SIZE, OUTPUT_SIZE, 1])
                # output = tf.nn.relu(output)
                output = tf.nn.sigmoid(output)
                
                return tf.reshape(output,[-1,784])           


        elif net is "Dis":
            
            ###Discriminator

            input_gan = d_inputs 
            with tf.variable_scope("EDSR_dis", reuse=reuse):     

                output = tf.reshape(input_gan, [-1, 28, 28, 1])
                output1 = self.conv2d('d_conv_1', output, ksize=5, out_dim=DEPTH)
                output2 = nf.lrelu(output1, 'd_lrelu_1')
        
                output3 = self.conv2d('d_conv_2', output2, ksize=5, out_dim=2*DEPTH)
                output4 = nf.lrelu(output3, 'd_lrelu_2')
        
                output5 = self.conv2d('d_conv_3', output4, ksize=5, out_dim=4*DEPTH)
                output6 = nf.lrelu(output5, 'd_lrelu_3')
        
                # output7 = conv2d('d_conv_4', output6, ksize=5, out_dim=8*DEPTH)
                # output8 = lrelu(output7, 'd_lrelu_4')
        
                chanel = output6.get_shape().as_list()
                output9 = tf.reshape(output6, [batch_size, chanel[1]*chanel[2]*chanel[3]])
                output0 = self.fully_connected('d_fc', output9, 1)
                
                return output0
            
    def EDSR_RaGAN_MNIST(self, kwargs):

        reuse = kwargs["reuse"]
        d_inputs = kwargs["d_inputs"]
        net = kwargs["net"]
        
        DEPTH = 28
        OUTPUT_SIZE = 28
        batch_size = 64

        if net is "Gen":
        
            ###Generator
            g_input = self.inputs
            with tf.variable_scope("EDSR_gen", reuse=reuse):     

                noise = g_input

                output = self.fully_connected('g_fc_1', noise, 2*2*8*DEPTH)
                output = tf.reshape(output, [batch_size, 2, 2, 8*DEPTH], 'g_conv')
        
                output = self.deconv2d('g_deconv_1', output, ksize=5, outshape=[batch_size, 4, 4, 4*DEPTH])
                output = tf.nn.relu(output)
                output = tf.reshape(output, [batch_size, 4, 4, 4*DEPTH])
        
                output = self.deconv2d('g_deconv_2', output, ksize=5, outshape=[batch_size, 7, 7, 2* DEPTH])
                output = tf.nn.relu(output)
        
                output = self.deconv2d('g_deconv_3', output, ksize=5, outshape=[batch_size, 14, 14, DEPTH])
                output = tf.nn.relu(output)
        
                output = self.deconv2d('g_deconv_4', output, ksize=5, outshape=[batch_size, OUTPUT_SIZE, OUTPUT_SIZE, 1])

                output = tf.nn.sigmoid(output)
                #output = tf.nn.tanh(output)
                
                return tf.reshape(output,[-1,784])           

        elif net is "Dis":
            
            ###Discriminator
            input_gan = d_inputs 
            with tf.variable_scope("EDSR_dis", reuse=reuse):     

                output = tf.reshape(input_gan, [-1, 28, 28, 1])
                output1 = self.conv2d('d_conv_1', output, ksize=5, out_dim=DEPTH)
                output2 = nf.lrelu(output1, 'd_lrelu_1')
        
                output3 = self.conv2d('d_conv_2', output2, ksize=5, out_dim=2*DEPTH)
                output4 = nf.lrelu(output3, 'd_lrelu_2')
        
                output5 = self.conv2d('d_conv_3', output4, ksize=5, out_dim=4*DEPTH)
                output6 = nf.lrelu(output5, 'd_lrelu_3')
        
                # output7 = conv2d('d_conv_4', output6, ksize=5, out_dim=8*DEPTH)
                # output8 = lrelu(output7, 'd_lrelu_4')
        
                chanel = output6.get_shape().as_list()
                output9 = tf.reshape(output6, [batch_size, chanel[1]*chanel[2]*chanel[3]])
                output0 = self.fully_connected('d_fc', output9, 1)
                
                return output0            

    def build_model(self, kwargs = {}):

        model_list = ["EDSR_RaGAN", "EDSR_WGAN", "EDSR_WGAN_att", "RCAN_WGAN_att", "RCAN_RSGAN_GP_att", "EDSR_WGAN_MNIST", "EDSR_RaGAN_MNIST"]
        
        if self.model_ticket not in model_list:
            print("sorry, wrong ticket!")
            return 0
        
        else:
           
            fn = getattr(self,self.model_ticket)
            
            if kwargs == {}:
                netowrk = fn()
            else:
                netowrk = fn(kwargs)
            return netowrk
        
    def conv2d(self, name, tensor,ksize, out_dim, stddev=0.01, stride=2, padding='SAME'):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [ksize, ksize, tensor.get_shape()[-1],out_dim], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=stddev))
            var = tf.nn.conv2d(tensor,w,[1,stride, stride,1],padding=padding)
            b = tf.get_variable('b', [out_dim], 'float32',initializer=tf.constant_initializer(0.01))
            return tf.nn.bias_add(var, b)
    
    def deconv2d(self, name, tensor, ksize, outshape, stddev=0.01, stride=2, padding='SAME'):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [ksize, ksize, outshape[-1], tensor.get_shape()[-1]], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=stddev))
            var = tf.nn.conv2d_transpose(tensor, w, outshape, strides=[1, stride, stride, 1], padding=padding)
            b = tf.get_variable('b', [outshape[-1]], 'float32', initializer=tf.constant_initializer(0.01))
            return tf.nn.bias_add(var, b)        

    def fully_connected(self, name,value, output_shape):
        with tf.variable_scope(name, reuse=None) as scope:
            shape = value.get_shape().as_list()
            w = tf.get_variable('w', [shape[1], output_shape], dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=0.01))
            b = tf.get_variable('b', [output_shape], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
    
            return tf.matmul(value, w) + b
        
def unit_test():

    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
    is_training = tf.placeholder(tf.bool, name='is_training')
    dropout = tf.placeholder(tf.float32, name='dropout')
    
    mz = model_zoo(x, dropout, is_training,"resNet_v1")
    return mz.build_model()
    

#m = unit_test()