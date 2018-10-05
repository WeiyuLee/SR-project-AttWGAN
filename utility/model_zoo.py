import tensorflow as tf
import netfactory as nf
import numpy as np

class model_zoo:
    
    def __init__(self, inputs, dropout, is_training, model_ticket):
        
        self.model_ticket = model_ticket
        self.inputs = inputs
        self.dropout = dropout
        self.is_training = is_training

    def EDSR_WGAN(self, kwargs):

        reuse = kwargs["reuse"]
        d_inputs = kwargs["d_inputs"]
        d_target = kwargs["d_target"]
        is_training = kwargs["is_training"]
        net = kwargs["net"]
        
        init = tf.random_normal_initializer(stddev=0.01)

        feature_size = 64
        scaling_factor = 1

        DEPTH = 28
#        DEPTH = 32

        model_params = {

                        'conv1': [3,3,feature_size],
                        'resblock': [3,3,feature_size],
                        'conv2': [3,3,feature_size],
                        'conv3': [3,3,3],
                        'd_output': [3,3,3],
                        
                        'conv1_wgan-gp': [5,5,DEPTH],
                        'conv2_wgan-gp': [5,5,DEPTH*2],
                        'conv3_wgan-gp': [5,5,DEPTH*4],
                        'd_output_wgan-gp': [5,5,3],
                        
#                        # v5-0
                        'conv1_wgan': [5,5,DEPTH],
                        'conv2_wgan': [5,5,DEPTH*2],
                        'conv3_wgan': [5,5,DEPTH*4],
                        'd_output_wgan': [5,5,3],                        
                        'maxpool_wgan': [1, 2, 2, 1],
#                        
#                        # v5-1
#                        'conv1_wgan': [3,3,DEPTH],
#                        'conv2_wgan': [3,3,DEPTH*2],
#                        'conv3_wgan': [3,3,DEPTH*4],
#                        'd_output_wgan': [3,3,3],                       
#                        'maxpool_wgan': [1, 3, 3, 1],
                        
#                        # v5-3
#                        'conv1_wgan': [9,9,DEPTH],
#                        'conv2_wgan': [9,9,DEPTH*2],
#                        'conv3_wgan': [9,9,DEPTH*4],
#                        'd_output_wgan': [9,9,3],                       
#                        'maxpool_wgan': [1, 2, 2, 1],                        
                        
#                        # v5-4
#                        'conv1_wgan': [5,5,DEPTH],
#                        'conv2_wgan': [7,7,DEPTH*2],
#                        'conv3_wgan': [9,9,DEPTH*4],
#                        'd_output_wgan': [5,5,3],                       
#                        'maxpool_wgan': [1, 2, 2, 1],    


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
                if d_model is "EDSR":
                    
                    x = nf.convolution_layer(input_gan, model_params["conv1"], [1,1,1,1], name="conv1",  activat_fn=nf.lrelu, initializer=init)
                    conv_1 = x
                    with tf.variable_scope("resblock", reuse=reuse):                   
                        #Add the residual blocks to the model
                        for i in range(num_resblock):
                            x = nf.resBlock(x,feature_size,scale=scaling_factor, reuse=reuse, idx = i, activation_fn=nf.lrelu, initializer=init)
                        x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2",activat_fn=nf.lrelu, initializer=init)
                        x += conv_1
                        
                    x = nf.convolution_layer(x, model_params["conv1"], [1,1,1,1], name="conv3",  activat_fn=nf.lrelu, initializer=init)
                    d_logits = nf.convolution_layer(x, model_params["d_output"], [1,1,1,1], name="conv4", activat_fn=nf.lrelu, flatten=False, initializer=init)
                    
                elif d_model is "WGAN-GP":
                    
                    x = nf.convolution_layer(input_gan, model_params["conv1_wgan-gp"],    [1,1,1,1], name="conv1_wgan-gp",     activat_fn=nf.lrelu, initializer=init)
                    x = nf.convolution_layer(x,         model_params["conv2_wgan-gp"],    [1,1,1,1], name="conv2_wgan-gp",     activat_fn=nf.lrelu, initializer=init)
                    x = nf.convolution_layer(x,         model_params["conv3_wgan-gp"],    [1,1,1,1], name="conv3_wgan-gp",     activat_fn=nf.lrelu, initializer=init)
                    x = nf.convolution_layer(x,         model_params["d_output_wgan-gp"], [1,1,1,1], name="d_output_wgan-gp",  activat_fn=nf.lrelu, initializer=init)
                    d_logits = x
                
                elif d_model is "PatchWGAN":    

                    x = nf.convolution_layer(input_gan,   model_params["conv1_wgan"],    [1,1,1,1], name="conv1_wgan",     activat_fn=nf.lrelu, initializer=init)
                    
                    pool1 = nf.max_pool_layer(x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv1_wgan_mp")
                    pool1_ = nf.max_pool_layer(-x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv1_wgan_mp")
                    minus_mask = tf.cast(tf.greater(tf.abs(pool1_), pool1), tf.float32)
                    plus_mask = tf.cast(tf.greater(pool1, tf.abs(pool1_)), tf.float32)
                    pool1 = plus_mask*pool1 + minus_mask*(-pool1_)
                    
                    x = nf.convolution_layer(pool1,       model_params["conv2_wgan"],    [1,1,1,1], name="conv2_wgan",     activat_fn=nf.lrelu, initializer=init)

                    pool2 = nf.max_pool_layer(x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv2_wgan_mp")
                    pool2_ = nf.max_pool_layer(-x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv2_wgan_mp")
                    minus_mask = tf.cast(tf.greater(tf.abs(pool2_), pool2), tf.float32)
                    plus_mask = tf.cast(tf.greater(pool2, tf.abs(pool2_)), tf.float32)
                    pool2 = plus_mask*pool2 + minus_mask*(-pool2_)

                    x = nf.convolution_layer(pool2,       model_params["conv3_wgan"],    [1,1,1,1], name="conv3_wgan",     activat_fn=nf.lrelu, initializer=init)
                    
                    pool3 = nf.max_pool_layer(x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv3_wgan_mp")
                    pool3_ = nf.max_pool_layer(-x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv3_wgan_mp")
                    minus_mask = tf.cast(tf.greater(tf.abs(pool3_), pool3), tf.float32)
                    plus_mask = tf.cast(tf.greater(pool3, tf.abs(pool3_)), tf.float32)
                    pool3 = plus_mask*pool3 + minus_mask*(-pool3_)
                    
                    x = nf.convolution_layer(pool3,           model_params["d_output_wgan"], [1,1,1,1], name="d_output_wgan",  activat_fn=nf.lrelu, initializer=init)

                    ### v4
#                    pool4 = nf.max_pool_layer(x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv4_wgan_mp")
#                    pool4_ = nf.max_pool_layer(-x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv4_wgan_mp")
#                    minus_mask = tf.cast(tf.greater(tf.abs(pool4_), pool4), tf.float32)
#                    plus_mask = tf.cast(tf.greater(pool4, tf.abs(pool4_)), tf.float32)
#                    x = plus_mask*pool4 + minus_mask*(-pool4_)

                    d_logits = x

                elif d_model is "PatchWGAN_GP":    

                    patch_size = 16
                    _, image_h, image_w, image_c = input_gan.get_shape().as_list()
                    
                    d_patch_list = []
                    for i in range(0, image_h//patch_size):
                        for j in range(0, image_w//patch_size):    
                            input_patch = input_gan[:, i:i+patch_size, j:j+patch_size, :] 
                            
                            x = nf.convolution_layer(input_patch, model_params["conv1_wgan-gp"],    [1,1,1,1], name="conv1_wgan-gp",     activat_fn=nf.lrelu, initializer=init)
                            x = nf.convolution_layer(x,           model_params["conv2_wgan-gp"],    [1,1,1,1], name="conv2_wgan-gp",     activat_fn=nf.lrelu, initializer=init)        
                            x = nf.convolution_layer(x,           model_params["conv3_wgan-gp"],    [1,1,1,1], name="conv3_wgan-gp",     activat_fn=nf.lrelu, initializer=init)        
                            x = nf.convolution_layer(x,           model_params["d_output_wgan-gp"], [1,1,1,1], name="d_output_wgan-gp",  activat_fn=nf.lrelu, initializer=init)        

                            d_curr_patch = x
                            d_curr_patch = tf.reduce_mean(d_curr_patch, axis=[1,2,3])
                            d_patch_list.append(d_curr_patch)
                            
                    d_patch_stack = tf.stack([d_patch_list[i] for i in range((image_h//patch_size)*(image_w//patch_size))], axis=1)
                    d_patch_weight = d_patch_stack / tf.reduce_sum(tf.abs(d_patch_stack), axis=1, keep_dims=True)
                    d_patch = d_patch_weight*d_patch_stack

                    d_logits = d_patch
                    
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
                output2 = nf.lrelu('d_lrelu_1', output1)
        
                output3 = self.conv2d('d_conv_2', output2, ksize=5, out_dim=2*DEPTH)
                output4 = nf.lrelu('d_lrelu_2', output3)
        
                output5 = self.conv2d('d_conv_3', output4, ksize=5, out_dim=4*DEPTH)
                output6 = nf.lrelu('d_lrelu_3', output5)
        
                # output7 = conv2d('d_conv_4', output6, ksize=5, out_dim=8*DEPTH)
                # output8 = lrelu('d_lrelu_4', output7)
        
                chanel = output6.get_shape().as_list()
                output9 = tf.reshape(output6, [batch_size, chanel[1]*chanel[2]*chanel[3]])
                output0 = self.fully_connected('d_fc', output9, 1)
                
                return output0

    def build_model(self, kwargs = {}):

        model_list = ["googleLeNet_v1", "resNet_v1", "srcnn_v1", "grr_srcnn_v1",
                      "grr_grid_srcnn_v1","edsr_v1", "espcn_v1","edsr_v2",
                      "edsr_attention_v1", "edsr_1X1_v1", "edsr_local_att_v1",
                      "edsr_local_att_v2_upsample", "edsr_attention_v2", "edsr_v2_dual",
                      "edsr_lsgan", "edsr_lsgan_up", "EDSR_WGAN", "EDSR_WGAN_att", "EDSR_WGAN_MNIST"]
        
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