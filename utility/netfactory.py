import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np

import sys
sys.path.append('./utility')
#import cifar10
import utility as ut


def lrelu(x, name = "leaky", alpha = 0.2):

#    with tf.variable_scope(name):
#        leaky = tf.nn.relu(x) - alpha * tf.nn.relu(-x)
#    return leaky

    return tf.maximum(x, alpha * x, name=name)

#def lrelu(name,x, leak=0.2):
#    return tf.maximum(x, leak * x, name=name)

def batchnorm(input, index = 0, reuse = False):
    with tf.variable_scope("batchnorm_{}".format(index), reuse = reuse):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
    return normalized

def convolution_layer(inputs, kernel_shape, stride, name, flatten = False ,padding = 'SAME',initializer=tf.contrib.layers.xavier_initializer(), activat_fn=tf.nn.relu, is_bn=False):
                                                                                            #initializer=tf.contrib.layers.xavier_initializer()
    pre_shape = inputs.get_shape()[-1]
    rkernel_shape = [kernel_shape[0], kernel_shape[1], pre_shape, kernel_shape[2]]     
    
    with tf.variable_scope(name) as scope:
        
        try:
            weight = tf.get_variable("weights",rkernel_shape, tf.float32, initializer=initializer)
            bias = tf.get_variable("bias",kernel_shape[2], tf.float32, initializer=tf.zeros_initializer())
        except:
            scope.reuse_variables()
            weight = tf.get_variable("weights",rkernel_shape, tf.float32, initializer=initializer)
            bias = tf.get_variable("bias",kernel_shape[2], tf.float32, initializer=tf.zeros_initializer())
        
        net = tf.nn.conv2d(inputs, weight,stride, padding=padding)
        net = tf.add(net, bias)

        if is_bn:
            net = batchnorm(net)
        else:
            net = net
        
        if not activat_fn==None:
            net = activat_fn(net, name=name+"_out")
        
        if flatten == True:
            net = tf.reshape(net, [-1, int(np.prod(net.get_shape()[1:]))], name=name+"_flatout")
        
    return net

def max_pool_layer(inputs, kernel_shape, stride, name=None, padding='VALID'):
           
    return tf.nn.max_pool(inputs, kernel_shape, stride, padding, name=name)

def fc_layer(inputs, out_shape, name,initializer=tf.contrib.layers.xavier_initializer(), activat_fn=tf.nn.relu):
    
    
    pre_shape = inputs.get_shape()[-1]
    
    with tf.variable_scope(name) as scope:
        
        
        try:
            weight = tf.get_variable("weights",[pre_shape, out_shape], tf.float32, initializer=initializer)
            bias = tf.get_variable("bias",out_shape, tf.float32, initializer=initializer)
        except:
            scope.reuse_variables()
            weight = tf.get_variable("weights",[pre_shape, out_shape], tf.float32, initializer=initializer)
            bias = tf.get_variable("bias",out_shape, tf.float32, initializer=initializer)
        
        
        if activat_fn != None:
            net = activat_fn(tf.nn.xw_plus_b(inputs, weight, bias, name=name + '_out'))
        else:
            net = tf.nn.xw_plus_b(inputs, weight, bias, name=name)
        
    return net

def inception_v1(inputs, module_shape, name, flatten = False ,padding = 'SAME',initializer=tf.contrib.layers.xavier_initializer(), activat_fn=tf.nn.relu):
    
    
    
    with tf.variable_scope(name):
        
            with tf.variable_scope("1x1"):
                
                kernel_shape = module_shape[name]["1x1"]
                net1x1 = convolution_layer(inputs, [1,1,kernel_shape], [1,1,1,1], name="conv1x1", padding=padding, initializer=initializer, activat_fn = activat_fn)
                
            with tf.variable_scope("3x3"):
                
                kernel_shape = module_shape[name]["3x3"]["1x1"]
                net3x3 = convolution_layer(inputs, [1,1,kernel_shape], [1,1,1,1], name="conv1x1", padding=padding, initializer=initializer, activat_fn = activat_fn)
                kernel_shape = module_shape[name]["3x3"]["3x3"]
                net3x3 = convolution_layer(net3x3, [3,3,kernel_shape], [1,1,1,1], name="conv3x3", padding=padding, initializer=initializer, activat_fn = activat_fn)
                
            with tf.variable_scope("5x5"):
                
                kernel_shape = module_shape[name]["5x5"]["1x1"]
                net5x5 = convolution_layer(inputs, [1,1,kernel_shape], [1,1,1,1], name="conv1x1", padding=padding, initializer=initializer, activat_fn = activat_fn)
                kernel_shape = module_shape[name]["5x5"]["5x5"]
                net5x5 = convolution_layer(net5x5, [5,5,kernel_shape], [1,1,1,1], name="conv5x5", padding=padding, initializer=initializer, activat_fn = activat_fn)
                
            with tf.variable_scope("s1x1"):
                            
                kernel_shape = module_shape[name]["s1x1"]
                net_s1x1 = tf.nn.max_pool(inputs, ksize=[1, 3, 3, 1],strides=[1, 1, 1, 1], padding=padding, name = "maxpool_s1x1")
                net_s1x1 = convolution_layer(net_s1x1, [1,1,kernel_shape], [1,1,1,1], name="conv_s1x1", padding=padding, initializer=initializer, activat_fn = activat_fn)
            
            net = tf.concat([net1x1, net3x3, net5x5, net_s1x1], axis=3)
            
            if flatten == True:
                net = tf.reshape(net, [-1, int(np.prod(net.get_shape()[1:]))], name=name+"_flatout")
                
            
            return net


def shortcut(inputs, identity, name):  #Use 1X1 conv with proper stride to match dimesions
    
    in_shape =  inputs.get_shape().as_list()
    res_shape = identity.get_shape().as_list()
    
    dim_diff = [res_shape[1]/in_shape[1],
                res_shape[2]/in_shape[2]]
    
    
    if dim_diff[0] > 1  and dim_diff[1] > 1:
    
        identity = convolution_layer(identity, [1,1,in_shape[3]], [1,dim_diff[0],dim_diff[1],1], name="shotcut", padding="VALID")
    
    resout = tf.add(inputs, identity, name=name)
    
    return resout

def global_avg_pooling(inputs, flatten="False", name= 'global_avg_pooling'):
    
    in_shape =  inputs.get_shape().as_list()  
    netout = tf.nn.avg_pool(inputs, [1,in_shape[1], in_shape[2],1], [1,1,1,1],padding = 'VALID')
    
    if flatten == True:
        netout = tf.reshape(netout, [-1, int(np.prod(netout.get_shape()[1:]))], name=name+"_flatout")
        
    return netout
    
   

### EDSR Specialized function
def resBlock(x,channels=64,kernel_size=[3,3],scale=1, reuse = False, is_bn = False, idx = 0, initializer=tf.contrib.layers.xavier_initializer(),activation_fn=tf.nn.relu):

    tmp = slim.conv2d(x,channels,kernel_size,activation_fn=activation_fn, weights_initializer=initializer)
    if is_bn:
        tmp = batchnorm(tmp, index = idx, reuse = reuse)
    else:
        tmp = tmp 
    tmp = slim.conv2d(tmp,channels,kernel_size,activation_fn=None, weights_initializer=initializer)
    tmp *= scale
    return x + tmp

def edsr_resblock(inputs, kernel_shape, stride = [1,1,1,1], repeations = 1, scale = 1, name="resblock"):

    assert len(kernel_shape) == repeations, "Provide kernel shape shall be equal to repeations!"

    for i in range(repeations):

        with tf.name_scope(name + str(i)): 
            pre_shape = inputs.get_shape()[-1]   
            k_shape = kernel_shape[i]
            rkernel_shape = [k_shape[0], k_shape[1], pre_shape, k_shape[2]]     
            net = convolution_layer(inputs, rkernel_shape, stride, name= name + str(i) + "_1")
            net = convolution_layer(net, rkernel_shape, stride, name= name + str(i)+ "_2".format(2), activat_fn=None)
            net *= scale
            inputs += net


    outputs = inputs

    return outputs

def upsample(x,scale=2,features=64,channels = 3,activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer()):

    assert scale in [2,3,4], "Only support scale 2,3,4"
    ch = channels 

    if ch == 3: isColor = True
    else: isColor = False

    x = slim.conv2d(x,features,[3,3],activation_fn=activation, weights_initializer=initializer)
    if scale == 2:

        ps_features = ch*(scale**2)
        x = slim.conv2d(x,ps_features,[3,3],activation_fn=activation, weights_initializer=initializer)
        #x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
        x = PS(x,2,color=isColor)
    elif scale == 3:
        ps_features =ch*(scale**2)
        x = slim.conv2d(x,ps_features,[3,3],activation_fn=activation)
        #x = slim.conv2d_transpose(x,ps_features,9,stride=1,activation_fn=activation)
        x = PS(x,3,color=isColor)
    elif scale == 4:
        ps_features = ch*(2**2)
        for i in range(2):
            x = slim.conv2d(x,ps_features,[3,3],activation_fn=activation)
            #x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
            x = PS(x,2,color=isColor)
    return x

def upsample_attention(x, att_weight, scale=2,features=64,channels = 3,  attentions = 1,activation=tf.nn.relu):

    assert scale in [2,3,4], "Only support scale 2,3,4"
    ch = channels
    if ch == 3: isColor = True
    else: isColor = False

    bsize, a, b, c = x.get_shape().as_list()
    bsize = tf.shape(x)[0]

    x = slim.conv2d(x,features,[3,3],activation_fn=activation)
    if scale == 2:

        ps_features = ch*(scale**2)*attentions
        x = slim.conv2d(x,ps_features,[3,3],activation_fn=activation)

        # Attention layer
       
        x = tf.reshape(x, (bsize, a, b, ch*(scale**2), attentions))
        x = tf.multiply(x, att_weight)
        x = tf.reduce_sum(x,4) 
        #x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
        x = PS(x,2,color=isColor)
    elif scale == 3:
        ps_features =ch*(scale**2)*attentions
        x = slim.conv2d(x,ps_features,[3,3],activation_fn=activation)

        # Attention layer
        x = tf.reshape(x, (bsize, a, b, ch*(scale**2), attentions))
        x = tf.multiply(x, att_weight)
        x = tf.reduce_sum(x,4) 
        #x = slim.conv2d_transpose(x,ps_features,9,stride=1,activation_fn=activation)
        x = PS(x,3,color=isColor)
    elif scale == 4:
        ps_features = ch*(2**2)*attentions
        for i in range(2):
            x = slim.conv2d(x,ps_features,[3,3],activation_fn=activation)
             # Attention layer
            x = tf.reshape(x, (bsize, a, b, ch*(scale**2), attentions))
            x = tf.multiply(x, att_weight)
            x = tf.reduce_sum(x,4) 
            #x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
            x = PS(x,2,color=isColor)
    return x


def upsample_local_attention(x, att_weight, scale=2,features=64,channels = 3,  attentions = 1,activation=tf.nn.relu):

    assert scale in [2,3,4], "Only support scale 2,3,4"
    ch = channels
    if ch == 3: isColor = True
    else: isColor = False

    bsize, a, b, c = x.get_shape().as_list()
    bsize = tf.shape(x)[0]

    x = slim.conv2d(x,features,[1,1],activation_fn=activation)
    if scale == 2:

        ps_features = ch*(scale**2)*attentions
        x = slim.conv2d(x,ps_features,[1,1],activation_fn=activation)

        # Attention layer  
        x = tf.reshape(x, (bsize, 48,a//48, 48, b//48, ch*(scale**2), attentions))
        x = tf.multiply(x, att_weight)
        x = tf.reshape(x, (bsize,a,b,ch*(scale**2), attentions))
        x = tf.reduce_sum(x,4) 
        x = tf.reshape(x, (bsize,a,b,ch*(scale**2)))
        #x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
        x = PS(x,2,color=isColor)

    elif scale == 3:
        ps_features =ch*(scale**2)*attentions
        x = slim.conv2d(x,ps_features,[1,1],activation_fn=activation)

        # Attention layer
        x = tf.reshape(x, (bsize, a, b, ch*(scale**2), attentions))
        x = tf.multiply(x, att_weight)
        x = tf.reduce_sum(x,4) 
        #x = slim.conv2d_transpose(x,ps_features,9,stride=1,activation_fn=activation)
        x = PS(x,3,color=isColor)

    elif scale == 4:
        ps_features = ch*(2**2)*attentions
        for i in range(2):
            x = slim.conv2d(x,ps_features,[3,3],activation_fn=activation)
             # Attention layer
            x = tf.reshape(x, (bsize, 12,a//12, 12, b//12, ch*(scale**2), attentions))
            x = tf.multiply(x, att_weight)
            x = tf.reshape(x, (bsize,a,b,ch*(scale**2), attentions))
            x = tf.reduce_sum(x,4) 
            x = tf.reshape(x, (bsize,a,b,ch*(scale**2)))
            #x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
            x = PS(x,2,color=isColor)
    return x


def upsample_local_attention_v2(x, att_weight, kernel_size = 3,scale=2, portion = 4,features=64,channels = 3,  attentions = 1,activation=tf.nn.relu):

    assert scale in [2,3,4], "Only support scale 2,3,4"
    ch = channels
    if ch == 3: isColor = True
    else: isColor = False


    bsize, a, b, c = x.get_shape().as_list()
    bsize = tf.shape(x)[0]

    x = slim.conv2d(x,features,[kernel_size,kernel_size],activation_fn=activation)
    if scale == 2:

        ps_features = ch*(scale**2)*attentions
        x = slim.conv2d(x,ps_features,[kernel_size,kernel_size],activation_fn=activation)

        # Attention layer  
        x = tf.reshape(x, (bsize, portion,a//portion, portion, b//portion, ch*(scale**2), attentions))
        x = tf.multiply(x, att_weight)
        x = tf.reshape(x, (bsize,a,b,ch*(scale**2), attentions))
        x = tf.reduce_sum(x,4) 
        x = tf.reshape(x, (bsize,a,b,ch*(scale**2)))
        #x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
        x = PS(x,2,color=isColor)

    elif scale == 3:
        ps_features =ch*(scale**2)*attentions
        x = slim.conv2d(x,ps_features,[1,1],activation_fn=activation)

        # Attention layer
        x = tf.reshape(x, (bsize, a, b, ch*(scale**2), attentions))
        x = tf.multiply(x, att_weight)
        x = tf.reduce_sum(x,4) 
        #x = slim.conv2d_transpose(x,ps_features,9,stride=1,activation_fn=activation)
        x = PS(x,3,color=isColor)

    return x

def upsample_ESPCN(x,scale=2,features=64,isColor=False,activation=tf.nn.relu):

    assert scale in [2,3,4], "Only support scale 2,3,4"

    if isColor : ch = 3
    else: ch = 1

    #x = slim.conv2d(x,features,[3,3],activation_fn=activation)
    if scale == 2:

        ps_features = ch*(scale**2)
        x = slim.conv2d(x,ps_features,[3,3],activation_fn=activation)
        #x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
        x = PS(x,2,color=isColor)
    elif scale == 3:
        ps_features =ch*(scale**2)
        x = slim.conv2d(x,ps_features,[3,3],activation_fn=activation)
        #x = slim.conv2d_transpose(x,ps_features,9,stride=1,activation_fn=activation)
        x = PS(x,3,color=isColor)
    elif scale == 4:
        ps_features = ch*(scale**2)
        
        x = slim.conv2d(x,ps_features,[3,3],activation_fn=activation)
        #x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
        x = PS(x,4,color=isColor)
    return x

def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X],2)  # bsize, b, a*r, r
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X],2)  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))

def PS(X, r, color=False):
    if color:
        Xc = tf.split(X, 3, 3)
        X = tf.concat([_phase_shift(x, r) for x in Xc],3)
    else:
        X = _phase_shift(X, r)
    return X





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    