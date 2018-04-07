from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

class DCGAN(object):
    def __init__(self, sess, input_size=28,
                 batch_size=100, sample_num=100, output_size=28,
                 z_dim=100, c_dim=1, dataset_name='default',
                 checkpoint_dir=None, sample_dir=None):
        """

        Args:
          sess: TensorFlow session
          input_size: The size of input image.
          batch_size: The size of batch. Should be specified before training.
          z_dim: (optional) Dimension of dim for Z. [100]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [1]
        """
        self.sess = sess

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_size = input_size
        self.output_size = output_size

        self.z_dim = z_dim
        self.c_dim = c_dim

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir

        self.g_dim = 64
        self.d_dim = 64

        self.build_model()

    def discriminator(self, image, reuse=False, train=True):
        with tf.variable_scope("discriminator", reuse=reuse):
            #######################################################
            # TODO: Define discrminator network structure here. op.py
            # includes some basic layer functions for you to use.
            # Please use batch normalization layer after conv layer.
            # And use 'train' argument to indicate the mode of bn.
            #######################################################

            d_conv1_bn = lrelu(batch_norm(conv2d(image,self.d_dim,name = 'd_conv1_bn'),train = train,name = 'd_bn_1'))
            #d_conv2_bn = lrelu(batch_norm(conv2d(d_conv1_bn,64*2,name = 'd_conv2_bn'),train = train,name = 'bn_2'))
            d_conv2_bn = lrelu(batch_norm(conv2d(d_conv1_bn,self.d_dim*2,name = 'd_conv2_bn'),train = train,name = 'd_bn_2'))
            d_conv3_bn = lrelu(batch_norm(conv2d(d_conv2_bn,self.d_dim*4,name = 'd_conv3_bn'),train = train,name = 'd_bn_3'))
            d_conv4_bn = lrelu(batch_norm(conv2d(d_conv3_bn,self.d_dim*8,name = 'd_conv4_bn'),train = train,name = 'd_bn_4'))


            d_fc1 = linear(tf.reshape(d_conv4_bn,[self.batch_size,-1]),1,'d_linear')

            return d_fc1,tf.nn.sigmoid(d_fc1)



            #######################################################
            #                   end of your code
            #######################################################

    def generator(self, z, reuse=False, train=True):
        with tf.variable_scope("generator", reuse=reuse):
            #######################################################
            # TODO: Define decoder network structure here. The size
            # of output should match the size of images. Image scale
            # in DCGAN is [-1, +1], so you need to add a tanh layer
            # before the output. Also use batch normalization layer
            # after deconv layer, and use 'train' argument to indicate
            # the mode of bn layer. Note that when sampling images
            # using trained model, you need to set train='False'.
            #######################################################
            def conv_out_size_same(size, stride):
                return int(math.ceil(float(size) / float(stride)))
            s = self.input_size
            s2 = conv_out_size_same(s,2)
            s4 = conv_out_size_same(s2,2)
            s8 = conv_out_size_same(s4,2)
            s16 = conv_out_size_same(s8,2)
            z_t,_,_ = linear(z,self.g_dim*8*s16*s16,'g_linear',with_w=True)
            print(s,s2,s4,s8,s16)

            h0 = tf.reshape(z_t,[-1,s16,s16,self.g_dim*8])
            print(h0.shape)
            h0 = lrelu(batch_norm(h0,name = 'g_bn_0',train = train))

            output1_shape = [self.batch_size,s8,s8,self.g_dim*4]
            g_conv1,_,_ = deconv2d(h0,output1_shape,name = 'g_dcon_1',with_w = True)
            g_conv1 = lrelu(batch_norm(g_conv1,train=train,name = 'g_bn_1'))
        

            
            output2_shape = [self.batch_size,s4,s4,self.g_dim*2]
            g_conv2,_,_ = deconv2d(g_conv1,output2_shape,name = 'g_dcon_2',with_w = True)
            g_conv2 = lrelu(batch_norm(g_conv2,train=train,name = 'g_bn_2'))
                            
            output3_shape = [self.batch_size,s2,s2,self.g_dim]
            g_conv3,_,_ = deconv2d(g_conv2,output3_shape,name = 'g_dcon_3',with_w = True)
            g_conv3 = lrelu(batch_norm(g_conv3,train=train,name = 'g_bn_3'))
            
            output4_shape = [self.batch_size,s,s,self.c_dim]
            g_conv4,_,_ = deconv2d(g_conv3,output4_shape,name = 'g_dcon_4',with_w = True)
            g_conv4 = tf.nn.tanh(batch_norm(g_conv4,train=train,name = 'g_bn_4'))
            print(g_conv1.shape,g_conv2.shape,g_conv3.shape,g_conv4.shape)
           
            return g_conv4


            #######################################################
            #                   end of your code
            #######################################################
    def sampler(self, z, train=False):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            #######################################################
            # TODO: Define decoder network structure here. The size
            # of output should match the size of images. Image scale
            # in DCGAN is [-1, +1], so you need to add a tanh layer
            # before the output. Also use batch normalization layer
            # after deconv layer, and use 'train' argument to indicate
            # the mode of bn layer. Note that when sampling images
            # using trained model, you need to set train='False'.
            #######################################################
            def conv_out_size_same(size, stride):
                return int(math.ceil(float(size) / float(stride)))
            s = self.input_size
            s2 = conv_out_size_same(s,2)
            s4 = conv_out_size_same(s2,2)
            s8 = conv_out_size_same(s4,2)
            s16 = conv_out_size_same(s8,2)
            z_t= linear(z,self.g_dim*8*s16*s16,'g_linear')
            print(s,s2,s4,s8,s16)

            h0 = tf.reshape(z_t,[-1,s16,s16,self.g_dim*8])
            print(h0.shape)
            h0 = lrelu(batch_norm(h0,name = 'g_bn_0',train = train))
            output1_shape = [self.batch_size,s8,s8,self.g_dim*4]
            g_conv1 = deconv2d(h0,output1_shape,name = 'g_dcon_1')
            g_conv1 = lrelu(batch_norm(g_conv1,train=train,name = 'g_bn_1'))
        

            
            output2_shape = [self.batch_size,s4,s4,self.g_dim*2]
            g_conv2 = deconv2d(g_conv1,output2_shape,name = 'g_dcon_2')
            g_conv2 = lrelu(batch_norm(g_conv2,train=train,name = 'g_bn_2'))
                            
            output3_shape = [self.batch_size,s2,s2,self.g_dim]
            g_conv3 = deconv2d(g_conv2,output3_shape,name = 'g_dcon_3')
            g_conv3 = lrelu(batch_norm(g_conv3,train=train,name = 'g_bn_3'))
            
            output4_shape = [self.batch_size,s,s,self.c_dim]
            g_conv4  = deconv2d(g_conv3,output4_shape,name = 'g_dcon_4')
            g_conv4 = tf.nn.tanh(batch_norm(g_conv4,train=train,name = 'g_bn_4'))
            
            print(g_conv1.shape,g_conv2.shape,g_conv3.shape,g_conv4.shape)
           
            return g_conv4


            #######################################################
            #                   end of your code
            #######################################################

    def build_model(self):
        #######################################################
        # TODO: In this build_model function, define inputs,
        # operations on inputs and loss of DCGAN. For input,
        # you need to define it as placeholders. Discriminator
        # loss has two parts: cross entropy for real images and
        # cross entropy for fake images generated by generator.
        # Set reuse=True for discriminator when calculating the
        # second cross entropy. Define two different loss terms
        # for discriminator and generator, and save them as
        # self.d_loss and self.g_loss respectively.
        #######################################################
        self.x_placeholder = tf.placeholder(tf.float32,shape = [self.batch_size,28,28,1],name = 'X')
        self.z_placeholder = tf.placeholder(tf.float32,shape = [None,self.z_dim],name = 'Z')

        self.Dx,self.Dx_s = self.discriminator(self.x_placeholder)
        self.Gz = self.generator(self.z_placeholder)
        self.Dg,self.Dg_s = self.discriminator(self.Gz,reuse = True)
        self.sampler = self.sampler(self.z_placeholder)
        
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.Dg,labels = tf.ones_like(self.Dg_s)))

        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.Dx,labels = tf.ones_like(self.Dx_s)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.Dg,labels = tf.zeros_like(self.Dg_s)))


        self.d_loss = d_loss_real+d_loss_fake
        
        #######################################################
        #                   end of your code
        #######################################################
        # define var lists for generator and discriminator
        t_vars = tf.trainable_variables()
        
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        # create two optimizers for generator and discriminator,
        # and only update the corresponding variables.
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)
        try:
            self.sess.run(tf.global_variables_initializer())
        except:
            tf.initialize_all_variables().run()

        # load MNIST data
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        data = mnist.train.images
        data = data.astype(np.float32)
        data_len = data.shape[0]
        data = np.reshape(data, [-1, 28, 28, 1])
        data = data * 2.0 - 1.0

        print(data.shape)

        counter = 1
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        sample_z = np.random.uniform(-1,1, size=(self.sample_num , self.z_dim))
        sample_inputs = data[0:self.sample_num,:]

        #self.g_sum = merge_summary([self.z_sum, self.d__sum,self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        #self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])

        for epoch in xrange(config.epoch):
            batch_idxs = min(data_len, config.train_size) // config.batch_size
            print(epoch)
            
            for idx in xrange(0, batch_idxs):
                batch_images = data[idx * config.batch_size:(idx + 1) * config.batch_size, :]
                #######################################################
                # TODO: Train your model here. Sample hidden z from
                # standard uniform distribution. In each step, run g_optim
                # twice to make sure that d_loss does not go to zero.
                # print the loss terms at each training step to monitor
                # the training process. Print sample images every
                # config.print_step steps.You may use function
                # save_images in utils.py to save images.
                #######################################################
                
                batch_z = np.random.uniform(-1,1,[config.batch_size,self.z_dim]).astype(np.float32)
                _,dLoss = self.sess.run([d_optim, self.d_loss],
                                         feed_dict={
                                             self.z_placeholder:batch_z,
                                             self.x_placeholder:batch_images}) #Update the discriminator
                _,gLoss = self.sess.run([g_optim,self.g_loss],
                                         feed_dict={
                                             self.z_placeholder:batch_z}) #Update the generator
                _,gLoss = self.sess.run([g_optim,self.g_loss],
                                         feed_dict={
                                             self.z_placeholder:batch_z}) #Update the generator
                print(dLoss,gLoss)
                
                if np.mod(counter, 100) == 1:
                    samples, d_Loss,g_Loss = self.sess.run(
                        [self.sampler,self.d_loss,self.g_loss],
                        feed_dict = {
                            self.z_placeholder:sample_z,
                            self.x_placeholder:sample_inputs
                            }
                        )
                    
                    save_images(samples, image_manifold_size(samples.shape[0]),'./{}/train_{:02d}_{:04d}.png'.format(config.sample_dir,epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_Loss, g_Loss)) 
                #######################################################
                #                   end of your code
                #######################################################

                counter += 1
                if np.mod(counter, 500) == 1:
                    self.save(config.checkpoint_dir, counter)


    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_size, self.output_size)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
