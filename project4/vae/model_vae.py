from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
import scipy.io as sio
from six.moves import xrange
from skimage import io

from ops import *
from utils import *
import random
import numpy as np

class VAE(object):
    def __init__(self, sess, image_size=28,
                 batch_size=100, sample_size=100, output_size=28,
                 z_dim=5, c_dim=1, dataset_name='default',
                 checkpoint_dir=None, sample_dir=None):
        """

        Args:
            sess: TensorFlow session
            image_size: The size of input image.
            batch_size: The size of batch. Should be specified before training.
            sample_size: (optional) The size of sampling. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [28]
            z_dim: (optional) Dimension of latent vectors. [5]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [1]
        """
        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.z_dim = z_dim
        self.c_dim = c_dim

        self.en_dim = 64
        self.de_dim = 64

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def encoder(self, image, reuse=False, train=True):
        with tf.variable_scope("encoder", reuse=reuse) as scope:
            #######################################################
            # TODO: Define encoder network structure here. op.py
            # includes some basic layer functions for you to use.
            # Please use batch normalization layer after conv layer.
            # And use 'train' argument to indicate the mode of bn.
            # The output of encoder network should have two parts:
            # A mean vector and a log(std) vector. Both of them have
            # the same dimension with latent vector z.
            #######################################################
            d_conv1_bn = lrelu(batch_norm(conv2d(image,self.en_dim,name = 'd_conv1_bn'),train = train,name = 'd_bn_1'))
            #d_conv2_bn = lrelu(batch_norm(conv2d(d_conv1_bn,self.en_dim*2,name = 'd_conv2_bn'),train = train,name = 'bn_2'))
            d_conv2_bn = lrelu(batch_norm(conv2d(d_conv1_bn,self.en_dim*2,name = 'd_conv2_bn'),train = train,name = 'bn_2'))
            d_conv3_bn = lrelu(batch_norm(conv2d(d_conv2_bn,self.en_dim*4,name = 'd_conv3_bn'),train = train,name = 'bn_3'))
            d_conv4_bn = lrelu(batch_norm(conv2d(d_conv3_bn,self.en_dim*8,name = 'd_conv4_bn'),train = train,name = 'bn_4'))

            d_fc1 = linear(tf.reshape(d_conv2_bn,[self.batch_size,-1]),1,'d_linear')

            
            mean_v = tf.layers.dense(d_fc1,units = self.z_dim)
            std_log = tf.layers.dense(d_fc1,units = self.z_dim)
            print('mean shape:',mean_v.shape)
            
            return mean_v,std_log
            #######################################################
            #                   end of your code
            #######################################################


    def decoder(self, z, reuse=False, train=True):
        with tf.variable_scope("decoder", reuse=reuse):
            #######################################################
            # TODO: Define decoder network structure here. The size
            # of output should match the size of images. To make the
            # output pixel values in [0,1], add a sigmoid layer before
            # the output. Also use batch normalization layer after
            # deconv layer, and use 'train' argument to indicate the
            # mode of bn layer. Note that when sampling images using
            # trained model, you need to set train='False'.
            #######################################################

            def conv_out_size_same(size, stride):
                return int(math.ceil(float(size) / float(stride)))
            s = self.image_size
            s2 = conv_out_size_same(s,2)
            s4 = conv_out_size_same(s2,2)
            s8 = conv_out_size_same(s4,2)
            s16 = conv_out_size_same(s8,2)
            z_t = linear(z,self.de_dim*8*s16*s16,'g_linear')
            print(s,s2,s4,s8,s16)

            h0 = tf.reshape(z_t,[-1,s16,s16,self.de_dim*8])
            print(h0.shape)
            h0 = lrelu(batch_norm(h0,name = 'g_bn_0'))
            output1_shape = [self.batch_size,s8,s8,self.de_dim*4]
            g_conv1= deconv2d(h0,output1_shape,name = 'g_dcon_1')
            g_conv1 = lrelu(batch_norm(g_conv1,train=train,name = 'g_bn_1'))
        

            
            output2_shape = [self.batch_size,s4,s4,sel  f.de_dim*2]
            g_conv2= deconv2d(g_conv1,output2_shape,name = 'g_dcon_2')
            g_conv2 = lrelu(batch_norm(g_conv2,train=train,name = 'g_bn_2'))
                            
            output3_shape = [self.batch_size,s2,s2,self.de_dim]
            g_conv3 = deconv2d(g_conv2,output3_shape,name = 'g_dcon_3')
            g_conv3 = lrelu(batch_norm(g_conv3,train=train,name = 'g_bn_3'))
            
            output4_shape = [self.batch_size,s,s,self.c_dim]
            g_conv4 = deconv2d(g_conv3,output4_shape,name = 'g_dcon_4')
            g_conv4 = tf.nn.sigmoid(lrelu(batch_norm(g_conv4,train=train,name = 'g_bn_4')))
            print(g_conv1.shape,g_conv2.shape,g_conv3.shape,g_conv4.shape)
           
            return tf.reshape(g_conv4,shape=[-1,self.image_size,self.image_size])
            #######################################################
            #                   end of your code
            #######################################################

    def build_model(self):
        #######################################################
        # TODO: In this build_model function, define inputs,
        # operations on inputs and loss of VAE. For input,
        # you need to define it as placeholders. Remember loss
        # term has two parts: reconstruction loss and KL divergence
        # loss. Save the loss as self.loss. Use the
        # reparameterization trick to sample z.
        #######################################################
        
        self.x_placeholder = tf.placeholder(tf.float32,shape = [self.batch_size,self.image_size,self.image_size,self.c_dim],name = 'x')        
        self.mu,self.log_std = self.encoder(self.x_placeholder)
        eps_batch = self.log_std.get_shape().as_list()[0] \
                            if self.log_std.get_shape().as_list()[0] is not None else self.batch_size
        eps = tf.random_normal([eps_batch, self.z_dim], 0.0, 1.0, dtype=tf.float32)
        

        self.z = tf.add(self.mu,tf.multiply(tf.exp(self.log_std), eps))
        print('z shape:',self.z.shape)
        self.x_hat = self.decoder(self.z)

        x_reshape = tf.reshape(self.x_placeholder,shape = [-1,self.image_size*self.image_size])
        x_hat_reshape = tf.reshape(self.x_hat,[-1,self.image_size*self.image_size])
        self.KL = -0.5 * tf.reduce_sum(1 + self.log_std - tf.pow(self.mu, 2) - tf.exp(self.log_std), reduction_indices=1)
        #self.reloss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.x_hat, labels=x), reduction_indices=1)
        #self.re_loss = -tf.reduce_sum(self.x_placeholder*tf.log(tf.clip_by_value(self.x_hat,1e-10,1.0)),1)
        self.re_loss = tf.reduce_sum(tf.squared_difference(x_hat_reshape,x_reshape), 1)
        #self.re_loss = binary_cross_entropy(x_hat_reshape,x_reshape,name = 'cross_entropy')
        self.loss = tf.reduce_mean(self.re_loss+self.KL)
        #######################################################
        #                   end of your code
        #######################################################
        self.saver = tf.train.Saver()

    def train(self, config):
        """Train VAE"""
        # load MNIST dataset
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        data = mnist.train.images
        data = data.astype(np.float32)
        data_len = data.shape[0]
        data = np.reshape(data, [-1, 28, 28, 1])

        optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss)
        try:
            self.sess.run(tf.global_variables_initializer())
        except:
            tf.initialize_all_variables().run()

        start_time = time.time()
        counter = 1

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        sample_dir = os.path.join(config.sample_dir, config.dataset)
        if not os.path.exists(config.sample_dir):
            os.mkdir(config.sample_dir)
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)

        for epoch in xrange(config.epoch):
            batch_idxs = min(data_len, config.train_size) // config.batch_size
            print(epoch)
            #sample_z = np.random.uniform(-1,1, size=(self.sample_size , self.z_dim))
            sample_z = [np.random.normal(0,1,self.z_dim) for _ in range(100)]
            sample_inputs = data[0:self.sample_size,:]
            


            for idx in xrange(0, batch_idxs):
                counter += 1
                batch_images = data[idx*config.batch_size:(idx+1)*config.batch_size, :]
                #######################################################
                # TODO: Train your model here, print the loss term at
                # each training step to monitor the training process.
                # Print reconstructed images and sample images every
                # config.print_step steps. Sample z from standard normal
                # distribution for sampling images. You may use function
                # save_images in utils.py to save images.
                #######################################################
                batch_z = np.random.uniform(-1,1,[config.batch_size,self.z_dim]).astype(np.float32)
                _,cost,out_v = self.sess.run([optim,self.loss,self.x_hat],
                                       feed_dict={
                                           self.x_placeholder:batch_images
                                           })
                print(cost)

                if np.mod(counter, 100) == 1:
                    samples,loss = self.sess.run(
                        [self.x_hat,self.loss],
                        feed_dict = {
                            self.z:sample_z,
                            self.x_placeholder:sample_inputs
                            }
                        )
                    samples = np.reshape(samples,[-1,28,28,1])
                    
                    save_images(samples, image_manifold_size(samples.shape[0]),'./{}/sample_{:02d}_{:04d}.png'.format(config.sample_dir,epoch, idx))
                    print("[Sample] loss: %.8f" % (loss)) 



                
                #######################################################
                #                   end of your code
                #######################################################
                if np.mod(counter, 500) == 2 or (epoch == config.epoch-1 and idx == batch_idxs-1):
                    self.save(config.checkpoint_dir, counter)
            imgs = np.reshape(out_v,[-1,self.image_size,self.image_size,self.c_dim])
            save_images(imgs, [self.image_size,self.image_size], self.sample_dir+'/reconstruct_pic'+str(epoch)+'.png')

            
    def save(self, checkpoint_dir, step):
        model_name = "mnist.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False
