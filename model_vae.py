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

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
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
            reshape_input = tf.reshape(image, [-1, 28, 28, 1])

            h0 = lrelu(batch_norm(conv2d(reshape_input, 16, name='e_h0_conv_vae'), train = train, name = 'e_batch_0_vae'))

            h1 = lrelu(batch_norm(conv2d(h0, 32, name='e_h1_conv_vae'), train = train, name = 'e_batch_1_vae'))

            h2 = lrelu(batch_norm(conv2d(h1, 64, name='e_h2_conv_vae'), train = train, name = 'e_batch_2_vae'))

            # A flatten tensor with shape [h2.shape(0), k]
            fc = tf.contrib.layers.flatten(h2)

            mu = tf.layers.dense(fc, units = self.z_dim)
            std = 0.5 * mu
            epsilon = tf.random_normal(tf.stack([tf.shape(fc)[0], self.z_dim]))
            z = mu + tf.multiply(epsilon, tf.exp(std))
            #h3 = lrelu(batch_norm(conv2d(h2, 64*8, name = 'd_h3_conv'), train = train, name = 'd_batch_3'))
            #h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

            return z, mu, std
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
            h0 = tf.layers.dense(z, 64*7*7, name = 'd_ense2_vae')
            h0_reshape = tf.reshape(h0, [self.batch_size, 7, 7, 64])
            h0_deconv = deconv2d(h0_reshape, [self.batch_size, 7, 7, 32], d_w=1, d_h=1, name='d_conv1_vae')
            h0 = lrelu(batch_norm(h0_deconv, train = train, name = 'd_bn_0_vae'))

            h1_deconv = deconv2d(h0, [self.batch_size, 14, 14, 16], name = 'd_h1_vae')
            h1 = lrelu(batch_norm(h1_deconv, train = train, name = 'd_bn_1_vae'))

            h2_deconv = deconv2d(h1, [self.batch_size, 28, 28, 1], name = 'd_h2_vae')
            h2 = lrelu(batch_norm(h2_deconv, train = train, name = 'd_bn_2_vae'))

            h3_fc = tf.contrib.layers.flatten(h2)

            h4 = tf.layers.dense(h3_fc, units = 28 * 28, activation = tf.nn.sigmoid)
            
            img_out = tf.reshape(h4, shape=[-1, 28, 28])

            return img_out
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
        self.X_inputs = tf.placeholder(dtype = tf.float32, shape=[self.batch_size, 28, 28, 1], name = 'x_vae')
        Y = tf.reshape(self.X_inputs, shape = [-1, 28*28])
        self.z_sample, mu_sample, sd_sample = self.encoder(self.X_inputs)
        self.z_out = self.decoder(self.z_sample)

        Y_out = tf.reshape(self.z_out, [-1, 28*28])

        img_loss = tf.reduce_sum(tf.squared_difference(Y_out, Y), 1)
        latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd_sample - tf.square(mu_sample) - tf.exp(2.0 * sd_sample), 1)

        self.loss = tf.reduce_mean(img_loss + latent_loss)
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
            randoms = [np.random.normal(0,1,self.z_dim) for _ in range(100)]
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
                _, loss, dec = self.sess.run([optim, self.loss, self.z_out], feed_dict={self.X_inputs: batch_images})
                if np.mod(counter, 10) == 1:
                    samples = self.sess.run(
                        [self.z_out],
                        feed_dict={
                            self.z_sample:randoms,
                            }
                            )
                    imgs = np.reshape(samples,[-1,28,28,1])
                    save_images(imgs, [10,10], './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                    #print("[Sample] d_loss: %.8f, g_loss: %.8f" % (vae_loss))
                #######################################################
                #                   end of your code
                #######################################################
                if np.mod(counter, 500) == 2 or (epoch == config.epoch-1 and idx == batch_idxs-1):
                    self.save(config.checkpoint_dir, counter)

            imgs = np.reshape(dec,[-1,28,28,1])
            save_images(imgs, [10,10], self.sample_dir+'/reconstruct_pic'+str(epoch)+'.png')



            
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
