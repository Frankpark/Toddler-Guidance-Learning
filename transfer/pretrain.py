import tensorflow as tf
import numpy as np
from constants import *

from agents import UniversalEncoder as Encoder, UniversalReconstructor as Reconstructor
#from agents import UniversalEncoder as Encoder, SimpleReconstructor as Reconstructor

from utils import *

class Model():
    def __init__(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.img = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, IMG_C])
        self.C = tf.placeholder(tf.float32)
        self.enc = Encoder('targetEA', 'VECA')
        self.rec = Reconstructor('R', 'VECA')

        data = (self.img, None, None, None)
        mean, logvar, self.s = self.enc.forward(data)
        self.x_hat = self.rec.forward(self.s)
        self.recon_loss = tf.reduce_sum(tf.reduce_mean(tf.square(self.img - self.x_hat), axis = [1,2,3]))
        self.kl_loss = kl_normal_loss(mean, logvar)
        self.loss = self.recon_loss + 0.1 * self.kl_loss#10 * tf.abs(self.kl_loss - self.C)
        self.obs = {'Rloss': self.recon_loss, 'KLloss': self.kl_loss, 'C': self.C}
        for key in self.obs:
            tf.summary.scalar(key, self.obs[key])
        tf.summary.image('imgR', tf.cast(255*self.x_hat[:,:,:,:3], tf.uint8), max_outputs = 8)
        tf.summary.histogram('mean', mean)
        tf.summary.histogram('logvar', logvar)
        tf.summary.histogram('z', self.s)
        self.merge = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./log/', self.sess.graph)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            print(train_vars)
            if VECA:
                self.opt = makeOptimizer(0.0003, self.loss, decay = False, var_list = train_vars)
            else:
                self.opt = makeOptimizer(0.005, self.loss, decay = False, var_list = train_vars)
        self.global_step = 0

        var_list = []
        var_list += tf.global_variables(scope = 'targetE')
        var_list += tf.global_variables(scope = 'R')
        self.saver = tf.train.Saver(var_list)
        self.sess.run(tf.global_variables_initializer())

    def train(self, x, C):
        dict_all = {self.img:x, self.C:C}
        if SPARSE:
            self.sess.run(self.optL, feed_dict = dict_all)
        _, loss = self.sess.run([self.opt, self.loss], feed_dict = dict_all) 
        return loss

    def test(self, x, C):
        dict_all = {self.img:x, self.C:C}
        if SPARSE:
            self.sess.run(self.optL, feed_dict = dict_all)
        loss = self.sess.run(self.loss, feed_dict = dict_all)
        return loss

    def debug_merge(self, x, C):
        dict_all = {self.img: x, self.C: C}
        summary = self.sess.run(self.merge, feed_dict = dict_all)
        self.writer.add_summary(summary, self.global_step)
        self.global_step += 1
   
    def save(self, name = None):
        if name is None:
            self.saver.save(self.sess, './model/my-model', global_step = self.global_step)
        else:
            self.saver.save(self.sess, './model/' + name)

    def load(self, name):
        self.loader.restore(self.sess, name)

import os
from constants import *
import time
import pickle

def RGB2G_f(x):
    res = 0.299 * x[:,:,:,0] + 0.587 * x[:,:,:,1] + 0.114 * x[:,:,:,2]
    print(res.shape)
    return res

def preprocess(x, RGB2G = False):
    x = np.transpose(x, [0, 2, 3, 1])
    if RGB2G:
        x = np.stack((RGB2G_f(x[:,:,:,:3]), RGB2G_f(x[:,:,:,3:])), axis = 3)
    return x

TRAIN_STEP = 80000
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

data_dir = '/data/BabyMind/transfer/vision/'
x_train = np.load(data_dir+'x_train.npy')
x_test = np.load(data_dir+'x_test.npy')
x_train = preprocess(x_train)
x_test = preprocess(x_test)

model = Model()

st = time.time()
for step in range(TRAIN_STEP):
    ind = np.random.choice(x_train.shape[0], BATCH_SIZE) 
    model.train(x_train[ind], 20 * (step / TRAIN_STEP))
    if (step+1) % 50 == 0:
        dt = time.time() - st
        print("Time elapsed : {:.5f}".format(dt))
        st = time.time()
        ind = np.random.permutation(x_test.shape[0])
        loss = model.test(x_test[ind], 20 * (step / TRAIN_STEP))
        model.debug_merge(x_test[ind], 20 * (step / TRAIN_STEP))
        #model.save(str(step))
        print("Step", step + 1, ":", loss)

model.save('AE')
