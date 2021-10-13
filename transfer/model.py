import tensorflow as tf
import numpy as np
from constants import *

from agents2 import UniversalEncoder as Encoder
if SPARSE:
    from agents2 import TwoLayerLARSPolicy as Policy
else:
    from agents2 import LinearPolicy as Policy

from utils import *
from pathlib import Path

class Model():
    def __init__(self, tag):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.img = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, IMG_C])
        self.y = tf.placeholder(tf.float32, [None, ANS_LENGTH])
        self.enc = Encoder('targetEC1', 'VECA')
        self.policy = Policy('P')

        data = (self.img, None, None, None)
        self.s = self.enc.forward(data)

        self.s = tf.stop_gradient(self.s)
        self.y_hat = self.policy.forward(self.s)
         
        if TASK == "CLASSIFICATION":
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_hat))
            self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y, axis=1), tf.argmax(self.y_hat, axis=1)), tf.float32))
            self.obs = {"loss": self.loss, "acc": self.acc}
        if TASK == "DISTANCE":
            self.loss = tf.reduce_mean(tf.square(self.y - self.y_hat))
            mean, var = 2.332, 0.108
            dist, res = tf.exp(self.y*var+mean), tf.exp(self.y_hat*var+mean)
            self.R1loss = tf.reduce_mean(tf.abs(1 - (res / dist)))
            self.R2loss = tf.reduce_mean(tf.square(1 - (res / dist)))
            self.obs = {"loss": self.loss, "R1": self.R1loss, "R2": self.R2loss}
        if TASK == "RECOGNITION":
            self.loss = tf.reduce_mean(tf.square(self.y - self.y_hat))
            boxA, boxB = self.y, self.y_hat
            xA = tf.maximum(boxA[:, 0], boxB[:, 0])
            xB = tf.minimum(boxA[:, 1], boxB[:, 1])
            yA = tf.maximum(boxA[:, 2], boxB[:, 2])
            yB = tf.minimum(boxA[:, 3], boxB[:, 3]) 

            interArea = tf.maximum(xB - xA, 0) * tf.maximum(yB - yA, 0)
            boxAArea = tf.abs((boxA[:, 1] - boxA[:, 0]) * (boxA[:, 3] - boxA[:, 2]))
            boxBArea = tf.abs((boxB[:, 1] - boxB[:, 0]) * (boxB[:, 3] - boxB[:, 2]))

            self.iou = tf.reduce_mean(interArea / (boxAArea + boxBArea - interArea))
            self.obs = {"loss": self.loss, "iou": self.iou}

        for key in self.obs:
            tf.summary.scalar(key, self.obs[key])
        self.merge = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(Path('./log/')/tag, self.sess.graph)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            if VECA:
                self.opt = makeOptimizer(0.001, self.loss, decay = False, var_list = train_vars)
            else:
                self.opt = makeOptimizer(0.005, self.loss, decay = False, var_list = train_vars)
        if SPARSE:
            self.optL = self.policy.update_LARS_weight(self.s, self.y)
        self.global_step = 0

        var_list = []
        var_list += tf.global_variables(scope = 'targetE')
        var_list += tf.global_variables(scope = 'P')
        self.saver = tf.train.Saver(var_list)
        self.loader = tf.train.Saver(tf.trainable_variables(scope = 'targetE'))
        self.sess.run(tf.global_variables_initializer())

    def train(self, x, y, u = None):
        dict_all = {self.img:x, self.y:y}
        if SPARSE:
            self.sess.run(self.optL, feed_dict = dict_all)
        _, loss = self.sess.run([self.opt, self.loss], feed_dict = dict_all) 
        return loss

    def test(self, x, y, u = None):
        dict_all = {self.img:x, self.y:y}
        if SPARSE:
            self.sess.run(self.optL, feed_dict = dict_all)
        loss = self.sess.run(self.loss, feed_dict = dict_all)
        return loss

    def debug_merge(self, x, y, u = None):
        dict_all = {self.img: x, self.y: y}
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
 
