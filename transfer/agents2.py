import tensorflow as tf
from utils import *
import numpy as np
import sys
import os
import cv2 
import time
from constants import *

zero_init = tf.constant_initializer(0.)
one_init = tf.constant_initializer(1.)

def u_init(mn, mx):
    return tf.random_uniform_initializer(mn,mx)

def n_init(mean, std):
    return tf.random_normal_initializer(mean, std)

def c_init(x):
    return tf.constant_initializer(x)

def o_init(x):
    return tf.orthogonal_initializer(x)

def normalize(x, axis):
    mean, var = tf.nn.moments(x, axes = axis, keep_dims = True)
    return (x - mean) * tf.rsqrt(var + 1e-8)

class UniversalEncoder():
    def __init__(self, name, SIM):
        self.name = name
        if SIM == 'VECA':
            IMG_C = 6
            WAV_C, WAV_LENGTH = 2, 66*13
            NUM_OBJS = 10
            #TACTILE_LENGTH = 1182 + 2 * 66     # GrabObject
            #TACTILE_LENGTH = 2 * 66            # GrabObject w/o tactile
            #TACTILE_LENGTH = 5 * 82 + 9888     # RunBaby
            #TACTILE_LENGTH = 5 * 82            # RunBaby w/o tactile
            TACTILE_LENGTH = 1
        if SIM == 'ATARI':
            IMG_C = 4
            WAV_C, WAV_LENGTH = 0, 0
            NUM_OBJS = 0
            TACTILE_LENGTH = 0
        if SIM == 'CartPole':
            IMG_C = 0
            WAV_C, WAV_LENGTH = 0, 0
            NUM_OBJS = 0
            TACTILE_LENGTH = 4
        if SIM == 'Pendulum':
            IMG_C = 0
            WAV_C, WAV_LENGTH = 0, 0
            NUM_OBJS = 0
            TACTILE_LENGTH = 3
        #IMG_C = env.observation_space['image'][0]
        #WAV_C, WAV_LENGTH = env.observation_space['audio']
        #NUM_OBJS = env.observation_space['obj']
        with tf.variable_scope(self.name):
            self.weights = {
                'wc1iA': tf.get_variable('wc1iA', [8, 8, IMG_C, 32]),
                'wc2iA': tf.get_variable('wc2iA', [4, 4, 32, 64]),
                'wc3iA': tf.get_variable('wc3iA', [3, 3, 64, 64]),
                #'wc4iA': tf.get_variable('wc4iA', [3, 3, 64, 64]),
                'wd1iA': tf.get_variable('wd1iA', [3136, 256]),
                'wd1wO': tf.get_variable('wd1wO', [NUM_OBJS, 256]),
                
                'wd1wT': tf.get_variable('wd1wT', [TACTILE_LENGTH, 256]),
                
                'wd1dA': tf.get_variable('wd1dA', [256 + 256 + 256, STATE_LENGTH])

            }
            
            '''
            'wd1wA': tf.get_variable('wd1wA', [WAV_C * WAV_LENGTH, 256]),
            '''

            self.biases = {
                'bc1iA': tf.get_variable('bc1iA', [32]),
                'bc2iA': tf.get_variable('bc2iA', [64]),
                'bc3iA': tf.get_variable('bc3iA', [64]),
                #'bc4iA': tf.get_variable('bc4iA', [64], initializer = zero_init),
                'bd1iA': tf.get_variable('bd1iA', [256]),

                #'bd1wA': tf.get_variable('bd1wA', [256]),
                
                'bd1wT': tf.get_variable('bd1wT', [256]),

                'bd1dA': tf.get_variable('bd1dA', [STATE_LENGTH], initializer = c_init(0.1))

            }
            if VAE:
                self.weights.update({'wd1ds': tf.get_variable('wd1ds', [256 + 256 + 256, STATE_LENGTH])})
                self.biases.update({'bd1ds': tf.get_variable('bd1ds', [STATE_LENGTH])})

    def get_params(self):
        return list(self.weights.values()) + list(self.biases.values())

    def normalize_filters(self, x):
        h, w, c1, c2 = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = tf.transpose(tf.reshape(x, [h*c1, w*c2, 1, 1]), [2, 0, 1, 3])
        x = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))       
        return tf.cast(x*255, tf.uint8)

    def summarize_filters(self):
        tf.summary.image('wc1', self.normalize_filters(self.weights['wc1iA']))#, max_outputs = 64)
        tf.summary.image('wc2', self.normalize_filters(self.weights['wc2iA']))#, max_outputs = 64)
        tf.summary.image('wc3', self.normalize_filters(self.weights['wc3iA']))#, max_outputs = 64)

    def forward(self, data):
        img, wav, obj, touch = data
        if img is not None:
            batch_size = tf.shape(img)[0]
        elif wav is not None:
            batch_size = tf.shape(wav)[0]
        elif obj is not None:
            batch_size = tf.shape(obj)[0]
        elif touch is not None:
            batch_size = tf.shape(touch)[0]
        with tf.variable_scope(self.name): 
            if img is None:
                im4 = tf.zeros([batch_size, 256])
            else:
                batch_size = tf.shape(img)[0]
                im1 = conv2D(img, self.weights['wc1iA'], self.biases['bc1iA'], strides = 4, padding = "VALID")
                im2 = conv2D(im1, self.weights['wc2iA'], self.biases['bc2iA'], strides = 2, padding = "VALID")
                im3 = conv2D(im2, self.weights['wc3iA'], self.biases['bc3iA'], strides = 1, padding = "VALID")
                im3 = tf.reshape(im3, [batch_size, 3136])
                im4 = dense(im3, self.weights['wd1iA'], self.biases['bd1iA'], activation = 'relu')
            
            au1 = tf.zeros([batch_size, 256])
            '''
            if wav is None:
                au1 = tf.zeros([batch_size, 256])
            else:
                au1 = dense(wav, self.weights['wd1wA'], self.biases['bd1wA'], activation = 'relu')
            '''

            if obj is None:
                ob1 = tf.ones([batch_size, 256])
            else:
                ob1 = tf.matmul(obj, self.weights['wd1wO'])

            if touch is None:
                to1 = tf.zeros([batch_size, 256])
            else:
                to1 = dense(touch, self.weights['wd1wT'], self.biases['bd1wT'])

            da0 = tf.concat([im4 * ob1, au1, to1], axis = 1)
            if VAE:
                eps = tf.random.normal((batch_size, STATE_LENGTH))
                mu = dense(da0, self.weights['wd1dA'], self.biases['bd1dA'], activation = 'x')
                logvar = dense(da0, self.weights['wd1ds'], self.biases['bd1ds'], activation = 'x')
                res = mu + eps * tf.exp(0.5 * logvar)
                return mu, logvar, res
            else:
                res = dense(da0, self.weights['wd1dA'], self.biases['bd1dA'], activation = 'tanh')
                return res

class UniversalReconstructor():
    def __init__(self, name, SIM):
        self.name = name
        if SIM == 'VECA':
            IMG_C = 6
            WAV_C, WAV_LENGTH = 2, 250
            NUM_OBJS = 3
            TACTILE_LENGTH = 1182 + 2 * 66     # GrabObject
            #TACTILE_LENGTH = 2 * 66            # GrabObject w/o tactile
            #TACTILE_LENGTH = 5 * 82 + 9888     # RunBaby
            #TACTILE_LENGTH = 5 * 82            # RunBaby w/o tactile
        if SIM == 'ATARI':
            IMG_C = 4
            WAV_C, WAV_LENGTH = 0, 0
            NUM_OBJS = 0
            TACTILE_LENGTH = 0
        if SIM == 'CartPole':
            IMG_C = 0
            WAV_C, WAV_LENGTH = 0, 0
            NUM_OBJS = 0
            TACTILE_LENGTH = 4
        if SIM == 'Pendulum':
            IMG_C = 0
            WAV_C, WAV_LENGTH = 0, 0
            NUM_OBJS = 0
            TACTILE_LENGTH = 3
        with tf.variable_scope(self.name):
            self.weights = {
                'wc1iA': tf.get_variable('wc1iA', [8, 8, 32, IMG_C]),
                'wc2iA': tf.get_variable('wc2iA', [4, 4, 64, 32]),
                'wc3iA': tf.get_variable('wc3iA', [3, 3, 256, 64]),
                #'wc4iA': tf.get_variable('wc4iA', [3, 3, 64, 64]),
                'wd1iA': tf.get_variable('wd1iA', [STATE_LENGTH, 3136*4]),
            }
            self.biases = {
                'bc1iA': tf.get_variable('bc1iA', [IMG_C]),
                'bc2iA': tf.get_variable('bc2iA', [32]),
                'bc3iA': tf.get_variable('bc3iA', [64]),
                #'bc4iA': tf.get_variable('bc4iA', [64], initializer = zero_init),
                'bd1iA': tf.get_variable('bd1iA', [3136*4]),
            }

    def get_params(self):
        return list(self.weights.values()) + list(self.biases.values())

    def normalize_filters(self, x):
        h, w, c1, c2 = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = tf.transpose(tf.reshape(x, [h*c1, w*c2, 1, 1]), [2, 0, 1, 3])
        x = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))       
        return tf.cast(x*255, tf.uint8)

    def summarize_filters(self):
        tf.summary.image('wc1', self.normalize_filters(self.weights['wc1iA']))#, max_outputs = 64)
        tf.summary.image('wc2', self.normalize_filters(self.weights['wc2iA']))#, max_outputs = 64)
        tf.summary.image('wc3', self.normalize_filters(self.weights['wc3iA']))#, max_outputs = 64)

    def forward(self, z):
        batch_size = tf.shape(z)[0]
        with tf.variable_scope(self.name):
            im3 = dense(z, self.weights['wd1iA'], self.biases['bd1iA'], activation = 'relu')
            im3 = tf.reshape(im3, [batch_size, 7, 7, 256])
            im2 = convT2D(im3, self.weights['wc3iA'], self.biases['bc3iA'], strides = 1, padding = "VALID", activation = 'lrelu', use_bn = True)
            print(im2.shape)
            im1 = convT2D(im2, self.weights['wc2iA'], self.biases['bc2iA'], strides = 2, padding = "VALID", activation = 'lrelu', use_bn = True)
            print(im1.shape)
            img = convT2D(im1, self.weights['wc1iA'], self.biases['bc1iA'], strides = 4, padding = "VALID", activation = 'x', use_bn = True)
            print(img.shape)
            img = tf.nn.sigmoid(img)
        return img

class SimpleReconstructor():
    def __init__(self, name, SIM):
        self.name = name
        with tf.variable_scope(self.name):
            self.weights = {
                'wd1dR': tf.get_variable('wd1dR', [STATE_LENGTH, 5 * 5 * 256]),
                'wc1dR': tf.get_variable('wc1dR', [4, 4, 256, 128]),
                'wc2dR': tf.get_variable('wc2dR', [4, 4, 128, 64]),
                'wc3dR': tf.get_variable('wc3dR', [4, 4, 64, 32]),
                'wc4dR': tf.get_variable('wc4dR', [6, 6, 32, 6])
            }
            self.biases = {
                'bd1dR': tf.get_variable('bd1dR', [256]),
                'bc1dR': tf.get_variable('bc1dR', [128]),
                'bc2dR': tf.get_variable('bc2dR', [64]),
                'bc3dR': tf.get_variable('bc3dR', [32]),
                'bc4dR': tf.get_variable('bc4dR', [6])
            }

    def forward(self, z):
        batch_size = tf.shape(z)[0]
        with tf.variable_scope(self.name):
            im0 = dense(z, self.weights['wd1dR'], self.biases['bd1dR'], activation = 'relu')
            im0 = tf.reshape(im0, [batch_size, 5, 5, 256])
            im1 = convT2D(im0, self.weights['wc1dR'], self.biases['bc1dR'], activation = 'lrelu', padding = "SAME", use_bn = True)
            im2 = convT2D(im1, self.weights['wc2dR'], self.biases['bc2dR'], activation = 'lrelu', padding = "SAME", use_bn = True)
            im3 = convT2D(im2, self.weights['wc3dR'], self.biases['bc3dR'], activation = 'lrelu', padding = "SAME", use_bn = True)
            im4 = convT2D(im3, self.weights['wc4dR'], self.biases['bc4dR'], activatino = 'x')
            res = tf.nn.sigmoid(im4)
        return res

class TwoLayerIndepPolicy():
    def __init__(self, name):
        self.name = name
        with tf.variable_scope(self.name):
            self.W1 = tf.get_variable('w1', [STATE_LENGTH, 1, 32], initializer = n_init(0, 1/np.sqrt(32)))
            self.B1 = tf.get_variable('b1', [STATE_LENGTH, 32])
            self.W2 = tf.get_variable('w2', [STATE_LENGTH, 32, 1], initializer = n_init(0, 1/np.sqrt(32)))
            self.B2 = tf.get_variable('b2', [STATE_LENGTH, 1])
            self.W = tf.get_variable('W', [STATE_LENGTH, ANS_LENGTH], initializer=n_init(0, 1/np.sqrt(STATE_LENGTH)))
            self.B = tf.get_variable('B', [ANS_LENGTH])

    def forward(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, STATE_LENGTH, 1])
        with tf.variable_scope(self.name):
            z = tf.einsum('bij,ijk->bik', x, self.W1) + self.B1
            z = tf.nn.relu(z)
            z = tf.einsum('bij,ijk->bik', z, self.W2) + self.B2
            z = tf.reshape(z, [batch_size, STATE_LENGTH])
            z = dense(z, self.W, self.B, activation = 'x')
        return z

class TwoLayerLARSPolicy():
    def __init__(self, name):
        self.name = name
        with tf.variable_scope(self.name):
            self.W1 = tf.get_variable('w1', [STATE_LENGTH, 1, 32])
            self.B1 = tf.get_variable('b1', [STATE_LENGTH, 32])
            self.W2 = tf.get_variable('w2', [STATE_LENGTH, 32, 1])
            self.B2 = tf.get_variable('b2', [STATE_LENGTH, 1])
            self.W = tf.get_variable('W', [STATE_LENGTH, ANS_LENGTH])

    def HAM(self, A, Y):
        INPUT_SIZE, OUTPUT_SIZE = A.shape[1], Y.shape[1]
        X = []
        X.append(tf.zeros([INPUT_SIZE, OUTPUT_SIZE]))
        c = tf.matmul(tf.transpose(A), Y)
        lamb = tf.reduce_max(tf.abs(c), axis = 0, keep_dims = True)
        I = tf.transpose(tf.one_hot(tf.argmax(tf.abs(c), axis=0), INPUT_SIZE, on_value=True, off_value=False, dtype = tf.bool))
        loss, w_L = 0., 1.
        loss += tf.reduce_mean(tf.square(Y))
        for i in range(NUM_ITERS):
            Xi, I, _, _ = LARS(A, X[i], Y, I, i+1)
            X.append(Xi)
            lossC = tf.reduce_mean(tf.square(Y - tf.matmul(A, Xi)))
            w_L *= 0.5
            loss += w_L * lossC
        return X, loss

    def update_LARS_weight(self, x, y):
        z = self.MLP(x)
        W, _ = self.HAM(x, y)
        return tf.assign(self.W, W[NUM_ITERS])

    def MLP(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, STATE_LENGTH, 1])
        with tf.variable_scope(self.name):
            z = tf.einsum('bij,ijk->bik', x, self.W1) + self.B1
            z = tf.nn.relu(z)
            z = tf.einsum('bij,ijk->bik', z, self.W2) + self.B2
            z = tf.reshape(z, [batch_size, STATE_LENGTH])
        return z
    
    def forward(self, x, HAM = False):
        with tf.variable_scope(self.name):
            z = self.MLP(x)
            if HAM:
                W, _ = self.HAM(x, y)
                z = tf.matmul(z, W)
            else:
                z = tf.matmul(z, self.W)
        return z

class TwoLayerPolicy():
    def __init__(self, name):
        self.name = name
        with tf.variable_scope(self.name):
            self.W1 = tf.get_variable('w1', [STATE_LENGTH, 128])
            self.B1 = tf.get_variable('b1', [128])
            self.W2 = tf.get_variable('w2', [128, ANS_LENGTH])
            self.B2 = tf.get_variable('b2', [ANS_LENGTH])

    def forward(self, x):
        with tf.variable_scope(self.name):
            x = dense(x, self.W1, self.B1, activation = 'relu')
            res = dense(x, self.W2, self.B2, activation = 'x')
        return res

class LinearPolicy():
    def __init__(self, name):
        self.name = name
        with tf.variable_scope(self.name):
            self.W = tf.get_variable('w', [STATE_LENGTH, ANS_LENGTH])
            self.B = tf.get_variable('b', [ANS_LENGTH])

    def forward(self, x):
        with tf.variable_scope(self.name):
            res = dense(x, self.W, self.B, activation = 'x')
        return res

class LARSPolicy():
    def __init__(self, name):
        self.name = name
        with tf.variable_scope(self.name):
            self.W = tf.get_variable('w', [STATE_LENGTH, ANS_LENGTH])
        
    def forward(self, x):
        with tf.variable_scope(self.name):
            res = tf.matmul(x, self.W)
        return res

    def optimizer(self, A, Y):
        INPUT_SIZE, OUTPUT_SIZE = A.shape[1], np.prod(Y.get_shape().as_list()[1:])
        X = []
        X.append(tf.zeros([INPUT_SIZE, OUTPUT_SIZE]))
        c = tf.matmul(tf.transpose(A), Y)
        lamb = tf.reduce_max(tf.abs(c), axis = 0, keep_dims = True)
        I = tf.transpose(tf.one_hot(tf.argmax(tf.abs(c), axis = 0), INPUT_SIZE, on_value = True, off_value = False, dtype = tf.bool))
        #obs.append((c, lamb, I, Y))
        for i in range(NUM_ITERS):
            Xi, I, _, ob = LARS(A, X[i], Y, I, i+1)
            X.append(Xi)
            #obs.append(ob)
        opt = tf.assign(self.W, X[NUM_ITERS])
        return opt
   
 
