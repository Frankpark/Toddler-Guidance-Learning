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
'''
def split(data):
    data_new = []
    for i in range(len(data)):
        if dat is None:
            data_new.append(None)
            continue
        dat = tf.reshape(dat, [NUM_AGENTS * TIME_STEP // RNN_STEP, RNN_STEP] + dat.get_shape().as_list()[2:])
        data_new.append(dat)
    res = []
    for i in range(RNN_STEP):
        res.append([])
        for dat in data:
'''

class UniversalRNNEncoder():
    def __init__(self, name, SIM):
        self.name = name
        self.enc = UniversalEncoder(name, SIM)
        with tf.variable_scope(self.name):
            #print("name", name)
            #print("scope", tf.get_variable_scope().name)
            #self.GRU = tf.nn.rnn_cell.GRUCell(STATE_LENGTH)
            self.GRU = GRU(name, STATE_LENGTH, STATE_LENGTH)
            #print("GRU", self.GRU.name)

    def get_params(self):
        return self.enc.get_params() + self.RNN.get_params()

    def forward(self, data):
        img, wav, obj, touch, h_0 = data
        NUM_AGENTS = tf.shape(h_0)[0]
        dat = (img, wav, obj, touch)
        with tf.variable_scope(self.name):
            i_t = self.enc.forward(dat)
            res = self.GRU.forward(i_t, h_0)
            #i_t = tf.reshape(i_t, [NUM_AGENTS, 1, STATE_LENGTH]) 
            #_, res = tf.nn.dynamic_rnn(self.GRU, i_t, initial_state = h_0)
        return res

    def forward_train(self, data):
        img, wav, obj, touch, h_0 = data
        NUM_AGENTS = tf.shape(h_0)[0]
        dat = (img, wav, obj, touch)
        with tf.variable_scope(self.name):
            z = self.enc.forward(dat) # [None, STATE_LENGTH]
            z = tf.reshape(z, [NUM_AGENTS * TIME_STEP // RNN_STEP, RNN_STEP, STATE_LENGTH])
            h_0 = tf.reshape(h_0, [NUM_AGENTS * TIME_STEP // RNN_STEP, RNN_STEP, STATE_LENGTH])
            h = [h_0[:, 0]]
            for i in range(RNN_STEP):
                h_iR = h_0[:, i]
                h_i = h[i]
                isSame = tf.tile(tf.reduce_mean(tf.square(h_iR - h_i), axis = 1, keep_dims = True) < 1e-6, [1, STATE_LENGTH])
                h_i = tf.where(isSame, h[i], h_i)
                h.append(self.GRU.forward(z[:, i], h_i))
            res = h[RNN_STEP]
            #_, res = tf.nn.dynamic_rnn(self.GRU, z, initial_state = h_0)
        return res

class LoadableVisionEncoder():
    def __init__(self, name):
        self.name = name
        IMG_C = 6
        with tf.variable_scope(self.name):
            self.weights = {
                'wc1iA': tf.get_variable('wc1iA', [4, 4, IMG_C, 32]),
                'wc2iA': tf.get_variable('wc2iA', [4, 4, 32, 64]),
                'wc3iA': tf.get_variable('wc3iA', [4, 4, 64, 128]),
                #'wc4iA': tf.get_variable('wc4iA', [3, 3, 64, 64]),
                'wd1iA': tf.get_variable('wd1iA', [128, 256]),
            }
            self.biases = {
                'bc1iA': tf.get_variable('bc1iA', [32]),
                'bc2iA': tf.get_variable('bc2iA', [64]),
                'bc3iA': tf.get_variable('bc3iA', [128]),
                #'bc4iA': tf.get_variable('bc4iA', [64], initializer = zero_init),
                'bd1iA': tf.get_variable('bd1iA', [256]),
            }
            self.BN_params = {
                'wc1BN': tf.get_variable('wc1BN', [32]),
                'wc2BN': tf.get_variable('wc2BN', [64]),
                'wc3BN': tf.get_variable('wc3BN', [128]),
                'bc1BN': tf.get_variable('bc1BN', [32]),
                'bc2BN': tf.get_variable('bc2BN', [64]),
                'bc3BN': tf.get_variable('bc3BN', [128]),
            }
            self.loadable_params = {
                'wc1iA': tf.placeholder(tf.float32, name='wc1iA', shape=[4, 4, IMG_C, 32]),
                'wc2iA': tf.placeholder(tf.float32, name='wc2iA', shape=[4, 4, 32, 64]),
                'wc3iA': tf.placeholder(tf.float32, name='wc3iA', shape=[4, 4, 64, 128]),
                'bc1iA': tf.placeholder(tf.float32, name='bc1iA', shape=[32]),
                'bc2iA': tf.placeholder(tf.float32, name='bc2iA', shape=[64]),
                'bc3iA': tf.placeholder(tf.float32, name='bc3iA', shape=[128]),
                'wc1BN': tf.placeholder(tf.float32, name='wc1BN', shape=[32]),
                'wc2BN': tf.placeholder(tf.float32, name='wc2BN', shape=[64]),
                'wc3BN': tf.placeholder(tf.float32, name='wc3BN', shape=[128]),
                'bc1BN': tf.placeholder(tf.float32, name='bc1BN', shape=[32]),
                'bc2BN': tf.placeholder(tf.float32, name='bc2BN', shape=[64]),
                'bc3BN': tf.placeholder(tf.float32, name='bc3BN', shape=[128]),
            }
            self.load_vision_params = [
                tf.assign(self.weights['wc1iA'], self.loadable_params['wc1iA']),
                tf.assign(self.biases['bc1iA'], self.loadable_params['bc1iA']),
                tf.assign(self.BN_params['wc1BN'], self.loadable_params['wc1BN']),
                tf.assign(self.BN_params['bc1BN'], self.loadable_params['bc1BN']),
                tf.assign(self.weights['wc2iA'], self.loadable_params['wc2iA']),
                tf.assign(self.biases['bc2iA'], self.loadable_params['bc2iA']),
                tf.assign(self.BN_params['wc2BN'], self.loadable_params['wc2BN']),
                tf.assign(self.BN_params['bc2BN'], self.loadable_params['bc2BN']),
                tf.assign(self.weights['wc3iA'], self.loadable_params['wc3iA']),
                tf.assign(self.biases['bc3iA'], self.loadable_params['bc3iA']),
                tf.assign(self.BN_params['wc3BN'], self.loadable_params['wc3BN']),
                tf.assign(self.BN_params['bc3BN'], self.loadable_params['bc3BN']),
            ]

    def load_params(self, sess, data):
        dict_all = {}
        for key in data.keys():
            dict_all[self.loadable_params[key]] = data[key]
        sess.run(self.load_vision_params, feed_dict = dict_all)

    def get_params(self):
        return list(self.weights.values()) + list(self.biases.values()) + list(self.BN_params.values())

    def forward(self, img):
        with tf.variable_scope(self.name):
            batch_size = tf.shape(img)[0]
            im1 = conv2D(img, self.weights['wc1iA'], self.biases['bc1iA'], strides = 2, padding = "SAME")
            im1 = im1 * self.BN_params['wc1BN'] + self.BN_params['bc1BN']
            im2 = conv2D(im1, self.weights['wc2iA'], self.biases['bc2iA'], strides = 2, padding = "SAME")
            im2 = im2 * self.BN_params['wc2BN'] + self.BN_params['bc2BN']
            im3 = conv2D(im2, self.weights['wc3iA'], self.biases['bc3iA'], strides = 2, padding = "VALID")
            im3 = im3 * self.BN_params['wc3BN'] + self.BN_params['bc3BN']
            im3 = tf.reduce_mean(im3, axis=[1,2])
            im4 = dense(im3, self.weights['wd1iA'], self.biases['bd1iA'], activation = 'relu')
        return im4

class LoadableAudioEncoder():
    def __init__(self, name, sr=16000, nfft=512, nhop=128):
        self.sr = sr
        self.nfft = nfft
        self.nhop = nhop
        self.name = name
        with tf.variable_scope(self.name):
            self.weights = {
                'wc1iA': tf.Variable(tf.random_normal([5, 5, 2, 16]), name = "wc1iA"),
                'wc2iA': tf.Variable(tf.random_normal([3, 3, 16, 32]), name = "wc2iA"),
                'wc3iA': tf.Variable(tf.random_normal([3, 3, 32, 64]), name = "wc3iA"),
                'wd1iA': tf.Variable(tf.random_normal([64, 128]), name = "wd1iA"),
            }
            self.biases = {
                'bc1iA': tf.Variable(tf.zeros([16]), name = "bc1iA"),
                'bc2iA': tf.Variable(tf.zeros([32]), name = "bc2iA"),
                'bc3iA': tf.Variable(tf.zeros([64]), name = "bc3iA"),
                'bd1iA': tf.Variable(tf.zeros([128]), name = "bd1iA"),
            }

    def get_params(self):
        return list(self.weights.values()) + list(self.biases.values())

    def mel(self, audio):
        stfts = tf.signal.stft(audio, frame_length=self.nfft, frame_step=self.nhop, fft_length=self.nfft)
        spectrograms = tf.abs(stfts)

        num_spectrogram_bins = stfts.shape[-1]

        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 128

        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, self.sr, lower_edge_hertz, upper_edge_hertz)
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

        # Compute MFCCs from log_mel_spectrograms and take the first 13.
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :13]
        
        return log_mel_spectrograms, mfccs

    def forward(self, data):
        print(data.shape)
        with tf.variable_scope(self.name): 
            batch_size = tf.shape(data)[0]
            data = tf.reshape(data, [batch_size, 2, -1])
            mel_spec1, mfcc1 = self.mel(data[:, 0])
            mel_spec2, mfcc2 = self.mel(data[:, 1])
            print(mel_spec1.shape, mfcc1.shape)
            mel_spec = tf.stack([mel_spec1, mel_spec2], axis=3)
            print(mel_spec.shape)
            im1 = conv2D(mel_spec, self.weights['wc1iA'], self.biases['bc1iA'], strides = 4, padding = "SAME")
            im2 = conv2D(im1, self.weights['wc2iA'], self.biases['bc2iA'], strides = 2, padding = "SAME")
            im3 = conv2D(im2, self.weights['wc3iA'], self.biases['bc3iA'], strides = 1, padding = "SAME")
            x = tf.reduce_mean(im3, axis=1)
            x1 = tf.reduce_max(x, axis=1)
            x2 = tf.reduce_mean(x, axis=1)
            x = x1 + x2
            res = dense(x, self.weights['wd1iA'], self.biases['bd1iA'], activation = 'x')
        return res

class LoadableEncoder():
    def __init__(self, name, SIM):
        self.name = name
        
        if SIM == 'VECA':
            NUM_OBJS = 3
            TACTILE_LENGTH = 1                  # COGNIANav simple tactile
        
        with tf.variable_scope(self.name):
            self.Evision = LoadableVisionEncoder('vision')
            self.Eaudio = LoadableAudioEncoder('audio')
            self.weights = {           
                'wd1wO': tf.get_variable('wd1wO', [NUM_OBJS, 256]),
                'wd1wT': tf.get_variable('wd1wT', [TACTILE_LENGTH, 128]),
                'wd1dA': tf.get_variable('wd1dA', [256 + 128 + 128, STATE_LENGTH])
            }
            self.biases = {
                'bd1wT': tf.get_variable('bd1wT', [128]),
                'bd1dA': tf.get_variable('bd1dA', [STATE_LENGTH], initializer = c_init(0.1))
            }

    def load_vision(self, sess, data):
        self.Evision.load_params(sess, data)

    def load_audio(self, sess, data):
        self.Eaudio.load_params(sess, data)

    def get_params(self):
        return self.Evision.get_params() + self.Eaudio.get_params() + list(self.weights.values()) + list(self.biases.values())

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
                im4 = self.Evision.forward(img)
            
            if wav is None:
                au1 = tf.zeros([batch_size, 256])
            else:
                wav = tf.reshape(wav, [batch_size, 2])
                au1 = tf.tile(wav, [1, 64])
                #au1 = self.Eaudio.forward(wav)

            if obj is None:
                ob1 = tf.ones([batch_size, 256])
            else:
                ob1 = tf.matmul(obj, self.weights['wd1wO'])

            if touch is None:
                to1 = tf.zeros([batch_size, 128])
            else:
                to1 = dense(touch, self.weights['wd1wT'], self.biases['bd1wT'])

            da0 = tf.concat([im4 * ob1, au1, to1], axis = 1)
            res = dense(da0, self.weights['wd1dA'], self.biases['bd1dA'], activation = 'tanh')
        return res


class UniversalEncoder():
    def __init__(self, name, SIM):
        self.name = name
        if SIM == 'VECA':
            IMG_C = 6
            WAV_C, WAV_LENGTH = 2, 66*13
            NUM_OBJS = 3
            #TACTILE_LENGTH = 1182 + 2 * 66     # GrabObject
            #TACTILE_LENGTH = 2 * 66            # GrabObject w/o tactile
            #TACTILE_LENGTH = 5 * 82 + 9888     # RunBaby
            #TACTILE_LENGTH = 5 * 82            # RunBaby w/o tactile
            TACTILE_LENGTH = 1                  # COGNIANav simple tactile
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
                
                'wd1wA': tf.get_variable('wd1wA', [WAV_C * WAV_LENGTH, 256]),

                'wd1wO': tf.get_variable('wd1wO', [NUM_OBJS, 256]),

                'wd1wT': tf.get_variable('wd1wT', [TACTILE_LENGTH, 256]),
                
                'wd1dA': tf.get_variable('wd1dA', [256 + 256 + 256, STATE_LENGTH])
            }
            self.biases = {
                'bc1iA': tf.get_variable('bc1iA', [32]),
                'bc2iA': tf.get_variable('bc2iA', [64]),
                'bc3iA': tf.get_variable('bc3iA', [64]),
                #'bc4iA': tf.get_variable('bc4iA', [64], initializer = zero_init),
                'bd1iA': tf.get_variable('bd1iA', [256]),

                'bd1wA': tf.get_variable('bd1wA', [256]),
                
                'bd1wT': tf.get_variable('bd1wT', [256]),

                'bd1dA': tf.get_variable('bd1dA', [STATE_LENGTH], initializer = c_init(0.1))
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
            
            if wav is None:
                au1 = tf.zeros([batch_size, 256])
            else:
                au1 = dense(wav, self.weights['wd1wA'], self.biases['bd1wA'], activation = 'relu')

            if obj is None:
                ob1 = tf.ones([batch_size, 256])
            else:
                ob1 = tf.matmul(obj, self.weights['wd1wO'])

            if touch is None:
                to1 = tf.zeros([batch_size, 256])
            else:
                to1 = dense(touch, self.weights['wd1wT'], self.biases['bd1wT'])

            da0 = tf.concat([im4 * ob1, au1, to1], axis = 1)
            res = dense(da0, self.weights['wd1dA'], self.biases['bd1dA'], activation = 'tanh')
        return res

class AtariEncoder():
    def __init__(self, name, env):
        IMG_C = env.observation_space['image'][0]
        self.name = name
        with tf.variable_scope(self.name):
            self.weights = {
                'wc1iA': tf.get_variable('wc1iA', [8, 8, IMG_C, 32]),
                'wc2iA': tf.get_variable('wc2iA', [4, 4, 32, 64]),
                'wc3iA': tf.get_variable('wc3iA', [3, 3, 64, 64]),
                #'wc4iA': tf.get_variable('wc4iA', [3, 3, 64, 64]),
                'wd1iA': tf.get_variable('wd1iA', [3136, STATE_LENGTH]),
            }
            self.biases = {
                'bc1iA': tf.get_variable('bc1iA', [32], initializer = zero_init),
                'bc2iA': tf.get_variable('bc2iA', [64], initializer = zero_init),
                'bc3iA': tf.get_variable('bc3iA', [64], initializer = zero_init),
                #'bc4iA': tf.get_variable('bc4iA', [64], initializer = zero_init),
                'bd1iA': tf.get_variable('bd1iA', [STATE_LENGTH], initializer = zero_init),
            }

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
        img = data
        with tf.variable_scope(self.name): 
            batch_size = tf.shape(img)[0]
            im1 = conv2D(img, self.weights['wc1iA'], self.biases['bc1iA'], strides = 4, padding = "VALID")
            im2 = conv2D(im1, self.weights['wc2iA'], self.biases['bc2iA'], strides = 2, padding = "VALID")
            im3 = conv2D(im2, self.weights['wc3iA'], self.biases['bc3iA'], strides = 1, padding = "VALID")
            im3 = tf.reshape(im3, [batch_size, 3136])
            res = dense(im3, self.weights['wd1iA'], self.biases['bd1iA'], activation = 'tanh')
        return res
'''
class MLPEncoder():
    def __init__(self, name, env):
        NUM_OBS = env.observation_space['touch']
        self.name = name
        with tf.variable_scope(self.name):
            self.weights = {
                'wd1iA': tf.get_variable('wd1iA', [NUM_OBS, STATE_LENGTH]),
            }
            self.biases = {
                'bd1iA': tf.get_variable('bd1iA', [STATE_LENGTH], initializer = zero_init),
            }

    def forward(self, data):
        touch = data
        with tf.variable_scope(self.name): 
            batch_size = tf.shape(touch)[0]
            res = dense(touch, self.weights['wd1iA'], self.biases['bd1iA'], activation = 'tanh')
        return res
'''
class AgentContinuousPPO():
    def __init__(self, name, encoder, action_space):
        self.name = name
        self.enc = encoder
        with tf.variable_scope(self.name):
            self.weights = {
                'wdmiA': tf.get_variable('wdmiA', [STATE_LENGTH, action_space]),
                'wdsiA': tf.get_variable('wdsiA', [STATE_LENGTH, action_space]),
           }
            self.biases = {
                'bdmiA': tf.get_variable('bdmiA', [action_space], initializer = zero_init),
                'bdsiA': tf.get_variable('bdsiA', [action_space], initializer = zero_init)
           }

    def forward(self, data):
        z = self.enc.forward(data)
        with tf.variable_scope(self.name):
            imm = dense(z, self.weights['wdmiA'], self.biases['bdmiA'], activation = 'x')
            ims = dense(z, self.weights['wdsiA'], self.biases['bdsiA'], activation = 'tanh')
            self.myu = imm
            self.sigma = tf.exp(ims)
        return (self.myu, self.sigma)
    
    def get_loss(self, data, oldmyu, oldsigma, action, adv, ent_coef, CLIPRANGE = 0.2, CLIPRANGE2 = 0.05):
        with tf.variable_scope(self.name):
            oldlogP = -2 * tf.square((action - oldmyu) / oldsigma)# - tf.log(oldsigma)
            myu, sigma = self.forward(data)
            logP = -2 * tf.square((action - myu) / sigma)# - tf.log(sigma)
            clipped_ratio = tf.exp(tf.clip_by_value(logP - oldlogP, np.log(1-CLIPRANGE), np.log(1+CLIPRANGE)))
            ratio = tf.exp(tf.clip_by_value(logP - oldlogP, -100, 10))
            print("ratio", ratio.shape, logP.shape, oldlogP.shape)
           
            # Defining Loss = - J is equivalent to max J
            pg_losses1 = -adv * ratio
            pg_losses2 = -adv * clipped_ratio
            # Final PG loss
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses1, pg_losses2))
            clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
 
            entropy = tf.reduce_mean(tf.reduce_mean(tf.log(sigma), 1))# + tf.reduce_mean(tf.abs(myu))
            #entropy = -tf.reduce_mean(tf.reduce_mean(logP - tf.log(1 - tf.square(tf.tanh(action)) + 1e-4), axis = 1))

            loss = pg_loss - ent_coef * entropy
            approxkl = 0.5 * tf.reduce_mean(tf.square(logP - oldlogP))
            obs = (myu, sigma, oldmyu, oldsigma, oldlogP, logP)
            return loss, clipfrac, entropy, approxkl, pg_loss, ratio, obs

class AgentDiscretePPO():
    def __init__(self, name, encoder, action_space):
        self.name = name
        self.enc = encoder
        with tf.variable_scope(self.name):
            self.weights = {
                'wdmiA': tf.get_variable('wdmiA', [STATE_LENGTH, action_space], initializer = n_init(0,0.06)) 
            }
            self.biases = {
                'bdmiA': tf.get_variable('bdmiA', [action_space], initializer = zero_init)
            }

    def forward(self, data):
        im4 = self.enc.forward(data)
        with tf.variable_scope(self.name):
            imm = dense(im4, self.weights['wdmiA'], self.biases['bdmiA'], activation = 'x')
            self.prob = tf.nn.softmax(imm, axis=1)
        return self.prob
    
    def get_loss(self, data, oldprob, action, adv, ent_coef, CLIPRANGE = 0.2, CLIPRANGE2 = 0.05):
        with tf.variable_scope(self.name):
            oldlogP = tf.log(tf.reduce_sum(oldprob * action, axis = 1) + 1e-4)
            prob = self.forward(data)
            logP = tf.log(tf.reduce_sum(prob * action, axis = 1) + 1e-4)
            clipped_ratio = tf.exp(tf.clip_by_value(logP - oldlogP, np.log(1-CLIPRANGE), np.log(1+CLIPRANGE)))
            ratio = tf.exp(tf.clip_by_value(logP - oldlogP, -100, 10))
            
            # Defining Loss = - J is equivalent to max J
            pg_losses1 = -adv * ratio
            pg_losses2 = -adv * clipped_ratio
            # Final PG loss
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses1, pg_losses2))
            clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
            
            entropy = tf.reduce_mean(tf.reduce_sum(-prob * tf.log(prob + 1e-4), 1))
            loss = pg_loss - ent_coef * entropy
            approxkl = 0.5 * tf.reduce_mean(tf.square(logP - oldlogP))
            obs = (prob, oldprob, oldlogP, logP)
            return loss, clipfrac, entropy, approxkl, pg_loss, ratio, obs

class AgentContinuousSAC():
    def __init__(self, name, encoder, critic, action_space):
        self.name = name
        self.enc = encoder
        self.critic = critic
        with tf.variable_scope(self.name):
            self.weights = {
                'wdmiA': tf.get_variable('wdmiA', [STATE_LENGTH, action_space]),
                'wdsiA': tf.get_variable('wdsiA', [STATE_LENGTH, action_space]),
            }
            self.biases = {
                'bdmiA': tf.get_variable('bdmiA', [action_space], initializer = zero_init),
                'bdsiA': tf.get_variable('bdsiA', [action_space], initializer = zero_init)
            }

    def get_params(self):
        return list(self.weights.values()) + list(self.biases.values()) + list(self.enc.get_params())

    def forward(self, data):
        z = self.enc.forward(data)
        with tf.variable_scope(self.name):
            imm = dense(z, self.weights['wdmiA'], self.biases['bdmiA'], activation = 'x')
            ims = dense(z, self.weights['wdsiA'], self.biases['bdsiA'], activation = 'tanh')
            self.myu = imm
            #ims = -20 + 22 * (ims + 1) * 0.5
            self.sigma = tf.exp(ims)
        return (self.myu, self.sigma)
    
    def get_loss(self, data, oldmyu, oldsigma, ent_coef):
        with tf.variable_scope(self.name):
            myu, sigma = self.forward(data)
            action = myu + sigma * tf.random.truncated_normal(tf.shape(myu))
            #oldlogP = log_prob_tf(oldmyu, oldsigma, action)
            oldlogP = -0.5 * tf.square((action - oldmyu) / oldsigma)# - tf.log(oldsigma)
            #logP = log_prob_tf(myu, sigma, action)
            myu, sigma = tf.stop_gradient(myu), tf.stop_gradient(sigma)
            logP = -0.5 * tf.square((action - myu) / sigma)# - tf.log(sigma)
            #clipped_ratio = tf.exp(tf.clip_by_value(logP - oldlogP, np.log(1-CLIPRANGE), np.log(1+CLIPRANGE)))
            ratio = tf.exp(tf.clip_by_value(logP - oldlogP, -100, 10))
            print("ratio", ratio.shape, logP.shape, oldlogP.shape) 
 
            pg_loss = -tf.reduce_mean(self.critic.forward(data, action))
            #entropy = tf.reduce_mean(tf.reduce_sum(tf.log(sigma), 1))
            #entropy = -tf.reduce_mean(tf.reduce_sum(logP - tf.log(1 - tf.square(tf.tanh(action)) + 1e-4), axis = 1))
            entropy = -tf.reduce_mean(tf.reduce_mean(logP - tf.log(1 - tf.square(tf.tanh(action)) + 1e-4), axis = 1))
            loss = pg_loss - ent_coef * entropy
            approxkl = 0.5 * tf.reduce_mean(tf.square(logP - oldlogP))
            obs = (myu, sigma, oldmyu, oldsigma, oldlogP, logP)
            return loss, entropy, approxkl, pg_loss, ratio, obs

class AgentDiscreteSAC():
    def __init__(self, name, encoder, critic, action_space):
        self.name = name
        self.enc = encoder
        self.critic = critic
        with tf.variable_scope(self.name):
            self.weights = {
                'wdmiA': tf.get_variable('wdmiA', [STATE_LENGTH, action_space]),
            }
            self.biases = {
                'bdmiA': tf.get_variable('bdmiA', [action_space], initializer = zero_init),
            }

    def get_params(self):
        return list(self.weights.values()) + list(self.biases.values()) + list(self.enc.get_params())

    def forward(self, data):
        z = self.enc.forward(data)
        with tf.variable_scope(self.name):
            imm = dense(z, self.weights['wdmiA'], self.biases['bdmiA'], activation = 'x')
            self.prob = tf.nn.softmax(imm)
        return self.prob
    
    def get_loss(self, data, oldprob, ent_coef):
        with tf.variable_scope(self.name):
            prob = self.forward(data)
            action = gumbel_softmax(tf.log(prob))
            #print(action.shape)
            #action = tf.random.
            oldlogP = tf.reduce_sum(tf.log(oldprob) * action, axis = 1)
            logP = tf.reduce_sum(tf.log(prob) * action, axis = 1)
            #clipped_ratio = tf.exp(tf.clip_by_value(logP - oldlogP, np.log(1-CLIPRANGE), np.log(1+CLIPRANGE)))
            ratio = tf.exp(tf.clip_by_value(logP - oldlogP, -100, 10))
            print("ratio", ratio.shape, logP.shape, oldlogP.shape) 
 
            pg_loss = -tf.reduce_mean(self.critic.forward(data, action))
            entropy = -tf.reduce_mean(logP)
            #entropy = tf.reduce_mean(tf.reduce_sum(tf.log(sigma), 1))
            #entropy = -tf.reduce_mean(logP)
            loss = pg_loss - ent_coef * entropy
            approxkl = 0.5 * tf.reduce_mean(tf.square(logP - oldlogP))
            obs = (prob, oldprob, oldlogP, logP)
            return loss, entropy, approxkl, pg_loss, ratio, obs

class AgentContinuous_MDN():
    def __init__(self, name, encoder):
        self.name = name
        self.enc = encoder
        with tf.variable_scope(self.name):
            self.weights = {
                'wdmiA': tf.get_variable('wdmiA', [256, ACTION_LENGTH * NUM_QR]),
                'wdsiA': tf.get_variable('wdsiA', [256, ACTION_LENGTH * NUM_QR]),
                'wdpiA': tf.get_variable('wdpiA', [256, ACTION_LENGTH * NUM_QR])
            }
            self.biases = {
                'bdmiA': tf.get_variable('bdmiA', [ACTION_LENGTH * NUM_QR], initializer = tf.random_uniform_initializer(-1, 1)),
                'bdsiA': tf.get_variable('bdsiA', [ACTION_LENGTH * NUM_QR], initializer = zero_init),
                'bdpiA': tf.get_variable('bdpiA', [ACTION_LENGTH * NUM_QR], initializer = zero_init)
            }

    def forward(self, data):
        im4 = self.enc.forward(data)
        with tf.variable_scope(self.name):
            batch_size = tf.shape(img)[0]
            imm = dense(im4, self.weights['wdmiA'], self.biases['bdmiA'], activation = 'x')
            ims = dense(im4, self.weights['wdsiA'], self.biases['bdsiA'], activation = 'tanh')
            imp = dense(im4, self.weights['wdpiA'], self.biases['bdpiA'], activation = 'x')
            imm = tf.reshape(imm, [batch_size, ACTION_LENGTH, NUM_QR])
            ims = tf.reshape(ims, [batch_size, ACTION_LENGTH, NUM_QR])
            imp = tf.nn.softmax(tf.reshape(imp, [batch_size, ACTION_LENGTH, NUM_QR]), axis = 2)
            self.myu = imm
            self.sigma = tf.exp(ims)
            self.P = imp
        return (self.myu, self.sigma, self.P)
    
    def get_loss(self, img0, obj, oldmyu, oldsigma, oldprob, action, adv, ent_coef, CLIPRANGE = 0.2, CLIPRANGE2 = 0.05):
        with tf.variable_scope(self.name):
            num_batch = tf.shape(action)[0]
            action = tf.tile(tf.reshape(action, [num_batch, ACTION_LENGTH, 1]), [1, 1, NUM_QR])
            oldlogP = -2 * tf.square((action - oldmyu) / oldsigma) - 0.5*np.log(2*np.pi) - tf.log(oldsigma)
            myu, sigma, prob = self.forward(img0, obj)
            logP = -2 * tf.square((action - myu) / sigma) - 0.5*np.log(2*np.pi) - tf.log(sigma)
            
            oldP = tf.reduce_sum(oldprob * tf.exp(oldlogP), axis = 2)
            P = tf.reduce_sum(prob * tf.exp(logP), axis = 2)
            ratio = P / oldP #ratio = tf.exp(logP - oldlogP)
           
            # Defining Loss = - J is equivalent to max J
            pg_losses1 = -adv * ratio
            pg_losses2 = -adv * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
            # Final PG loss
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses1, pg_losses2))
            clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE))) 

            entropy = tf.reduce_mean(ratio * tf.log(P))
            loss = pg_loss - ent_coef * entropy
            approxkl = tf.reduce_mean(-tf.log(ratio))
            return loss, clipfrac, entropy, approxkl, pg_loss, ratio

class CriticPPO():
    def __init__(self, name, encoder):
        self.name = name
        self.enc = encoder
        with tf.variable_scope(self.name):
            self.weights = {
                'wd2iC': tf.get_variable('wd2iC', [STATE_LENGTH, 1], initializer = n_init(0, 0.06)),
            }
            self.biases = {
                'bd2iC': tf.get_variable('bd2iC', [1], initializer = zero_init),
            }

    def forward(self, data):
        im4 = self.enc.forward(data)
        with tf.variable_scope(self.name):
            res = dense(im4, self.weights['wd2iC'], self.biases['bd2iC'], activation = 'x')
        return res
    
    def get_loss(self, data, Vtarget, oldV0, CLIPRANGE = 0.2):
        with tf.variable_scope(self.name):
            vpred = self.forward(data)
            vpredclipped = oldV0 + tf.clip_by_value(vpred - oldV0, - CLIPRANGE, CLIPRANGE)
            # Unclipped value
            vf_losses1 = tf.square(vpred - Vtarget)
            # Clipped value
            vf_losses2 = tf.square(vpredclipped - Vtarget)
            vf_loss = tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
            #vf_loss = tf.reduce_mean(vf_losses1)
            return vf_loss

class CriticContinuousSAC():
    def __init__(self, name, encoder1, encoder2, ACTION_LENGTH):
        self.name = name
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        with tf.variable_scope(self.name):
            self.Q1 = SubCriticContinuousSAC('Q1', self.encoder1, ACTION_LENGTH)
            self.Q2 = SubCriticContinuousSAC('Q2', self.encoder2, ACTION_LENGTH)

    def get_params(self):
        return self.Q1.get_params() + self.Q2.get_params() + self.encoder1.get_params() + self.encoder2.get_params()

    def forward(self, data, action):
        action = tf.tanh(action)
        Q1 = self.Q1.forward(data, action)
        Q2 = self.Q2.forward(data, action)
        Q = tf.minimum(Q1, Q2)
        return Q

    def get_loss(self, data, action, Qtarget):
        action = tf.tanh(action)
        Qf_loss1 = self.Q1.get_loss(data, action, Qtarget)
        Qf_loss2 = self.Q2.get_loss(data, action, Qtarget)
        return Qf_loss1 + Qf_loss2

class SubCriticContinuousSAC():
    def __init__(self, name, encoder, ACTION_LENGTH):
        self.name = name
        self.enc = encoder
        with tf.variable_scope(self.name):
            self.weights = {
                'wd1iC': tf.get_variable('wd1iC', [ACTION_LENGTH + STATE_LENGTH, STATE_LENGTH], initializer = n_init(0, 0.06)),
                'wd2iC': tf.get_variable('wd2iC', [ACTION_LENGTH + STATE_LENGTH, 1], initializer = n_init(0, 0.06)),
            }
            self.biases = {
                'bd1iC': tf.get_variable('bd1iC', [STATE_LENGTH], initializer = zero_init),
                'bd2iC': tf.get_variable('bd2iC', [1], initializer = zero_init),
            }

    def get_params(self):
        return list(self.weights.values()) + list(self.biases.values())# + self.enc.get_params()

    def forward(self, data, action):
        z0 = self.enc.forward(data)
        with tf.variable_scope(self.name):
            da0 = tf.concat([z0, action], axis = 1)
            z1 = dense(da0, self.weights['wd1iC'], self.biases['bd1iC'], activation = 'relu')
            da1 = tf.concat([z1, action], axis = 1)
            res = dense(da1, self.weights['wd2iC'], self.biases['bd2iC'], activation = 'x')
        return res
    
    def get_loss(self, data, action, Qtarget):
        with tf.variable_scope(self.name):
            Qpred = self.forward(data, action)
            Qf_loss = tf.reduce_mean(tf.square(Qpred - Qtarget))
            return Qf_loss

class CriticDiscreteSAC():
    def __init__(self, name, encoder1, encoder2, ACTION_LENGTH):
        self.name = name
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        with tf.variable_scope(self.name):
            self.Q1 = SubCriticDiscreteSAC('Q1', self.encoder1, ACTION_LENGTH)
            self.Q2 = SubCriticDiscreteSAC('Q2', self.encoder2, ACTION_LENGTH)

    def get_params(self):
        return self.Q1.get_params() + self.Q2.get_params() + self.encoder1.get_params() + self.encoder2.get_params()

    def forward(self, data, action):
        Q1 = self.Q1.forward(data, action)
        Q2 = self.Q2.forward(data, action)
        Q = tf.minimum(Q1, Q2)
        return Q

    def get_loss(self, data, action, Qtarget):
        Qf_loss1 = self.Q1.get_loss(data, action, Qtarget)
        Qf_loss2 = self.Q2.get_loss(data, action, Qtarget)
        return Qf_loss1 + Qf_loss2

class SubCriticDiscreteSAC():
    def __init__(self, name, encoder, ACTION_LENGTH):
        self.name = name
        self.enc = encoder
        with tf.variable_scope(self.name):
            self.weights = {
                'wd2iC': tf.get_variable('wd2iC', [STATE_LENGTH, ACTION_LENGTH], initializer = n_init(0, 0.06)),
            }
            self.biases = {
                'bd2iC': tf.get_variable('bd2iC', [ACTION_LENGTH], initializer = zero_init),
            }

    def get_params(self):
        return list(self.weights.values()) + list(self.biases.values())# + self.enc.get_params()

    def forward(self, data, action):
        im4 = self.enc.forward(data)
        with tf.variable_scope(self.name):
            res = dense(im4, self.weights['wd2iC'], self.biases['bd2iC'], activation = 'x')
            Q = tf.reduce_sum(action * res, axis = 1, keep_dims = True)
            #print(Q.shape)
        return Q
    
    def get_loss(self, data, action, Qtarget):
        with tf.variable_scope(self.name):
            Qpred = self.forward(data, action)
            Qf_loss = tf.reduce_mean(tf.square(Qpred - Qtarget))
            return Qf_loss

