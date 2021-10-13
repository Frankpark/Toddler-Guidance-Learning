import tensorflow as tf
import cv2
import numpy as np
import random
import tensorflow.contrib.slim as slim 
import math

def dense(x, W, b, activation = 'relu', use_bn = False):
    x = tf.matmul(x, W)
    if b is not None:
        x = tf.add(x, b)
    if use_bn:
        try:
            is_training = tf.get_default_graph().get_tensor_by_name("isT:0")
        except:
            print('No tensor assigned for training; Automatically True')
            is_training = True
        x = tf.layers.batch_normalization(x, training = is_training)
    if activation == 'x': return x
    if activation == 'sigmoid': return tf.nn.sigmoid(x)
    if activation == 'tanh': return tf.nn.tanh(x) 
    if activation == 'relu': return tf.nn.relu(x)
    if activation == 'lrelu': return tf.nn.leaky_relu(x)

def conv2D(x, W, b, strides = 1, activation = 'relu', use_bn = False, padding = "VALID"):
    x = tf.nn.conv2d(x, W, strides=[1,strides,strides,1], padding = padding)
    if use_bn:
        try:
            is_training = tf.get_default_graph().get_tensor_by_name("isT:0")
        except:
            print('No tensor assigned for training; Automatically True')
            is_training = True
        x = tf.layers.batch_normalization(x, training = is_training)
    else: x = tf.nn.bias_add(x, b)
    if activation == 'x': return x
    if activation == 'relu': return tf.nn.relu(x)
    if activation == 'lrelu': return tf.nn.leaky_relu(x)

def convT2D(x, W, b, strides = 1, activation = 'relu', use_bn = False, padding = "VALID"):
    if padding == "VALID":
        output_shape = tf.stack([tf.shape(x)[0], (x.shape[1] - 1) * strides + W.shape[0], (x.shape[2] - 1) * strides + W.shape[1], W.shape[3]])
    if padding == "SAME":
        output_shape = tf.stack([tf.shape(x)[0], x.shape[1] * strides, x.shape[2] * strides, W.shape[3]])
    W = tf.transpose(W, [0, 1, 3, 2])
    x = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1,strides,strides,1], padding = padding)
    if use_bn:
        try:
            is_training = tf.get_default_graph().get_tensor_by_name("isT:0")
        except:
            print('No tensor assigned for training; Automatically True')
            is_training = True
        x = tf.layers.batch_normalization(x, training = is_training)
    else: x = tf.nn.bias_add(x, b)
    if activation == 'x': return x
    if activation == 'relu': return tf.nn.relu(x)
    if activation == 'lrelu': return tf.nn.leaky_relu(x)
    if activation == 'sigmoid': return tf.nn.sigmoid(x)

def maxpool2D(x, k = 2):
    return tf.nn.max_pool(x,ksize=[1,k,k,1], strides=[1,k,k,1], padding = "SAME")

def unzip_obs(obs, IMG_H, IMG_W, RAW_WAV_LENGTH):
    imgs, wavs = [], []
    for i in range(NUM_AGENTS):
        img, wav = obs['img'][i], obs['wav'][i]
        img = np.reshape(img, [2, IMG_H, IMG_W])
        wav = np.reshape(wav, [2, RAW_WAV_LENGTH])
        #wav = abs(np.fft.rfft(wav))[:int(MAX_FREQ/FREQ_STEP)]
        #print(np.min(wav), np.max(wav))
        wav0, wav1 = wav[0], wav[1]
        wav0 = abs(np.fft.rfft(wav0))[:250]
        wav0 = np.log10(wav0 + 1e-8)
        wav1 = abs(np.fft.rfft(wav1))[:250]
        wav1 = np.log10(wav1 + 1e-8)
        wav = np.array([wav0, wav1])
        #wav = np.reshape(wav0 - wav1, [2, 250])
        #print(np.min(wav), np.max(wav))
        #print(np.min(wav0), np.max(wav0), np.min(wav1), np.max(wav1))
        #print(wav0)
        #print(img.mean(), img.var())
        print(np.max(wav0), np.max(wav1))
        imgs.append(img), wavs.append(wav)
    obs['img'] = np.array(imgs)
    obs['wav'] = np.array(wavs)

def wav2freq(wav):
    wav0, wav1 = wav[0], wav[1]
    wav0 = abs(np.fft.rfft(wav0))[:250]
    wav0 = np.log10(wav0 + 1e-8)
    wav1 = abs(np.fft.rfft(wav1))[:250]
    wav1 = np.log10(wav1 + 1e-8)
    wav = np.array([wav0, wav1])
    print(np.max(wav0), np.max(wav1))
    return wav


def getActionFromPolicy(p):
    res = np.zeros_like(p).astype(np.int32)
    for i in range(p.shape[0]):
        #p0 = (p[i] + 0.05) / (1 + 0.05 * ACTION_LENGTH)
        p0 = p[i]
        action = np.random.choice(p.shape[1], 1, p = p0)
        res[i][action] = 1
    return res

class rewardTracker():
    def __init__(self, GAMMA):
        self.mean = 0
        self.N = 0
        self.var = 0
        self.SSE = 0
        self.X0 = None
        self.GAMMA = GAMMA

    def update(self, x0):
        if self.X0 is None:
            self.X0 = x0
        else:
            self.X0 = self.X0 * self.GAMMA + x0
        #print(self.X0)
        for x in self.X0:
            self.N += 1
            error = x - self.mean
            self.mean += (error / self.N)
            self.SSE += error * (x - self.mean)

    def get_std(self):
        return math.sqrt(self.SSE / self.N) + 1e-8

def makeOptimizer(lr, loss, decay = False, var_list = None):
    if decay:
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(lr, global_step, 1000, 0.96, staircase = False)
        opt = tf.train.AdamOptimizer(lr)
        if var_list == None:
            gradients, variables = zip(*opt.compute_gradients(loss))
        else:
            gradients, variables = zip(*opt.compute_gradients(loss, var_list = var_list))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        final_opt = opt.apply_gradients(zip(gradients, variables), global_step=global_step)
    else:
        opt = tf.train.AdamOptimizer(lr)
        if var_list == None:
            gradients, variables = zip(*opt.compute_gradients(loss))
        else:
            gradients, variables = zip(*opt.compute_gradients(loss, var_list = var_list))
        print(gradients, variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        final_opt = opt.apply_gradients(zip(gradients, variables))
    return final_opt

def getRewardFromPos(pos, action, wav):
    wav = wav.flatten()
    wavL, wavR = np.max(wav[:250]), np.max(wav[250:])
    wav = wavL - wavR
    cosdeg = pos[2] / math.sqrt(1e-4 + pos[0]*pos[0] + pos[2]*pos[2])
    sindeg = pos[0] / math.sqrt(1e-4 + pos[0]*pos[0] + pos[2]*pos[2])
    if pos[0] * wav > 0:
        print("HOLYSHIT", wavL, wavR, pos)
    else:
        print("GOOD", wavL, wavR, pos)
    res = 0
    if action[1] * sindeg >= 0:
        res += abs(action[1]) * 0.03# * abs(sindeg)
    else:
        res -= abs(action[1]) * 0.03
    #res += max(0, action[0]) * (0.01 * (1 + cosdeg) * (1 + cosdeg))
    res += action[0] * (0.1 * cosdeg - 0.07)
    print(action, sindeg, cosdeg, res)
    return res
    '''
    if cosdeg >= 0.9:
        if action[2] == 1:
            return 0.03
        else:
            return 0
    elif pos[2] < 0:
        if action[0] == 1:
            return 0.03
        else:
            return 0
    else:
        if action[1] == 1:
            return 0.03
        else:
            return 0
    '''

def getActionFromPos(pos, wav):
    wav = wav.flatten()
    wavL, wavR = np.max(wav[:250]), np.max(wav[250:])
    wav = wavL - wavR
    cosdeg = pos[2] / math.sqrt(1e-4 + pos[0]*pos[0] + pos[2]*pos[2])
    if cosdeg > 0.8:
        return np.array([0, 0, 1])
    if wav > 0:
        return np.array([1, 0, 0])
    else:
        return np.array([0, 1, 0])
    
def variable_summaries(var):
  name = "_"+var.name
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean'+name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev'+name, stddev)
    tf.summary.scalar('max'+name, tf.reduce_max(var))
    tf.summary.scalar('min'+name, tf.reduce_min(var))
    tf.summary.histogram('histogram'+name, var)

def get_debug_images(imgs):
    IMG_H, IMG_W = imgs.shape[1], imgs.shape[2]
    res = tf.reshape(tf.transpose(imgs, [0, 3, 1, 2]), [-1, IMG_H, IMG_W, 1])
    res = tf.cast(255 * tf.clip_by_value(res, 0, 1), tf.uint8)
    return res

def resize(imgs, h, w):
    res = []
    for i in range(imgs.shape[0]):
        res.append(cv2.resize(imgs[i], dsize = (h, w)))
    return np.array(res)

def display(imgs, newH = None, newW = None):
    n, h, w, c = imgs.shape[0], imgs.shape[1], imgs.shape[2], imgs.shape[3]
    if newH is None:
        newH = h
    if newW is None:
        newW = w
    imgs = np.average(imgs, axis = 3)
    imgs = resize(imgs, newH, newW)
    for i in range(newH):
        res = ""
        for k in range(n):
            for j in range(newW):
                if imgs[k][i][j] > 0.8:
                    res += '@'
                elif imgs[k][i][j] > 0.6:
                    res += '#'
                elif imgs[k][i][j] > 0.4:
                    res += "*"
                else:
                    res += "."
            res += "|"
        print(res)
    print()

def pinv(a, rcond=1e-15):
    s, u, v = tf.svd(a)
    # Ignore singular values close to zero to prevent numerical overflow
    limit = rcond * tf.reduce_max(s)
    non_zero = tf.greater(s, limit)
    print(a.shape, s.shape)
    reciprocal = tf.where(non_zero, tf.reciprocal(s), tf.zeros_like(s))
    lhs = tf.matmul(v, tf.matrix_diag(reciprocal))
    return tf.matmul(lhs, u, transpose_b=True)

def BCE(x, _x):
    return x * tf.log(_x+1e-4) + (1 - x) * tf.log(1-_x+1e-4)

def batch_ATA_lstsq(A0, A, y, lamb):
    AT = tf.transpose(A, [2, 1, 0])                                           # [OUTPUT_SIZE, |I|, BATCH_SIZE
    AD = tf.transpose(A, [2, 0, 1])                                           # [OUTPUT_SIZE, BATCH_SIZE, |I|]
    A0T = tf.transpose(A0, [2, 1, 0])                                         # [OUTPUT_SIZE, INPUT_SIZE, BATCH_SIZE]
    y = tf.expand_dims(tf.transpose(y), 2)                                    # [OUTPUT_SIZE, BATCH_SIZE, 1]
    x = tf.matrix_solve_ls(tf.matmul(AT, AD), y, lamb)                        # [OUTPUT_SIZE, |I|, 1]
    d_l = tf.transpose(tf.squeeze(x, axis = 2))                               # [|I|, OUTPUT_SIZE]
    v_l = tf.matmul(AD, x)                                                    # [OUTPUT_SIZE, BATCH_SIZE, 1]
    ATv_l = tf.transpose(tf.squeeze(tf.matmul(A0T, v_l), axis = 2))           # [INPUT_SIZE. OUTPUT_SIZE]
    obs = (tf.matmul(AT, AD), x, y, tf.matmul(tf.matmul(AT, AD), x))
    return d_l, ATv_l, obs

def boolean_maskT(A, mask, shape, axis = 0):
    mask = tf.transpose(mask)
    if axis == 0:
        A = tf.transpose(A)
        res = tf.reshape(tf.boolean_mask(A, mask), [shape[1], shape[0]])
        res = tf.transpose(res)
    else:
        A = tf.transpose(A, [0, 2, 1])
        res = tf.reshape(tf.boolean_mask(A, mask, axis = 1), [shape[0], shape[2], shape[1]])
        res = tf.transpose(res, [0, 2, 1])
    #print("BM:", A.shape, res.shape)
    return res

def LARS(A, X, Y, I, NUM_I):
    '''
    A : Data inputs             [BATCH_SIZE, INPUT_SIZE]  
    X : Weight before iteration [INPUT_SIZE, OUTPUT_SIZE]
    Y : Data outputs            [BATCH_SIZE, OUTPUT_SIZE]
    I : Lambda zero space       [INPUT_SIZE, OUTPUT_SIZE]
    NUM_I : |I|                  int
    One step of LARS algorithm, i.e. single iteration in finding X such that argmin(|AX - Y| + lamb * |X|)
    '''
    #print(A.shape, X.shape, Y.shape, I.shape)
    BATCH_SIZE, INPUT_SIZE, OUTPUT_SIZE = tf.shape(A)[0], X.shape[0], X.shape[1]
    A_D = tf.tile(tf.expand_dims(A, 2), [1, 1, OUTPUT_SIZE])                                                    #[BATCH_SIZE, INPUT_SIZE, OUTPUT_SIZE]
    batch_size = tf.shape(A)[0]
    I_C = ~I
    MX = 1e9
    maskI = tf.where(I, tf.ones_like(I, dtype = tf.bool), tf.zeros_like(I, dtype = tf.bool))     #[INPUT_SIZE, OUTPUT_SIZE]
    maskI_C = tf.where(I_C, tf.ones_like(I, dtype = tf.bool), tf.zeros_like(I, dtype = tf.bool)) #[INPUT_SIZE, OUTPUT_SIZE]
    c = tf.matmul(tf.transpose(A), Y - tf.matmul(A, X)) #residual correlation                                   #[INPUT_SIZE, OUTPUT_SIZE]
    lamb = tf.reduce_max(tf.abs(c), axis = 0, keep_dims = True)                                                 #[1, OUTPUT_SIZE]
    
    A_I = boolean_maskT(A_D, maskI, [batch_size, NUM_I, OUTPUT_SIZE], axis = 1)                   #[BATCH_SIZE, |I|, OUTPUT_SIZE`]
    c_I = boolean_maskT(c, maskI, [NUM_I, OUTPUT_SIZE])                                           #[|I|, OUTPUT_SIZE]
    x_I = boolean_maskT(X, maskI, [NUM_I, OUTPUT_SIZE])                                           #[|I|, OUTPUT_SIZE]
    d_l, ATv_l, obs0 = batch_ATA_lstsq(A_D, A_I, tf.sign(c_I), 1e-2)                                                  #[|I|, OUTPUT_SIZE], [INPUT_SIZE, OUTPUT_SIZE]

    S1 = (lamb - c) / (1 - ATv_l)                                                                               #[INPUT_SIZE, OUTPUT_SIZE]
    S2 = (lamb + c) / (1 + ATv_l)                                                                               #[INPUT_SIZE, OUTPUT_SIZE]
    #print(S1.shape, maskI_C.shape)
    S1 = tf.where(tf.logical_and(S1>1e-12, maskI_C), S1, MX * tf.ones_like(S1))                                  #[INPUT_SIZE, OUTPUT_SIZE] 
    S2 = tf.where(tf.logical_and(S2>1e-12, maskI_C), S2, MX * tf.ones_like(S2))                                  #[INPUT_SIZE, OUTPUT_SIZE]
    S = tf.minimum(S1, S2)
    gamma = tf.reduce_min(S, axis = 0, keep_dims = True)                                                        #[1, OUTPUT_SIZE]
    #dI = tf.where(tf.abs(S - gamma) < 1e-6, tf.ones_like(S), tf.zeros_like(S))
    dI = tf.transpose(tf.one_hot(tf.argmin(S, axis = 0), INPUT_SIZE, on_value = True, off_value = False, dtype = tf.bool))

    ind = tf.where(tf.transpose(maskI))
    #print("d_l:", d_l.shape)
    d_l0 = d_l
    d_l = tf.reshape(tf.transpose(d_l), [NUM_I * OUTPUT_SIZE])
    d_l = tf.transpose(tf.scatter_nd(ind, d_l, shape = [OUTPUT_SIZE, INPUT_SIZE]))
    X += gamma * d_l
    #print(A.shape, c.shape, c_I.shape, d_l.shape, x.shape, y.shape, I.shape)
    I = tf.logical_or(I, dI)
    X = tf.stop_gradient(X)
    lossC = tf.reduce_mean(tf.square(Y - tf.matmul(A, X)))
    lossS = tf.reduce_mean(tf.abs(X))
    #loss += w_L * (lossC + lamb * lossS)
    loss = lossC
    obs = (I, d_l0, d_l, X, lossC, lossS, gamma, lamb, ATv_l, A, A_I, c, c_I, S1, S2, S, obs0)
    return [X, I, loss, obs]

def kl_normal_loss(mean, logvar):
    return tf.reduce_sum(tf.reduce_mean(0.5 * (-1 - logvar + mean*mean + tf.exp(logvar)), axis = 0))
