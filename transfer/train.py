import tensorflow as tf
import numpy as np
import os
from constants import *
import time
import pickle
from itertools import combinations
from tqdm import tqdm
from pathlib import Path

def RGB2G_f(x):
    res = 0.299 * x[:,:,:,0] + 0.587 * x[:,:,:,1] + 0.114 * x[:,:,:,2]
    print(res.shape)
    return res

def preprocess(x, RGB2G = False):
    #x = np.transpose(x, [0, 2, 3, 1])
    if RGB2G:
        x = np.stack((RGB2G_f(x[:,:,:,:3]), RGB2G_f(x[:,:,:,3:])), axis = 3)
    return x

#------------------------------------------------------------------------------

if TASK == "CLASSIFICATION":
    data_dir = '/data/BabyMind/transfer/vision/cls/'
if TASK == "DISTANCE":
    data_dir = '/data/BabyMind/transfer/vision/distance/'
if TASK == "RECOGNITION":
    data_dir = '/data/BabyMind/transfer/vision/recognition/'
TRAIN_STEP = 500000

#------------------------------------------------------------------------------

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('tag',  type=str, help='tag for checkpoint')
parser.add_argument('gpunum',  type=int, help='gpunumber')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpunum)

print("Load data from ",data_dir)
x_train, y_train = np.load(data_dir+'x_train.npy'), np.load(data_dir+'y_train.npy')
print("Hello")
print(x_train.shape)
print(y_train.shape)
y_train = np.eye(10)[y_train]

x_test, y_test = np.load(data_dir+'x_test.npy'), np.load(data_dir+'y_test.npy')
print(x_test.shape)
print(y_test.shape)
y_test = np.eye(10)[y_test]
x_train = preprocess(x_train)
x_test = preprocess(x_test)

print("Hello")
from model import Model
model = Model(args.tag)
#tag = "1Mbase"
#modelpath = Path("/root/scripts/baseline/") / args.tag / "SAC-NavigationCogSci-2000008"
modelpath = Path("/home/jspark/projects/baselineMM/") / args.tag / "SAC-NavigationCogSci-2000008"
#modelpath = Path("/home/jspark/projects/baseline/") / args.tag / "SAC-NavigationCogSci-2000008"
print("Load model from ",modelpath)
model.load(str(modelpath))

st = time.time()
for step in tqdm(range(TRAIN_STEP)):
    ind = np.random.choice(x_train.shape[0], BATCH_SIZE) 
    train_loss = model.train(x_train[ind], y_train[ind])
    if (step+1) % 1000 == 0:
        dt = time.time() - st
        print("Time elapsed : {:.5f}".format(dt))
        st = time.time()
        ind = np.random.permutation(x_test.shape[0])
        loss = model.test(x_test[ind], y_test[ind])
        model.debug_merge(x_test[ind], y_test[ind])
        #model.save(str(step))
        print("Step", step + 1, ":", train_loss, loss)

if TASK == "CLASSIFICATION":
    model.save('classification')
if TASK == "DISTANCE":
    model.save('distance')
if TASK == "RECOGNITION":
    model.save('recognition')
