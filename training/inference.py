import tensorflow as tf
import numpy as np
import os
#from model_DDPG.headquarter import HeadQuarter
from constants import *
from utils import *
if METHOD == 'PPO':
    from mtl_model_PPO import Model as MTLModel
if METHOD == 'SAC':
    from mtl_model_SAC import Model as MTLModel
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

envs = []
for i, TASK in enumerate(TASKS):
    if SIM == 'VECA':
        if TASK == 'Navigation':
            from tasks.navigation.navigationEnv import Environment
        elif TASK == 'KickTheBall':
            from tasks.kicktheball.kickTheBallEnv import Environment
        elif TASK == 'MANavigation':
            from tasks.MANavigation.navigationEnv import Environment
        elif TASK == 'Transport':
            from tasks.transport.transportEnv import Environment
        elif TASK == 'COGNIANav':
            from tasks.COGNIANav.COGNIADemoEnv import Environment
        env = Environment(num_envs = NUM_ENVS, port = 8874 + i)
    elif SIM == 'ATARI':
        from tasks.atari.atariEnv import Environment
        env = Environment(num_envs = NUM_ENVS, env_name = TASK)
    elif SIM == 'CartPole':
        from tasks.cartpole.cartpoleEnv import Environment
        env = Environment(num_envs = NUM_ENVS)
    elif SIM == 'Pendulum':
        from tasks.pendulum.pendulumEnv import Environment
        env = Environment(num_envs = NUM_ENVS)
    envs.append(env)

model = MTLModel(envs)
#model.load('./model/SAC-Navigation-3379208')
#model.load('./video_models/KickTheBall/SAC-KickTheBall-8')
#model.load('./model_baseline/SAC/Navigation/medium/SAC-Navigation-3225608')
#model.load('./model_baseline/PPO/GrabObject/full/my-model-7577600')
#model.load('./model_baseline/SAC/KickTheBall/medium/SAC-KickTheBall-3200008')
#model.load('./model_baseline/SAC/COGNIADemo/SAC-COGNIANav-1680001')
TRAIN_STEP = 2000000
REC_STEP = 100000

for step in range(TRAIN_STEP):
    model.step()
    if (step+1) % 1280 == 0:
        model.restart()

