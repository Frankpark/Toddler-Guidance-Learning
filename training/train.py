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
        elif TASK == 'MazeNav':
            from tasks.MazeNav.navigationEnv import Environment
        elif TASK == 'RunBaby':
            from tasks.runBaby.runBabyEnv import Environment
        elif TASK == 'COGNIANav':
            from tasks.COGNIANav.COGNIADemoEnv import Environment
        env = Environment(num_envs = NUM_ENVS, port = 8875 + i)
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
#model.load('model/PPO-RunBaby-1280000')

TRAIN_STEP = 2000000
SAVE_STEP = 80000 / NUM_ENVS
REC_STEP = 100000

if METHOD == 'PPO':
    SCHEDULE = True
    if SCHEDULE:
        lrAM, lrA0, lrA = 1e-3, 1e-4, 1e-4
    else:
        lrA = 2.5e-4
    for step in range(TRAIN_STEP):
        frac = 1.# - step / TRAIN_STEP
        #lrA = lr_schedule(1e-4, step, 500, 1.)
        model.step()
        if (step+1) % 1280 == 0:
            model.restart()
        if (step+1) % TIME_STEP == 0:
            print("Training Actor & Critic...")
            model.make_batch(ent_coef = 0.01 * frac, num_chunks = NUM_CHUNKS, update_gpu = True)
            loss, _, _, _, _, _, _ = model.make_batch(ent_coef = 0.01 * frac, add_merge = True, update_gpu = True, num_chunks = NUM_CHUNKS, lr = lrA)
            time_s = time.time()
            for substep in range(TRAIN_LOOP):
                approxkl = model.trainA(lrA / (TRAIN_LOOP), num_chunks = NUM_CHUNKS, ent_coef = 0.01 * frac)
                if approxkl > 0.01:
                    break
            dt = time.time() - time_s
            if approxkl > 0.01:
                 print("KL-divergence {:.3f}, early stopping at iter {:d}.".format(approxkl, substep))
            else:
                 print("KL-divergence {:.3f}, updated full gradient step, {:.3f}s elapsed.".format(approxkl, dt))
            
            lossP, _, _, _, _, _, dictA = model.make_batch(ent_coef = 0.01 * frac, add_merge = True, num_chunks = NUM_CHUNKS, lr = lrA)
            if SCHEDULE:
                if (approxkl > 0.01) or (loss < lossP):
                    lrA /= 2
                    lrA0 *= 0.95
                else:
                    lrA = min(lrA0, lrA * 2)
                    lrA0 *= 1.02
                if (approxkl > 0.015) or (loss < lossP):
                    model.revertNetwork()
                else:
                    model.updateNetwork()
                lrA0 = min(lrA0, lrAM) 
            else:
                model.updateNetwork()
            #lossP, _, _, _, _, _, dictA = model.make_batch(batch, ent_coef = 0.01 * frac, add_merge = True, num_chunks = NUM_CHUNKS, lr = lrA)
            model.debug_merge(dictA)
            model.clear()
        if (step + 1) % SAVE_STEP == 0:
            model.save()

if METHOD == 'SAC':
    lrA = 2.5e-4
    for step in range(TRAIN_STEP):
        frac = 1.# - step / TRAIN_STEP
        model.step()
        if (step+1) % 1280 == 0:
            model.restart()
        if step >= (BUFFER_LENGTH // 10) and (step + 1) % TIME_STEP == 0:
            print("Training Actor & Critic...")
            model.make_batch(ent_coef = 0.01 * frac, num_chunks = NUM_CHUNKS, update_gpu = True)
            loss, _, _, _, _, _, _ = model.make_batch(ent_coef = 0.01 * frac, add_merge = True, update_gpu = True, num_chunks = NUM_CHUNKS, lr = lrA)
            time_s = time.time()
            for substep in range(TRAIN_LOOP):
                approxkl = model.trainA(lrA / (TRAIN_LOOP), num_chunks = NUM_CHUNKS, ent_coef = 0.01 * frac)
                #if approxkl > 0.01:
                #    break
            dt = time.time() - time_s
            '''
            if approxkl > 0.01:
                 print("KL-divergence {:.3f}, early stopping at iter {:d}.".format(approxkl, substep))
            else:
            '''
            print("KL-divergence {:.3f}, updated full gradient step, {:.3f}s elapsed.".format(approxkl, dt))
            
            lossP, _, _, _, _, _, dictA = model.make_batch(ent_coef = 0.01 * frac, add_merge = True, update_gpu = True, num_chunks = NUM_CHUNKS, lr = lrA)
            model.updateNetwork()
            #lossP, _, _, _, _, _, dictA = model.make_batch(batch, ent_coef = 0.01 * frac, add_merge = True, num_chunks = NUM_CHUNKS, lr = lrA)
            model.debug_merge(dictA)
            #model.clear()
        if (step) % SAVE_STEP == 0:
            model.save()
