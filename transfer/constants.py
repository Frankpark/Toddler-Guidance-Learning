# headquarter
TIME_STEP = 16
VECA = True
VAE = False
SPARSE = False
#SPARSE = True
#TASK = "CLASSIFICATION"
#TASK = "DISTANCE"
TASK = "CLASSIFICATION"
if TASK == "CLASSIFICATION":
    RL = False
    ANS_LENGTH = 10
if TASK == "DISTANCE":
    RL = False
    ANS_LENGTH = 1
if TASK == "RECOGNITION":
    RL = False
    ANS_LENGTH = 4
if TASK == "KICKTHEBALL":
    RL = True
    ACTION_LENGTH = 2
if SPARSE:
    NUM_ITERS = 3
    BATCH_SIZE = 2240
if VECA:
    IMG_H, IMG_W, IMG_C = 84, 84, 6
    STATE_LENGTH = 512
    BATCH_SIZE = 32
else:
    STATE_LENGTH = 5
    IMG_H, IMG_W, IMG_C = 28, 28, 1
    BATCH_SIZE = 16
'''
UNSUPERVISED = True
if UNSUPERVISED:
    DYNAMICS = False
    if DYNAMICS:
        CONSIDER_U = True
        ACTION_LENGTH = 2
    else:
        CONSIDER_U = False
else:
    DYNAMICS = False
    CONSIDER_U = False
METHOD = "LARS"
if METHOD == "LARS":
    NUM_ITER = 5
    NO_HAM = False
if METHOD == "HAM":
    NO_HAM = False
    HAM_M = False
eta = 0.5
'''
