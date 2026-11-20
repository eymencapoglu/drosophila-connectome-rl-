import os


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = r"C:\Users\Eymen\Desktop\SİNEK\sinek_from_scratch"

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "checkpoints"
)

LOG_DIR = os.path.join(
    PROJECT_ROOT,
    "logs"
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

os.makedirs(
    LOG_DIR,
    exist_ok=True
)


# ============================================================
# WORLD
# ============================================================

WORLD_WIDTH = 120.0
WORLD_HEIGHT = 100.0

DT = 1.0 / 30.0


# ============================================================
# PHYSICS
# ============================================================

GRAVITY = 9.81

AIR_DRAG = 0.985

ANGULAR_DRAG = 0.92

MAX_SPEED = 30.0

MAX_ANGULAR_SPEED = 4.0


# ============================================================
# FLY
# ============================================================

FLY_RADIUS = 2.0

FLY_MASS = 1.0

MAX_LIFT = 13.0

MAX_THRUST = 7.0

MAX_TORQUE = 3.0


# ============================================================
# WIND
# ============================================================

WIND_BASE_X = 0.0
WIND_BASE_Y = 0.0

WIND_VARIATION = 1.5


# ============================================================
# VISION
# ============================================================

VISION_WIDTH = 64
VISION_HEIGHT = 64

VISION_CHANNELS = 2

VISION_RANGE = 65.0

FOV_DEGREES = 110.0

# Vision refresh rate.
# Environment is still 30 Hz.
VISION_UPDATE_EVERY = 1


# ============================================================
# SUGAR
# ============================================================

SUGAR_RADIUS = 2.5

SUGAR_REWARD = 100.0

# Small continuous reward.
# This is deliberately much smaller than the terminal reward.
DISTANCE_REWARD_SCALE = 0.04

# Penalty for wasting time.
TIME_PENALTY = 0.0015

# Energy expenditure penalty.
ENERGY_COST_SCALE = 0.0008

# Reward for staying airborne.
FLIGHT_REWARD = 0.003


# ============================================================
# EPISODE
# ============================================================

MAX_STEPS_PER_EPISODE = 1200


# ============================================================
# SNN
# ============================================================

SNN_SIZE = 256

SNN_BETA = 0.92

SNN_THRESHOLD = 1.0

SNN_RESET = 0.0

SNN_TIME_STEPS = 6

SNN_SYNAPTIC_DECAY = 0.85


# ============================================================
# CNN
# ============================================================

CNN_FEATURE_SIZE = 256

PROPRIO_FEATURE_SIZE = 48


# ============================================================
# RL
# ============================================================

ACTION_SIZE = 9

LEARNING_RATE = 2.5e-4

GAMMA = 0.99

GAE_LAMBDA = 0.95

PPO_CLIP = 0.2

PPO_EPOCHS = 4

PPO_BATCH_SIZE = 32

ROLLOUT_STEPS = 2048

UPDATES = 1000

ENTROPY_COEF = 0.01

VALUE_COEF = 0.5

MAX_GRAD_NORM = 0.5


# ============================================================
# RECURRENT TRAINING
# ============================================================

SEQUENCE_LENGTH = 32


# ============================================================
# VISUALIZATION
# ============================================================

VISUALIZE_EVERY = 4

DASHBOARD_UPDATE_HZ = 15

BRAIN_NEURONS_TO_DRAW = 256

BRAIN_CONNECTIONS_TO_DRAW = 500

BRAIN_SPIKE_HISTORY = 60


# ============================================================
# CHECKPOINTING
# ============================================================

CHECKPOINT_EVERY = 10

SAVE_BEST = True


# ============================================================
# RANDOM
# ============================================================

SEED = 42