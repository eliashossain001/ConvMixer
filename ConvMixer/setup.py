import os

BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = '/home/elias/clips/imagenet-mini/train/'
VAL_DIR = '/home/elias/clips/imagenet-mini/val/'
NUM_CLASSES = len(os.listdir(TRAIN_DIR))
INPUT_SHAPE = (224, 224, 3)
