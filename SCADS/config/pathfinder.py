from os.path import join, dirname, abspath 

PROJECT_DIR = dirname(dirname(abspath(__file__)))
DATA_PATH = join(PROJECT_DIR, 'data')
TRAIN_PATH = join(PROJECT_DIR, 'training')
CHECKPOINT_PATH = join(TRAIN_PATH, 'checkpoints')
INFERENCE_PATH = join(PROJECT_DIR, 'inference')



