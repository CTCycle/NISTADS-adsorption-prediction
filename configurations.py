# Advanced settings for training 
#------------------------------------------------------------------------------
use_mixed_precision = False
use_tensorboard = False
XLA_acceleration = False
training_device = 'GPU'
num_processors = 6

# Settings for training routine
#------------------------------------------------------------------------------
epochs = 6000
learning_rate = 0.0001
batch_size = 1024

# Model settings
#------------------------------------------------------------------------------
embedding_dims = 256
generate_model_graph = True

# Settings for training data 
#------------------------------------------------------------------------------
num_samples = 30000 #set higher than available samples to take all of them
test_size = 0.1
pad_length = 40
pad_value = -1
min_points = 6
max_pressure = 10e06
max_uptake = 10

# General settings
#------------------------------------------------------------------------------
seed = 54
split_seed = 425