# Advanced settings for training 
#------------------------------------------------------------------------------
use_mixed_precision = False
use_tensorboard = False
XLA_acceleration = False
training_device = 'GPU'
num_processors = 6

# Settings for training routine
#------------------------------------------------------------------------------
epochs = 2500
learning_rate = 0.0001
batch_size = 512

# Model settings
#------------------------------------------------------------------------------
embedding_dims = 400
generate_model_graph = True

# Settings for training data 
#------------------------------------------------------------------------------
num_samples = 18000
test_size = 0.1
pad_length = 40
pad_value = -1
min_points = 10
max_pressure = 2000000
max_uptake = 20

# General settings
#------------------------------------------------------------------------------
seed = 514