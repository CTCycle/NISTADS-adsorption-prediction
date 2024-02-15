# define other variables for training
#------------------------------------------------------------------------------
use_mixed_precision = True
generate_model_graph = True
use_tensorboard = False
XLA_acceleration = False

# Define variables for the training
#------------------------------------------------------------------------------
seed = 42
training_device = 'GPU'
embedding_dims = 256
epochs = 500
learning_rate = 0.0001
batch_size = 512
test_size = 0.1

# Define variables for the dataset
#------------------------------------------------------------------------------
num_samples = 20000
pad_length = 40
pad_value = -1
min_points = 5
max_pressure = 2000000
max_uptake = 30

