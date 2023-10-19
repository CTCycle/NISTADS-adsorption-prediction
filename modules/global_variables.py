import os

# Define paths for the script
#------------------------------------------------------------------------------
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')    
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
pred_path = os.path.join(save_path, r'predictions')
SCADS_path = os.path.join(save_path, r'SCADS modeling')
SCADS_pp_path = os.path.join(SCADS_path, r'preprocessing')

# Create folders if they do not exist
#------------------------------------------------------------------------------
if not os.path.exists(data_path):
    os.mkdir(data_path)
if not os.path.exists(save_path):
    os.mkdir(save_path) 
if not os.path.exists(model_path):
    os.mkdir(model_path) 
if not os.path.exists(pred_path):
    os.mkdir(pred_path) 
if not os.path.exists(SCADS_path):
    os.mkdir(SCADS_path) 
if not os.path.exists(SCADS_pp_path):
    os.mkdir(SCADS_pp_path) 

# Define variables for the dataset
#------------------------------------------------------------------------------
num_samples = 20000
test_size = 0.2
pad_length = 50
pad_value = 20

# normalization is to be intended for the final units, which are mol/g for the uptakes
# and Pascal for the pressure (dataset filtered at 20 bar maximal pressure)
#------------------------------------------------------------------------------
pressure_ceil = 2000000
uptake_ceil = 20
min_points = 5

# Define variables for the training
#------------------------------------------------------------------------------
seed = 42
training_device = 'GPU'
embedding_dims = 800
epochs = 2000
learning_rate = 0.0001
batch_size = 2500

# define other variables for training
#------------------------------------------------------------------------------
use_tensorboard = True
generate_model_graph = True
XLA_acceleration = True
use_mixed_precision = True
