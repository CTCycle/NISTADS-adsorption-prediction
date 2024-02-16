import os
import sys
import pandas as pd
import numpy as np
import pickle 
import tensorflow as tf
from keras.utils import plot_model

# set warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# add modules to sys path
#------------------------------------------------------------------------------
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import modules and classes
#------------------------------------------------------------------------------
from modules.components.model_assets import ModelTraining, RealTimeHistory, SCADSModel
import modules.global_variables as GlobVar
import configurations as cnf

# [LOAD ADSORPTION DATA FROM FILES]
#==============================================================================
# Load datasets from .csv files
#==============================================================================
print(f'''
-------------------------------------------------------------------------------
NISTADS model training
-------------------------------------------------------------------------------
This module analyses the NIST adsorption dataset obtained by extracting data from 
NIST database online. The procedure will be separately performed on the single 
component isotherm dataset''')

# identify columns
#------------------------------------------------------------------------------
features = ['temperature', 'mol_weight', 'complexity', 'covalent_units', 
            'H_acceptors', 'H_donors', 'heavy_atoms']
ads_col = ['adsorbent_name'] 
sorb_col = ['adsorbates_name']
P_col = 'pressure_in_Pascal'
Q_col = 'uptake_in_mol/g'

# load preprocessing module
#------------------------------------------------------------------------------
import modules.data_preprocessing

# load preprocessed csv files (train and test datasets)
#------------------------------------------------------------------------------
pp_path = os.path.join(GlobVar.model_folder_path, 'preprocessing')
file_loc = os.path.join(pp_path, 'train_X.csv') 
train_X = pd.read_csv(file_loc, sep = ';', encoding = 'utf-8')
file_loc = os.path.join(pp_path, 'train_Y.csv')
train_Y = pd.read_csv(file_loc, sep = ';',  encoding = 'utf-8') 
file_loc = os.path.join(pp_path, 'test_X.csv')
test_X = pd.read_csv(file_loc, sep = ';', encoding = 'utf-8') 
file_loc = os.path.join(pp_path, 'test_Y.csv')
test_Y = pd.read_csv(file_loc, sep = ';', encoding = 'utf-8') 

# Load normalizer and encoders
#------------------------------------------------------------------------------
normalizer_path = os.path.join(pp_path, 'features_normalizer.pkl')
with open(normalizer_path, 'rb') as file:
    features_normalizer = pickle.load(file)
normalizer_path = os.path.join(pp_path, 'pressure_normalizer.pkl')
with open(normalizer_path, 'rb') as file:
    pressure_normalizer = pickle.load(file)
normalizer_path = os.path.join(pp_path, 'uptake_normalizer.pkl')
with open(normalizer_path, 'rb') as file:
    uptake_normalizer = pickle.load(file)
encoder_path = os.path.join(pp_path, 'host_encoder.pkl')
with open(encoder_path, 'rb') as file:
    host_encoder = pickle.load(file)
encoder_path = os.path.join(pp_path, 'guest_encoder.pkl')
with open(encoder_path, 'rb') as file:
    guest_encoder = pickle.load(file)

# convert pressure and uptake series from strings to array of floats. Sequences
# were saved as strings before being saved in the .csv file
#------------------------------------------------------------------------------ 
train_pressure = train_X[P_col].apply(lambda x: np.array([float(val) for val in x.split()]))
test_pressure = test_X[P_col].apply(lambda x: np.array([float(val) for val in x.split()]))
train_output = train_Y[Q_col].apply(lambda x: np.array([float(val) for val in x.split()]))
test_output = test_Y[Q_col].apply(lambda x: np.array([float(val) for val in x.split()]))

# [BUILD SCADS MODEL]
#==============================================================================
# ....
#==============================================================================
print(f'''
-------------------------------------------------------------------------------
Model training
-------------------------------------------------------------------------------
...
''')
  
# initialize training device
#------------------------------------------------------------------------------
trainer = ModelTraining(device=cnf.training_device, use_mixed_precision=cnf.use_mixed_precision,
                        seed=cnf.seed) 

# Reshape sequences of pressure and uptakes to 2D arrays and create list of inputs
#------------------------------------------------------------------------------
train_pressure = np.stack(train_pressure.values)
test_pressure = np.stack(test_pressure.values)
train_output = np.stack(train_output.values)
test_output = np.stack(test_output.values)

# create list of inputs
#------------------------------------------------------------------------------
train_inputs = [train_X[features], train_X[ads_col], train_X[sorb_col], train_pressure]
test_inputs = [test_X[features], test_X[ads_col], test_X[sorb_col], test_pressure]

# determine number of classes and features, then initialize and build the model
#------------------------------------------------------------------------------
num_features = len(features)   
unique_adsorbents, unique_sorbates = len(host_encoder.categories_[0]), len(guest_encoder.categories_[0]) 
modelworker = SCADSModel(cnf.learning_rate, num_features, cnf.pad_length, 
                         cnf.pad_value, unique_adsorbents, unique_sorbates, 
                         cnf.embedding_dims, cnf.seed, XLA_acceleration=cnf.XLA_acceleration)

model = modelworker.get_model(summary=True) 

# generate graphviz plot for the model layout
#------------------------------------------------------------------------------
if cnf.generate_model_graph==True:
    plot_path = os.path.join(GlobVar.model_folder_path, 'model_layout.png')       
    plot_model(model, to_file = plot_path, show_shapes = True, 
               show_layer_names = True, show_layer_activations = True, 
               expand_nested = True, rankdir='TB', dpi = 400)

# [TRAIN SCADS MODEL]
#==============================================================================
# Setting callbacks and training routine for the XRAY captioning model. 
# to visualize tensorboard report, use command prompt on the model folder and 
# upon activating environment, use the bash command: 
# python -m tensorboard.main --logdir tensorboard/
#==============================================================================

# initialize real time plot callback 
#------------------------------------------------------------------------------
RTH_callback = RealTimeHistory(GlobVar.model_folder_path, validation=True)

# setting for validation data
#------------------------------------------------------------------------------
validation_data = (test_inputs, test_output)  

# initialize tensorboard
#------------------------------------------------------------------------------
if cnf.use_tensorboard == True:
    log_path = os.path.join(GlobVar.model_folder_path, 'tensorboard')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
    callbacks = [RTH_callback, tensorboard_callback]    
else:    
    callbacks = [RTH_callback]

# define and execute training loop, then save the model weights at end
#------------------------------------------------------------------------------
training = model.fit(x=train_inputs, y=train_output, batch_size=cnf.batch_size, 
                     validation_data=validation_data, epochs=cnf.epochs, 
                     verbose=1, shuffle=True, callbacks=callbacks, workers=6, use_multiprocessing=True)

model_files_path = os.path.join(GlobVar.model_folder_path, 'model')
model.save(model_files_path, save_format='tf')

print(f'''
-------------------------------------------------------------------------------
Training session is over. Model has been saved in folder {GlobVar.model_folder_name}
-------------------------------------------------------------------------------
''')

       
# save model data and model parameters in txt files
#------------------------------------------------------------------------------
parameters = {'Train_samples' : train_X.shape[0],
              'Test_samples' : test_X.shape[0],             
              'Sequence_lenght' : cnf.pad_length,
              'Padding_value' : cnf.pad_value,
              'Embedding_dimensions' : cnf.embedding_dims,             
              'Batch_size' : cnf.batch_size,
              'Learning_rate' : cnf.learning_rate,
              'Epochs' : cnf.epochs}

trainer.model_parameters(parameters, GlobVar.model_folder_path)

