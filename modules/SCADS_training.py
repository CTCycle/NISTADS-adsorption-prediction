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
from modules.components.data_assets import PreProcessing
from modules.components.training_assets import ModelTraining, RealTimeHistory, SCADSModel, ModelValidation
import modules.global_variables as GlobVar
import configurations as cnf

# [LOAD ADSORPTION DATA FROM FILES]
#==============================================================================
# Load datasets from .csv files
#==============================================================================
print(f'''
-------------------------------------------------------------------------------
NIST Adsorption Dataset Training
-------------------------------------------------------------------------------
This module analyses the NIST adsorption dataset obtained by extracting data from 
NIST database online. The procedure will be separately performed on the single 
component isotherm dataset''')

file_loc = os.path.join(GlobVar.SCADS_path, 'SCADS_train_X.csv') 
train_X = pd.read_csv(file_loc, sep = ';', encoding = 'utf-8')
file_loc = os.path.join(GlobVar.SCADS_path, 'SCADS_train_Y.csv')
train_Y = pd.read_csv(file_loc, sep = ';', encoding = 'utf-8') 
file_loc = os.path.join(GlobVar.SCADS_path, 'SCADS_test_X.csv')
test_X = pd.read_csv(file_loc, sep = ';', encoding = 'utf-8') 
file_loc = os.path.join(GlobVar.SCADS_path, 'SCADS_test_Y.csv')
test_Y = pd.read_csv(file_loc, sep = ';', encoding = 'utf-8') 

# Load normalizer and encoders
#------------------------------------------------------------------------------
normalizer_path = os.path.join(GlobVar.SCADS_pp_path, 'features_normalizer.pkl')
with open(normalizer_path, 'rb') as file:
    features_normalizer = pickle.load(file)
normalizer_path = os.path.join(GlobVar.SCADS_pp_path, 'pressure_normalizer.pkl')
with open(normalizer_path, 'rb') as file:
    pressure_normalizer = pickle.load(file)
normalizer_path = os.path.join(GlobVar.SCADS_pp_path, 'uptake_normalizer.pkl')
with open(normalizer_path, 'rb') as file:
    uptake_normalizer = pickle.load(file)
encoder_path = os.path.join(GlobVar.SCADS_pp_path, 'adsorbent_encoder.pkl')
with open(encoder_path, 'rb') as file:
    adsorbent_encoder = pickle.load(file)
encoder_path = os.path.join(GlobVar.SCADS_pp_path, 'sorbates_encoder.pkl')
with open(encoder_path, 'rb') as file:
    sorbates_encoder = pickle.load(file) 

# [PREPARE INPUTS AND OUTPUTS]
#==============================================================================
# module for the selection of different operations
#==============================================================================

# identify columns
#------------------------------------------------------------------------------ 
continuous_features = ['temperature', 'mol_weight', 'complexity', 'covalent_units', 'H_acceptors', 'H_donors', 'heavy_atoms']
ads_col = ['adsorbent_name'] 
sorb_col = ['adsorbates_name']

# generate list of series and reshape them
#------------------------------------------------------------------------------ 
train_pressures = np.reshape([[float(x) for x in seq.split(' ')] for seq in train_X['pressure_in_Pascal'].to_list()], (-1, cnf.pad_length))
train_uptakes = np.reshape([[float(x) for x in seq.split(' ')] for seq in train_Y['uptake_in_mol/g'].to_list()], (-1, cnf.pad_length))
test_pressures = np.reshape([[float(x) for x in seq.split(' ')] for seq in test_X['pressure_in_Pascal'].to_list()], (-1, cnf.pad_length))
test_uptakes = np.reshape([[float(x) for x in seq.split(' ')] for seq in test_Y['uptake_in_mol/g'].to_list()], (-1, cnf.pad_length))

# prepare inputs 
#------------------------------------------------------------------------------
train_inputs = [train_X[continuous_features], train_X[ads_col], train_X[sorb_col], train_pressures]
test_inputs = [test_X[continuous_features], test_X[ads_col], test_X[sorb_col], test_pressures]
                   
# [CHECKSUM]
#==============================================================================
# ....
#==============================================================================      
print(f'''
-------------------------------------------------------------------------------
PREFLIGHT CHECK
-------------------------------------------------------------------------------
Train dataset size (samples) = {train_X.shape[0]}       
Test dataset size (samples) =  {test_X.shape[0]}
Number of features (continuous):   {train_X[continuous_features].shape[1]}
Number of features (categoricals): {train_X[continuous_features].shape[1]}
''')

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

# generate model save folder
#------------------------------------------------------------------------------
preprocessor = PreProcessing()
trainworker = ModelTraining(device=cnf.training_device, seed=cnf.seed) 
model_savepath = preprocessor.model_savefolder(GlobVar.model_path, 'SCADS')

# save model parameters in txt files
#------------------------------------------------------------------------------
parameters = {'Number of samples' : cnf.num_samples,
              'Train samples:' : train_X.shape[0],
              'Test samples:' : test_X.shape[0],
              'Padding' : 'Yes',
              'Pad sequence length' : cnf.pad_length,
              'Pad sequence value' : cnf.pad_value,
              'Batch size' : cnf.batch_size,
              'Learning rate' : cnf.learning_rate,
              'Epochs' : cnf.epochs}

trainworker.model_parameters(parameters, model_savepath)


# initialize model class
#------------------------------------------------------------------------------
unique_adsorbents = len(adsorbent_encoder.categories_[0])
unique_sorbates = len(sorbates_encoder.categories_[0])
num_features = len(continuous_features)     
SCADS_frame = SCADSModel(cnf.learning_rate, num_features, cnf.pad_length, 
                         cnf.pad_value, unique_adsorbents, unique_sorbates, 
                         cnf.embedding_dims, XLA_acceleration=cnf.XLA_acceleration)

# build model
#------------------------------------------------------------------------------
model = SCADS_frame.SCADS()
model.summary(expand_nested=True)
if GlobVar.generate_model_graph == True:
    plot_path = os.path.join(model_savepath, 'SCADS_model.png')       
    plot_model(model, to_file = plot_path, show_shapes = True, 
               show_layer_names = True, show_layer_activations = True, 
               expand_nested = True, rankdir = 'TB', dpi = 400) 

# [TRAINING WITH SCADS]
#==============================================================================
# Setting callbacks and training routine for the features extraction model. 
# use command prompt on the model folder and (upon activating environment), 
# use the bash command: python -m tensorboard.main --logdir tensorboard/
#==============================================================================

# initialize callbacks
#------------------------------------------------------------------------------
RTH_callback = RealTimeHistory(model_savepath, validation=True)

# training loop and model saving at end
#------------------------------------------------------------------------------
print(f'''Start model training for {cnf.epochs} epochs and batch size of {cnf.batch_size}
       ''')
if cnf.use_tensorboard == True:
    log_path = os.path.join(model_savepath, 'tensorboard')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
    training = model.fit(train_inputs, train_uptakes, validation_data=(test_inputs, test_uptakes),
                        epochs = cnf.epochs, batch_size=cnf.batch_size, 
                        callbacks = [RTH_callback, tensorboard_callback],
                        workers = 6, use_multiprocessing=True)
else:
    training = model.fit(train_inputs, train_uptakes, validation_data=(test_inputs, test_uptakes),
                        epochs=cnf.epochs, batch_size=cnf.batch_size, 
                        callbacks = [RTH_callback], workers = 6, use_multiprocessing=True)

model.save(model_savepath)
        
# [PERFORM QUICK EVALUATION]
#==============================================================================
# Training the LSTM model using the functions specified in the designated class.
# The model is saved in .h5 format at the end of the training
#==============================================================================        
print(f'''
-------------------------------------------------------------------------------
Model validation
-------------------------------------------------------------------------------
...
''')

validator = ModelValidation(model)

# sample subset of samples for validation
#------------------------------------------------------------------------------
val_X = train_X.sample(n=4, random_state = 92)
val_Y = train_Y.sample(n=4, random_state = 92)
val_pressures = [[float(x) for x in seq.split(' ')] for seq in val_X['pressure_in_Pascal'].to_list()]
val_uptakes = [[float(x) for x in seq.split(' ')] for seq in val_Y['uptake_in_mol/g'].to_list()]

# generate inputs and predict values
#------------------------------------------------------------------------------
val_input_pressures = np.reshape([[float(x) for x in seq.split(' ')] for seq in val_X['pressure_in_Pascal'].to_list()], (-1, GlobVar.pad_length))
val_inputs = [val_X[continuous_features], val_X[ads_col], val_X[sorb_col], val_input_pressures]
val_predictions = model.predict(val_inputs)

# remove padding and normalization from the series of true labels, predicted labels
# and pressure (series of inputs)
#------------------------------------------------------------------------------
absolute_pressures = []
absolute_uptakes = []
absolute_predictions = []

for series in val_pressures:
    true_length = len([x for x in series if x != cnf.pad_value])
    sliced_series = series[:true_length]
    reverse_series = [pressure_normalizer.inverse_transform(np.array(x).reshape(-1, 1)).flatten().tolist() for x in sliced_series]
    absolute_pressures.append(reverse_series)

for series in val_uptakes:
    true_length = len([x for x in series if x != cnf.pad_value])
    sliced_series = series[:true_length]
    reverse_series = [uptake_normalizer.inverse_transform(np.array(x).reshape(-1, 1)).flatten().tolist() for x in sliced_series]
    absolute_uptakes.append(reverse_series)

for series, abs in zip(val_predictions, absolute_uptakes):    
    sliced_series = series[:len(abs)]
    reverse_series = [uptake_normalizer.inverse_transform(np.array(x).reshape(-1, 1)).flatten().tolist() for x in sliced_series]
    absolute_predictions.append(reverse_series)

# perform validation
#------------------------------------------------------------------------------
validator.SCADS_validation(absolute_pressures, absolute_uptakes, absolute_predictions, model_savepath)





            


    
    




    
    










    




               

    












