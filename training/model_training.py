import os
import sys
import pandas as pd
import numpy as np
import pickle 
import tensorflow as tf
from keras.utils import plot_model
from tqdm import tqdm
tqdm.pandas()

# set warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# add parent folder path to the namespace
#------------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 

# import modules and classes
#------------------------------------------------------------------------------
from utils.preprocessing import PreProcessing, model_savefolder
from utils.models import ModelTraining, SCADSModel
from utils.callbacks import RealTimeHistory
import utils.global_paths as globpt
import configurations as cnf

# specify relative paths from global paths and create subfolders
#------------------------------------------------------------------------------
cp_path = os.path.join(globpt.train_path, 'checkpoints')
os.mkdir(cp_path) if not os.path.exists(cp_path) else None

# [PREPROCESS DATASET]
#==============================================================================
#==============================================================================

# create model folder and subfolder for preprocessed data 
#------------------------------------------------------------------------------
preprocessor = PreProcessing()
model_folder_path, model_folder_name = model_savefolder(cp_path, 'SCADS')

# create subfolder where to store preprocessed data
pp_path = os.path.join(model_folder_path, 'preprocessing')
os.mkdir(pp_path) if not os.path.exists(pp_path) else None

# load data from .csv files
#------------------------------------------------------------------------------
file_loc = os.path.join(globpt.data_path, 'SCADS_dataset.csv') 
df_adsorption = pd.read_csv(file_loc, sep=';', encoding = 'utf-8')

# transform series from unique string to lists
#------------------------------------------------------------------------------
df_adsorption['pressure_in_Pascal'] = df_adsorption['pressure_in_Pascal'].apply(lambda x : [float(f) for f in x.split()])
df_adsorption['uptake_in_mol_g'] = df_adsorption['uptake_in_mol_g'].apply(lambda x : [float(f) for f in x.split()])

# split dataset in train and test subsets
#------------------------------------------------------------------------------
train_X, test_X, train_Y, test_Y = preprocessor.split_dataset(df_adsorption, cnf.test_size, cnf.split_seed)

# [PREPROCESS DATASET: NORMALIZING AND ENCODING]
#==============================================================================
#==============================================================================
print('\nEncoding categorical variables')

# determine number of unique adsorbents and adsorbates from the train dataset
unique_adsorbents = train_X['adsorbent_name'].nunique() + 1
unique_sorbates = train_X['adsorbates_name'].nunique() + 1

# extract pretrained encoders to numerical indexes
train_X, test_X = preprocessor.GH_encoding(unique_adsorbents, unique_sorbates, train_X, test_X)

# extract pretrained encoders 
host_encoder = preprocessor.host_encoder
guest_encoder = preprocessor.guest_encoder

# normalize parameters (temperature, physicochemical properties) and sequences
#------------------------------------------------------------------------------
print('\nNormalizing continuous variables (temperature, physicochemical properties)\n')        

train_X, train_Y, test_X, test_Y = preprocessor.normalize_parameters(train_X, train_Y.to_frame(), 
                                                                     test_X, test_Y.to_frame())

# normalize sequences of pressure and uptake
train_X, test_X, pressure_normalizer = preprocessor.normalize_sequences(train_X, test_X, preprocessor.P_col)
train_Y, test_Y, uptake_normalizer = preprocessor.normalize_sequences(train_Y, test_Y, preprocessor.Q_col)

# extract pretrained parameters normalizer
param_normalizer = preprocessor.param_normalizer

# apply padding to the pressure and uptake series (default value is -1 to avoid
# interfering with real values)
#------------------------------------------------------------------------------ 
train_X = preprocessor.sequence_padding(train_X, preprocessor.P_col, cnf.pad_value, cnf.pad_length)
test_X = preprocessor.sequence_padding(test_X, preprocessor.P_col, cnf.pad_value, cnf.pad_length)
train_Y = preprocessor.sequence_padding(train_Y, preprocessor.Q_col, cnf.pad_value, cnf.pad_length)
test_Y = preprocessor.sequence_padding(test_Y, preprocessor.Q_col, cnf.pad_value, cnf.pad_length)

# prepare inputs and outputs
#------------------------------------------------------------------------------ 
train_parameters = train_X[preprocessor.parameters].values
train_hosts = train_X[preprocessor.ads_col].values
train_guests = train_X[preprocessor.sorb_col].values
train_pressures = np.array(train_X[preprocessor.P_col].to_list()).reshape(-1, cnf.pad_length)
train_uptakes = np.array(train_Y[preprocessor.Q_col].to_list()).reshape(-1, cnf.pad_length)

test_parameters = test_X[preprocessor.parameters].values
test_hosts = test_X[preprocessor.ads_col].values
test_guests = test_X[preprocessor.sorb_col].values
test_pressures = np.array(test_X[preprocessor.P_col].to_list()).reshape(-1, cnf.pad_length)
test_uptakes = np.array(test_Y[preprocessor.Q_col].to_list()).reshape(-1, cnf.pad_length)

# create list of inputs for both train and test datasets
train_inputs = [train_parameters, train_hosts, train_guests, train_pressures] 
test_inputs = [test_parameters, test_hosts, test_guests, test_pressures] 
validation_data = (test_inputs, test_uptakes)  

# save normalizers and encoders  
#------------------------------------------------------------------------------
normalizer_path = os.path.join(pp_path, 'parameters_normalizer.pkl')
with open(normalizer_path, 'wb') as file:
    pickle.dump(param_normalizer, file)
normalizer_path = os.path.join(pp_path, 'pressure_normalizer.pkl')
with open(normalizer_path, 'wb') as file:
    pickle.dump(pressure_normalizer, file)
normalizer_path = os.path.join(pp_path, 'uptake_normalizer.pkl')
with open(normalizer_path, 'wb') as file:
    pickle.dump(uptake_normalizer, file)
encoder_path = os.path.join(pp_path, 'host_encoder.pkl')
with open(encoder_path, 'wb') as file:
    pickle.dump(host_encoder, file) 
encoder_path = os.path.join(pp_path, 'guest_encoder.pkl')
with open(encoder_path, 'wb') as file:
    pickle.dump(guest_encoder, file) 

# save npy files
#------------------------------------------------------------------------------    
# Save training data
np.save(os.path.join(pp_path, 'train_parameters.npy'), train_parameters)
np.save(os.path.join(pp_path, 'train_hosts.npy'), train_hosts)
np.save(os.path.join(pp_path, 'train_guests.npy'), train_guests)
np.save(os.path.join(pp_path, 'train_pressures.npy'), train_pressures)
np.save(os.path.join(pp_path, 'train_uptakes.npy'), train_uptakes)

# Save testing data
np.save(os.path.join(pp_path, 'test_parameters.npy'), test_parameters)
np.save(os.path.join(pp_path, 'test_hosts.npy'), test_hosts)
np.save(os.path.join(pp_path, 'test_guests.npy'), test_guests)
np.save(os.path.join(pp_path, 'test_pressures.npy'), test_pressures)
np.save(os.path.join(pp_path, 'test_uptakes.npy'), test_uptakes)

# [BUILD SCADS MODEL]
#==============================================================================
#==============================================================================

# print report
#------------------------------------------------------------------------------ 
print(f'''
-------------------------------------------------------------------------------   
TRAINING REPORT
-------------------------------------------------------------------------------
Total number of experiments (train set): {train_X.shape[0]}
Total number of experiments (test set):  {test_X.shape[0]}
-------------------------------------------------------------------------------
''')
  
# initialize training device
#------------------------------------------------------------------------------
trainer = ModelTraining(device=cnf.training_device, use_mixed_precision=cnf.use_mixed_precision,
                        seed=cnf.seed) 

# determine number of classes and features, then initialize and build the model
#------------------------------------------------------------------------------
num_features = len(preprocessor.parameters)   
unique_adsorbents, unique_sorbates = len(host_encoder.categories_[0]), len(guest_encoder.categories_[0]) 
modelworker = SCADSModel(cnf.learning_rate, num_features, cnf.pad_length, 
                         cnf.pad_value, unique_adsorbents, unique_sorbates, 
                         cnf.embedding_dims, cnf.seed, XLA_acceleration=cnf.XLA_acceleration)

model = modelworker.get_model(summary=True) 

# generate graphviz plot for the model layout
#------------------------------------------------------------------------------
if cnf.generate_model_graph==True:
    plot_path = os.path.join(model_folder_path, 'model_layout.png')       
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
RTH_callback = RealTimeHistory(model_folder_path, validation=True)

# initialize tensorboard
#------------------------------------------------------------------------------
if cnf.use_tensorboard == True:
    log_path = os.path.join(model_folder_path, 'tensorboard')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
    callbacks = [RTH_callback, tensorboard_callback]    
else:    
    callbacks = [RTH_callback]

# define and execute training loop, then save the model weights at end
#------------------------------------------------------------------------------
multiprocessing = cnf.num_processors > 1
training = model.fit(x=train_inputs, y=train_uptakes, batch_size=cnf.batch_size, 
                     validation_data=validation_data, epochs=cnf.epochs, 
                     verbose=1, shuffle=True, callbacks=callbacks, workers=cnf.num_processors,
                     use_multiprocessing=multiprocessing)

model_files_path = os.path.join(model_folder_path, 'model')
model.save(model_files_path, save_format='tf')

print(f'''
-------------------------------------------------------------------------------
Training session is over. Model has been saved in folder {model_folder_name}
-------------------------------------------------------------------------------
''')
       
# save model data and model parameters in txt files
#------------------------------------------------------------------------------
parameters = {'train_samples' : train_X.shape[0],
              'test_samples' : test_X.shape[0],             
              'sequence_lenght' : cnf.pad_length,
              'padding_value' : cnf.pad_value,
              'embedding_dimensions' : cnf.embedding_dims,             
              'batch_size' : cnf.batch_size,
              'learning_rate' : cnf.learning_rate,
              'epochs' : cnf.epochs}

trainer.model_parameters(parameters, model_folder_path)

