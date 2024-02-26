import os
import sys
import pandas as pd
import numpy as np
import pickle 
import tensorflow as tf
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
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
from components.data_assets import PreProcessing
from components.model_assets import ModelTraining, RealTimeHistory, SCADSModel
import components.global_paths as globpt
import configurations as cnf

# specify relative paths from global paths and create subfolders
#------------------------------------------------------------------------------
cp_path = os.path.join(globpt.train_path, 'checkpoints')
os.mkdir(cp_path) if not os.path.exists(cp_path) else None

# [PREPROCESS DATA]
#==============================================================================
#==============================================================================
print(f'''
-------------------------------------------------------------------------------
NISTADS model training
-------------------------------------------------------------------------------
This module analyses the NIST adsorption dataset obtained by extracting data from 
NIST database online. The procedure will be separately performed on the single 
component isotherm dataset''')

# create model folder and subfolder for preprocessed data 
#------------------------------------------------------------------------------
preprocessor = PreProcessing()
model_folder_path = preprocessor.model_savefolder(cp_path, 'SCADS')
model_folder_name = preprocessor.folder_name

# create subfolder where to store preprocessed data
pp_path = os.path.join(model_folder_path, 'preprocessing')
os.mkdir(pp_path) if not os.path.exists(pp_path) else None

# load data from .csv files
#------------------------------------------------------------------------------
file_loc = os.path.join(globpt.data_path, 'SCADS_dataset.csv') 
df_adsorption = pd.read_csv(file_loc, sep=';', encoding = 'utf-8')
file_loc = os.path.join(globpt.data_path, 'adsorbates_dataset.csv') 
df_adsorbates = pd.read_csv(file_loc, sep=';', encoding = 'utf-8')
file_loc = os.path.join(globpt.data_path, 'adsorbents_dataset.csv') 
df_adsorbents = pd.read_csv(file_loc, sep=';', encoding = 'utf-8')

# add molecular properties based on PUG CHEM API data
#------------------------------------------------------------------------------ 
print(f'''Preprocessing adsorption isotherm data\n''')
dataset = preprocessor.guest_properties(df_adsorption, df_adsorbates)

# filter experiments by allowed uptake units
#------------------------------------------------------------------------------ 
dataset = dataset[dataset[preprocessor.Q_unit_col].isin(preprocessor.valid_units)]

# filter experiments by allowed uptake units and convert pressure and uptake 
# to Pa (pressure) and mol/kg (uptake) 
#------------------------------------------------------------------------------ 
dataset = dataset[dataset[preprocessor.Q_unit_col].isin(preprocessor.valid_units)]
dataset[preprocessor.P_col] = dataset.progress_apply(lambda x : preprocessor.pressure_converter(x[preprocessor.P_unit_col], 
                                                                                                x['pressure']), 
                                                                                                axis = 1)
dataset[preprocessor.Q_col] = dataset.progress_apply(lambda x : preprocessor.uptake_converter(x[preprocessor.Q_unit_col], 
                                                                                              x['adsorbed_amount'], 
                                                                                              x['mol_weight']), 
                                                                                              axis = 1)


# filter the dataset to remove experiments with units are outside desired boundaries, 
# such as experiments with negative values of temperature, pressure and uptake
#------------------------------------------------------------------------------ 
dataset = dataset[dataset['temperature'].astype(int) > 0]
dataset = dataset[dataset[preprocessor.P_col].astype(float).between(0.0, cnf.max_pressure)]
dataset = dataset[dataset[preprocessor.Q_col].astype(float).between(0.0, cnf.max_uptake)]

# Aggregate values using groupby function in order to group the dataset by experiments
#------------------------------------------------------------------------------ 
aggregate_dict = {'temperature' : 'first',                  
                  'adsorbent_name' : 'first',
                  'adsorbates_name' : 'first',                  
                  'complexity' : 'first',                  
                  'mol_weight' : 'first',
                  'covalent_units' : 'first',
                  'H_acceptors' : 'first',
                  'H_donors' : 'first',
                  'heavy_atoms' : 'first', 
                  'pressure_in_Pascal' : list,
                  'uptake_in_mol/g' : list}
   
# group dataset by experiments and drop filename column as it is not necessary
dataset_grouped = dataset.groupby('filename', as_index=False).agg(aggregate_dict)
dataset_grouped.drop(columns='filename', axis=1, inplace=True)
total_num_exp = dataset_grouped.shape[0]

# remove series of pressure/uptake with less than X points, drop rows containing nan
# values and select a subset of samples for training
#------------------------------------------------------------------------------ 
dataset_grouped = dataset_grouped[dataset_grouped[preprocessor.P_col].apply(lambda x: len(x)) >= cnf.min_points]
dataset_grouped = dataset_grouped.dropna()

# check to avoid errors when selecting number of samples higher than effectively 
# available samples. If less are available, the entire dataset is selected
if cnf.num_samples < total_num_exp:
    dataset_grouped = dataset_grouped.sample(n=cnf.num_samples, random_state=30).reset_index()

# check pressure and uptake series and force them to converge at zero at the beginning: 
# f(x) = 0 for x = 0
#------------------------------------------------------------------------------
dataset_grouped = dataset_grouped.apply(preprocessor.zero_convergence, args=(preprocessor.P_col, preprocessor.Q_col), axis=1)

# split train and test dataset
#------------------------------------------------------------------------------ 
inputs = dataset_grouped[[x for x in dataset_grouped.columns if x != preprocessor.Q_col]]
labels = dataset_grouped[preprocessor.Q_col]
train_X, test_X, train_Y, test_Y = train_test_split(inputs, labels, test_size=cnf.test_size, 
                                                    random_state=cnf.seed, shuffle=True, stratify=None) 

# normalize all continuous variables (temperature, physicochemical properties,
# pressure, uptake)
#------------------------------------------------------------------------------
print(f'''Normalizing continuous variables\n''') 
train_Y, test_Y = train_Y.to_frame(), test_Y.to_frame()
train_X, train_Y, test_X, test_Y = preprocessor.normalize_data(train_X, train_Y, test_X, test_Y)
features_normalizer = preprocessor.features_normalizer
pressure_normalizer = preprocessor.pressure_normalizer
uptake_normalizer = preprocessor.uptake_normalizer

# encode categorical variables
#------------------------------------------------------------------------------
print(f'''Encoding categorical variables\n''')
unique_adsorbents = train_X['adsorbent_name'].nunique() + 1
unique_sorbates = train_X['adsorbates_name'].nunique() + 1

train_X, test_X = preprocessor.data_encoding(unique_adsorbents, unique_sorbates, train_X, test_X)
host_encoder = preprocessor.host_encoder
guest_encoder = preprocessor.guest_encoder

# apply padding to the pressure and uptake series
#------------------------------------------------------------------------------ 
train_X[preprocessor.P_col] = train_X[preprocessor.P_col].apply(lambda x : preprocessor.sequence_padding(x, pad_length=cnf.pad_length, pad_value=cnf.pad_value))
test_X[preprocessor.P_col] = test_X[preprocessor.P_col].apply(lambda x : preprocessor.sequence_padding(x, pad_length=cnf.pad_length, pad_value=cnf.pad_value))
train_Y = train_Y.apply(lambda x : preprocessor.sequence_padding(x, pad_length=cnf.pad_length, pad_value=cnf.pad_value))
test_Y = test_Y.apply(lambda x : preprocessor.sequence_padding(x, pad_length=cnf.pad_length, pad_value=cnf.pad_value))

# print report
#------------------------------------------------------------------------------ 
print(f'''
-------------------------------------------------------------------------------   
PREPROCESSING REPORT
-------------------------------------------------------------------------------
Number of experiments before filtering: {df_adsorption.groupby('filename').ngroup().nunique()}
Number of experiments upon filtering:   {dataset.groupby('filename').ngroup().nunique()}
Number of experiments removed:          {df_adsorption.groupby('filename').ngroup().nunique() - dataset.groupby('filename').ngroup().nunique()}
-------------------------------------------------------------------------------
Total number of experiments:             {total_num_exp}
Selected number of experiments:          {cnf.num_samples}
Actual number of experiments:            {dataset_grouped.shape[0]}
Total number of experiments (train set): {train_X.shape[0]}
Total number of experiments (test set):  {test_X.shape[0]}
-------------------------------------------------------------------------------
''')

# save normalizers and encoders  
#------------------------------------------------------------------------------
normalizer_path = os.path.join(pp_path, 'features_normalizer.pkl')
with open(normalizer_path, 'wb') as file:
    pickle.dump(features_normalizer, file)
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

# create deep copy of the dataframes to avoid modifying the original data.
# the copies will be used to save the preprocessed data (this way the sequences
# of pressur and uptakes can be converted to a single string per row)
#------------------------------------------------------------------------------
file_loc = os.path.join(pp_path, 'train_X.csv')
train_X_csv = train_X.copy(deep=True)
train_X_csv[preprocessor.P_col] = train_X_csv[preprocessor.P_col].apply(lambda x : ' '.join([str(f) for f in x]))
train_X_csv.to_csv(file_loc, index=False, sep=';', encoding='utf-8')
file_loc = os.path.join(pp_path, 'train_Y.csv')
train_Y_csv = train_Y.copy(deep=True)
train_Y_csv = train_Y_csv.apply(lambda x : ' '.join([str(f) for f in x]))
train_Y_csv.to_csv(file_loc, index=False, sep=';', encoding='utf-8')
file_loc = os.path.join(pp_path, 'test_X.csv')
test_X_csv = test_X.copy(deep=True)
test_X_csv[preprocessor.P_col] = test_X_csv[preprocessor.P_col].apply(lambda x : ' '.join([str(f) for f in x]))
test_X_csv.to_csv(file_loc, index=False, sep=';', encoding='utf-8')
file_loc = os.path.join(pp_path, 'test_Y.csv')
test_Y_csv = test_Y.copy(deep=True)
test_Y_csv = test_Y_csv.apply(lambda x : ' '.join([str(f) for f in x]))
test_Y_csv.to_csv(file_loc, index=False, sep=';', encoding='utf-8')

# [BUILD SCADS MODEL]
#==============================================================================
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

# determine number of classes and features, then initialize and build the model
#------------------------------------------------------------------------------
num_features = len(preprocessor.features)   
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

# Reshape sequences of pressure and uptakes to 2D arrays and create list of inputs
#------------------------------------------------------------------------------
train_pressure = np.stack(train_X[preprocessor.P_col].values)
test_pressure = np.stack(test_X[preprocessor.P_col].values)
train_output = np.stack(train_Y.values)
test_output = np.stack(test_Y.values)

# create list of inputs for both train and test datasets
train_inputs = [train_X[preprocessor.features], train_X[preprocessor.ads_col], train_X[preprocessor.sorb_col], train_pressure]
test_inputs = [test_X[preprocessor.features], test_X[preprocessor.ads_col], test_X[preprocessor.sorb_col], test_pressure]
validation_data = (test_inputs, test_output)  

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
training = model.fit(x=train_inputs, y=train_output, batch_size=cnf.batch_size, 
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

