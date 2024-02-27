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
from components.data_assets import PreProcessing
from components.model_assets import ModelTraining, RealTimeHistory, SCADSModel
import components.global_paths as globpt
import configurations as cnf

# specify relative paths from global paths and create subfolders
#------------------------------------------------------------------------------
cp_path = os.path.join(globpt.train_path, 'checkpoints')
os.mkdir(cp_path) if not os.path.exists(cp_path) else None

# [GENERATE CLEAN AGGREGATED DATASET]
#==============================================================================
#==============================================================================
print(f'''
-------------------------------------------------------------------------------
NISTADS model training
-------------------------------------------------------------------------------
...
''')

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

# add molecular properties based on PUGCHEM API data
#------------------------------------------------------------------------------ 
print(f'''Adding physicochemical properties from guest species dataset\n''')
dataset = preprocessor.add_guest_properties(df_adsorption, df_adsorbates)

# filter experiments leaving only valid uptake and pressure units, then convert 
# pressure and uptake to Pa (pressure) and mol/kg (uptake)
#------------------------------------------------------------------------------

# filter experiments by pressure and uptake units 
dataset = dataset[dataset[preprocessor.Q_unit_col].isin(preprocessor.valid_units)]

# convert pressures to Pascal
dataset[preprocessor.P_col] = dataset.progress_apply(lambda x : preprocessor.pressure_converter(x[preprocessor.P_unit_col], 
                                                                                                x['pressure']), 
                                                                                                axis = 1)
# convert uptakes to mol/g
dataset[preprocessor.Q_col] = dataset.progress_apply(lambda x : preprocessor.uptake_converter(x[preprocessor.Q_unit_col], 
                                                                                              x['adsorbed_amount'], 
                                                                                              x['mol_weight']), 
                                                                                              axis = 1)

# further filter the dataset to remove experiments which values are outside desired boundaries, 
# such as experiments with negative temperature, pressure and uptake values
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

# remove series of pressure/uptake with less than X points, drop rows containing nan
# values and select a subset of samples for training
#------------------------------------------------------------------------------ 
dataset_grouped = dataset_grouped[~dataset_grouped[preprocessor.P_col].apply(lambda x: all(elem == 0 for elem in x))]
dataset_grouped = dataset_grouped[dataset_grouped[preprocessor.P_col].apply(lambda x: len(x)) >= cnf.min_points]
dataset_grouped = dataset_grouped.dropna()
total_experiments = dataset_grouped.shape[0]

# check to avoid errors when selecting number of samples higher than effectively 
# available samples. If less are available, the entire dataset is selected
if cnf.num_samples < total_experiments:
    dataset_grouped = dataset_grouped.sample(n=cnf.num_samples, random_state=30).reset_index()

# [MACHINE LEARNING PREPROCESSING]
#==============================================================================
#==============================================================================

# preprocess sequences to remove leading 0 values (some experiments may have several
# zero measurements at the start), make sure that every experiment starts with pressure
# of 0 Pa and uptake of 0 mol/g (effectively converges to zero)
#------------------------------------------------------------------------------
dataset_grouped[[preprocessor.P_col, preprocessor.Q_col]] = dataset_grouped.apply(lambda row: 
                 preprocessor.remove_leading_zeros(row[preprocessor.P_col],
                 row[preprocessor.Q_col]), axis=1, result_type='expand')

# split dataset in train and test subsets
#------------------------------------------------------------------------------
train_X, test_X, train_Y, test_Y = preprocessor.split_dataset(dataset_grouped, cnf.test_size, cnf.seed)

# encode categorical variables (adsorbents and adsorbates names)
#------------------------------------------------------------------------------
print('''Encoding categorical variables\n''')

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
print('''\nNormalizing continuous variables (temperature, physicochemical properties)\n''')        

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
Total number of experiments:             {total_experiments}
Selected number of experiments:          {cnf.num_samples}
Actual number of experiments:            {dataset_grouped.shape[0]}
Total number of experiments (train set): {train_X.shape[0]}
Total number of experiments (test set):  {test_X.shape[0]}
-------------------------------------------------------------------------------
''')

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

