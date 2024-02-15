import os
import sys
import pandas as pd
import numpy as np
import pickle 

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
from modules.components.model_assets import ModelTraining
import modules.global_variables as GlobVar
import configurations as cnf

# [LOAD ADSORPTION DATA FROM FILES]
#==============================================================================
# Load datasets from .csv files
#==============================================================================
print(f'''
-------------------------------------------------------------------------------
SCADS predictions
-------------------------------------------------------------------------------
This module analyses the NIST adsorption dataset obtained by extracting data from 
NIST database online. The procedure will be separately performed on the single 
component isotherm dataset''')

file_loc = os.path.join(GlobVar.pred_path, 'SCADS_inputs.csv') 
df_predictions = pd.read_csv(file_loc, sep = ';', encoding = 'utf-8')
file_loc = os.path.join(GlobVar.data_path, 'adsorbates_dataset.csv') 
df_adsorbates = pd.read_csv(file_loc, sep = ';', encoding = 'utf-8')

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

# [ADD MOLECULAR PROPERTIES TO PREDICTIONS DATASET AND CONVERT UNITS]
#==============================================================================
# Retrieve chemical properties for adsorbates and adsorbents and add new columns to
# the respective datasets. Such information will be used for calculation purposes
#==============================================================================
print(f'''STEP 1 - Add molecular properties 
      ''')
preprocessor = PreProcessing()
df_inputs = preprocessor.properties_assigner(df_predictions, df_adsorbates)
print()

# [PREPARE INPUTS AND OUTPUTS]
#==============================================================================
# module for the selection of different operations
#==============================================================================

# identify columns
#------------------------------------------------------------------------------ 
ads_col = ['adsorbent_name'] 
sorb_col = ['adsorbates_name']
continuous_features = ['temperature', 'mol_weight', 'complexity', 'covalent_units', 
                       'H_acceptors', 'H_donors', 'heavy_atoms']


# [GROUP DATASET BY EXPERIMENT]
#==============================================================================
# Aggregate values using groupby function and perform other operations on dataset variables
#==============================================================================
aggregate_dict = {'temperature' : 'first',                  
                  'adsorbent_name' : 'first',
                  'adsorbates_name' : 'first',                  
                  'complexity' : 'first',                  
                  'mol_weight' : 'first',
                  'covalent_units' : 'first',
                  'H_acceptors' : 'first',
                  'H_donors' : 'first',
                  'heavy_atoms' : 'first', 
                  'pressure' : list}

# perform grouping based on aggregated dictionary
#------------------------------------------------------------------------------          
df_predictions_grouped = df_inputs.groupby('filename', as_index = False).agg(aggregate_dict)
df_predictions_grouped.drop(columns='filename', axis = 1, inplace=True)
total_num_exp = df_predictions_grouped.shape[0]

# [NORMALIZE AND ENCODE VARIABLES]
#==============================================================================
# encodes strings as ordinal indexes and normalize continuous 
# variable within a defined range 
#==============================================================================
print(f'''STEP 2 - Use saved normalizers and encoders for preprocessing
      ''')

# enforce float type and normalize
#------------------------------------------------------------------------------ 
df_predictions_grouped[continuous_features] = df_predictions_grouped[continuous_features].astype(float)        
df_predictions_grouped[continuous_features] = features_normalizer.transform(df_predictions_grouped[continuous_features])

# fit encoder on the train sets
#------------------------------------------------------------------------------        
df_predictions_grouped[ads_col] = adsorbent_encoder.transform(df_predictions_grouped[ads_col])
df_predictions_grouped[sorb_col] = sorbates_encoder.transform(df_predictions_grouped[sorb_col])
unique_adsorbents = len(adsorbent_encoder.categories_[0])
unique_sorbates = len(sorbates_encoder.categories_[0])

# apply normalization to pressure series
#------------------------------------------------------------------------------ 
df_predictions_grouped['max_pressure'] = df_predictions_grouped['pressure'].apply(lambda x : max(x))
df_predictions_grouped['processed_pressure'] = df_predictions_grouped['pressure'].apply(lambda x : preprocessor.series_preprocessing(x, pad_length=GlobVar.pad_length, pad_value=GlobVar.pad_value))

# [PRINT REPORT]
#==============================================================================
# encodes strings as ordinal indexes 
#==============================================================================
print(f'''
-------------------------------------------------------------------------------
Total number of experiments: {total_num_exp}
-------------------------------------------------------------------------------
''')

# [LOAD PRETRAINED SCADS MODEL]
#==============================================================================
# ....
#==============================================================================
print(f'''STEP 3 - Load pretrained model
      ''')
trainworker = ModelTraining(device = GlobVar.training_device) 
model = trainworker.load_pretrained_model(GlobVar.model_path)
model.summary(expand_nested=True)

# [PERFORM PREDICTIONS]
#==============================================================================
# ....
#==============================================================================
print(f'''STEP 4 - Perform predictions and save files
      ''')

# organise inputs
#------------------------------------------------------------------------------ 
pred_pressures = np.reshape(df_predictions_grouped['processed_pressure'].to_list(), (-1, GlobVar.pad_length))
pred_inputs = [df_predictions_grouped[continuous_features], 
               df_predictions_grouped[ads_col],
               df_predictions_grouped[sorb_col],
               pred_pressures]

# extract info on the maximal pressure
#------------------------------------------------------------------------------ 
max_pressures = df_predictions_grouped['max_pressure'].to_list()
P_series_size = [len(x) for x in df_predictions_grouped['pressure'].to_list()]

predictions = model.predict(pred_inputs)
corrected_predictions = [pred[:len] for pred, len in zip(predictions, P_series_size)]

df_predictions['predicted adsorption'] = np.concatenate(corrected_predictions).tolist()


# [SAVE FILES]
#==============================================================================
# Save the trained preprocessing systems (normalizer and encoders) for further use 
#==============================================================================
file_loc = os.path.join(GlobVar.pred_path, 'SCADS_predictions.csv')
df_predictions.to_csv(file_loc, index = False, sep = ';', encoding = 'utf-8')


