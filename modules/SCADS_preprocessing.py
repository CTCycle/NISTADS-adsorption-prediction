import os
import sys
import pandas as pd
import pickle 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder 
from tqdm import tqdm
tqdm.pandas()


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
from modules.components.data_classes import PreProcessing
import modules.global_variables as GlobVar

# [LOAD ADSORPTION DATA FROM FILES]
#==============================================================================
# Load datasets from .csv files
#==============================================================================
file_loc = os.path.join(GlobVar.data_path, 'SCADS_dataset.csv') 
df_adsorption = pd.read_csv(file_loc, sep = ';', encoding = 'utf-8')
file_loc = os.path.join(GlobVar.data_path, 'adsorbates_dataset.csv') 
df_adsorbates = pd.read_csv(file_loc, sep = ';', encoding = 'utf-8')
file_loc = os.path.join(GlobVar.data_path, 'adsorbents_dataset.csv') 
df_adsorbents = pd.read_csv(file_loc, sep = ';', encoding = 'utf-8')

print(f'''
-------------------------------------------------------------------------------
NIST Adsorption Dataset Preprocessing
-------------------------------------------------------------------------------
This module analyses the NIST adsorption dataset obtained by extracting data from 
NIST database online. The procedure will be separately performed on the single 
component isotherm dataset
''')

# [FILTER DATASET]
#==============================================================================
# Retrieve chemical properties for adsorbates and adsorbents and add new columns to
# the respective datasets. Such information will be used for calculation purposes
#==============================================================================
print(f'''STEP 1 - Selecting specific units and convert uptakes and pressures
      ''')
preprocessor = PreProcessing()

# add molecular properties
#------------------------------------------------------------------------------ 
df_adsorption = preprocessor.properties_assigner(df_adsorption, df_adsorbates)

# filter experiments by units
#------------------------------------------------------------------------------ 
valid_units = ['mmol/g', 'mol/kg', 'mol/g', 'mmol/kg', 'mg/g', 'g/g', 
               'wt%', 'g Adsorbate / 100g Adsorbent', 'g/100g', 'ml(STP)/g', 
               'cm3(STP)/g']
df_adsorption_unit = df_adsorption[df_adsorption['adsorptionUnits'].isin(valid_units)]

# convert pressure and uptake using designated functions
#------------------------------------------------------------------------------ 
df_adsorption_unit['pressure_in_Pascal'] = df_adsorption_unit.progress_apply(lambda x : preprocessor.pressure_converter(x['pressureUnits'], x['pressure']), axis = 1)
df_adsorption_unit['uptake_in_mol/g'] = df_adsorption_unit.progress_apply(lambda x : preprocessor.uptake_converter(x['adsorptionUnits'], x['adsorbed_amount'], x['mol_weight']), axis = 1)
print()

# filter SCADS dataset to remove unvalid adsorption units, and experiments with
# negative values of temperature, pressure and uptake
#------------------------------------------------------------------------------ 
df_adsorption_ML = df_adsorption_unit[df_adsorption_unit['temperature'].astype(int) > 0]
df_adsorption_ML = df_adsorption_ML[df_adsorption_ML['pressure_in_Pascal'].astype(float).between(0.0, GlobVar.pressure_ceil)]
df_adsorption_ML = df_adsorption_ML[df_adsorption_ML['uptake_in_mol/g'].astype(float).between(0.0, GlobVar.uptake_ceil)]

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
                  'pressure_in_Pascal' : list,
                  'uptake_in_mol/g' : list}

# perform grouping based on aggregated dictionary
#------------------------------------------------------------------------------          
df_adsorption_grouped = df_adsorption_ML.groupby('filename', as_index = False).agg(aggregate_dict)
df_adsorption_grouped.drop(columns='filename', axis = 1, inplace=True)
total_num_exp = df_adsorption_grouped.shape[0]

# remove series of pressure/uptake with less than X points
#------------------------------------------------------------------------------ 
df_adsorption_grouped = df_adsorption_grouped[df_adsorption_grouped['pressure_in_Pascal'].apply(lambda x: len(x)) >= GlobVar.min_points]

# select data fraction for preprocessing
#------------------------------------------------------------------------------
df_adsorption_grouped = df_adsorption_grouped.sample(n = GlobVar.num_samples, random_state=30)

# [SPLIT DATASET INTO TRAIN, VALIDATION AND TEST]
#==============================================================================
# Split datasets into training, test and validation partitions
#==============================================================================
print(f'''STEP 2 - Splitting adsorption dataset into train and test
      ''')

# split X and Y series
#------------------------------------------------------------------------------ 
dataset_X = df_adsorption_grouped[[x for x in df_adsorption_grouped.columns if x != 'uptake_in_mol/g']]
dataset_Y = df_adsorption_grouped['uptake_in_mol/g']

# split train and test dataset
#------------------------------------------------------------------------------ 
train_X, test_X, train_Y, test_Y = train_test_split(dataset_X, dataset_Y, test_size = GlobVar.test_size, 
                                                    random_state = GlobVar.seed, shuffle = True, stratify = None) 
train_Y = train_Y.to_frame()
test_Y = test_Y.to_frame()

# identify columns
#------------------------------------------------------------------------------ 
continuous_features = ['temperature', 'mol_weight', 'complexity', 'covalent_units', 'H_acceptors', 'H_donors', 'heavy_atoms']
ads_col = ['adsorbent_name'] 
sorb_col = ['adsorbates_name']

# [NORMALIZE VARIABLES]
#==============================================================================
# encodes strings as ordinal indexes and normalize continuous 
# variable within a defined range 
#==============================================================================
print(f'''STEP 3 - Normalizing variables 
      ''') 

# enforce float type for the continuous features columns
#------------------------------------------------------------------------------ 
train_X[continuous_features] = train_X[continuous_features].astype(float)        
test_X[continuous_features] = test_X[continuous_features].astype(float)

# normalize features
#------------------------------------------------------------------------------ 
features_normalizer = MinMaxScaler(feature_range=(0, 1))
features_normalizer.fit(train_X[continuous_features])
train_X[continuous_features] = features_normalizer.transform(train_X[continuous_features])
test_X[continuous_features] = features_normalizer.transform(test_X[continuous_features])

# normalize pressures
#------------------------------------------------------------------------------ 
column = 'pressure_in_Pascal'
pressure_normalizer = MinMaxScaler(feature_range=(0, 1))
pressure_normalizer.fit(np.concatenate(train_X[column].to_list()).reshape(-1, 1))
train_X[column] = train_X[column].apply(lambda x: pressure_normalizer.transform(np.array(x).reshape(-1, 1)).flatten().tolist())
test_X[column] = test_X[column].apply(lambda x: pressure_normalizer.transform(np.array(x).reshape(-1, 1)).flatten().tolist())

# normalize uptakes
#------------------------------------------------------------------------------ 
column = 'uptake_in_mol/g'
uptake_normalizer = MinMaxScaler(feature_range=(0, 1))
uptake_normalizer.fit(np.concatenate(train_Y[column].to_list()).reshape(-1, 1))
train_Y[column] = train_Y[column].apply(lambda x: uptake_normalizer.transform(np.array(x).reshape(-1, 1)).flatten().tolist())
test_Y[column] = test_Y[column].apply(lambda x: uptake_normalizer.transform(np.array(x).reshape(-1, 1)).flatten().tolist())

# [ENCODING CATEGORICAL VARIABLES]
#==============================================================================
# encodes strings as ordinal indexes 
#==============================================================================
unique_adsorbents = train_X['adsorbent_name'].nunique() + 1
unique_sorbates = train_X['adsorbates_name'].nunique() + 1

# define the encoders
#------------------------------------------------------------------------------ 
adsorbent_encoder = OrdinalEncoder(categories = 'auto', handle_unknown = 'use_encoded_value', unknown_value=unique_adsorbents - 1)
sorbates_encoder = OrdinalEncoder(categories = 'auto', handle_unknown = 'use_encoded_value',  unknown_value=unique_sorbates - 1)

# fit encoder on the train sets
#------------------------------------------------------------------------------        
encoded_adsorbents = adsorbent_encoder.fit(train_X[['adsorbent_name']])
encoded_sorbates = sorbates_encoder.fit(train_X[['adsorbates_name']])

# apply encoding on the adsorbent and sorbates columns
#------------------------------------------------------------------------------ 
train_X[['adsorbent_name']] = adsorbent_encoder.transform(train_X[['adsorbent_name']])
train_X[['adsorbates_name']] = sorbates_encoder.transform(train_X[['adsorbates_name']])
test_X[['adsorbent_name']] = adsorbent_encoder.transform(test_X[['adsorbent_name']])
test_X[['adsorbates_name']] = sorbates_encoder.transform(test_X[['adsorbates_name']])

# apply encoding on the adsorbent and sorbates columns
#------------------------------------------------------------------------------ 
train_X['pressure_in_Pascal'] = train_X['pressure_in_Pascal'].apply(lambda x : preprocessor.series_preprocessing(x, pad_length=GlobVar.pad_length, pad_value=GlobVar.pad_value, normalization=False, str_output=True))
test_X['pressure_in_Pascal'] = test_X['pressure_in_Pascal'].apply(lambda x : preprocessor.series_preprocessing(x, pad_length=GlobVar.pad_length, pad_value=GlobVar.pad_value, normalization=False, str_output=True))
train_Y['uptake_in_mol/g'] = train_Y['uptake_in_mol/g'].apply(lambda x : preprocessor.series_preprocessing(x, pad_length=GlobVar.pad_length, pad_value=GlobVar.pad_value, normalization=False, str_output=True))
test_Y['uptake_in_mol/g'] = test_Y['uptake_in_mol/g'].apply(lambda x : preprocessor.series_preprocessing(x, pad_length=GlobVar.pad_length, pad_value=GlobVar.pad_value, normalization=False, str_output=True))

# [PRINT REPORT]
#==============================================================================
# encodes strings as ordinal indexes 
#==============================================================================
print(f'''
-------------------------------------------------------------------------------
Number of experiments before filtering: {df_adsorption.groupby('filename').ngroup().nunique()}
Number of experiments upon filtering:   {df_adsorption_ML.groupby('filename').ngroup().nunique()}
Number of experiments removed:          {df_adsorption.groupby('filename').ngroup().nunique() - df_adsorption_ML.groupby('filename').ngroup().nunique()}
-------------------------------------------------------------------------------
Total number of experiments:             {total_num_exp}
Select number of experiments:            {GlobVar.num_samples}
Total number of experiments (train set): {train_X.shape[0]}
Total number of experiments (test set):  {test_X.shape[0]}
-------------------------------------------------------------------------------
''')

# [MAKE FOLDERS AND SAVE PREPROCESSING UNITS]
#==============================================================================
# Save the trained preprocessing systems (normalizer and encoders) for further use 
#==============================================================================
pp_path = os.path.join(GlobVar.SCADS_path, r'preprocessing')
if not os.path.exists(pp_path):
    os.mkdir(pp_path)

normalizer_path = os.path.join(pp_path, 'features_normalizer.pkl')
with open(normalizer_path, 'wb') as file:
    pickle.dump(features_normalizer, file)
normalizer_path = os.path.join(pp_path, 'pressure_normalizer.pkl')
with open(normalizer_path, 'wb') as file:
    pickle.dump(pressure_normalizer, file)
normalizer_path = os.path.join(pp_path, 'uptake_normalizer.pkl')
with open(normalizer_path, 'wb') as file:
    pickle.dump(uptake_normalizer, file)
encoder_path = os.path.join(pp_path, 'adsorbent_encoder.pkl')
with open(encoder_path, 'wb') as file:
    pickle.dump(adsorbent_encoder, file) 
encoder_path = os.path.join(pp_path, 'sorbates_encoder.pkl')
with open(encoder_path, 'wb') as file:
    pickle.dump(sorbates_encoder, file)     


# [SAVE FILES]
#==============================================================================
# Save the trained preprocessing systems (normalizer and encoders) for further use 
#==============================================================================
file_loc = os.path.join(GlobVar.SCADS_path, 'SCADS_train_X.csv')
train_X.to_csv(file_loc, index = False, sep = ';', encoding = 'utf-8')
file_loc = os.path.join(GlobVar.SCADS_path, 'SCADS_train_Y.csv')
train_Y.to_csv(file_loc, index = False, sep = ';', encoding = 'utf-8')
file_loc = os.path.join(GlobVar.SCADS_path, 'SCADS_test_X.csv')
test_X.to_csv(file_loc, index = False, sep = ';', encoding = 'utf-8')
file_loc = os.path.join(GlobVar.SCADS_path, 'SCADS_test_Y.csv')
test_Y.to_csv(file_loc, index = False, sep = ';', encoding = 'utf-8')

