import os
import sys
import pandas as pd
import pickle 
from sklearn.model_selection import train_test_split
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
from modules.components.data_assets import PreProcessing
import modules.global_variables as GlobVar
import configurations as cnf

# [PREPARE RAW DATASET]
#==============================================================================
# module for the selection of different operations
#==============================================================================
print('''
-------------------------------------------------------------------------------
SCADS data preprocessing
-------------------------------------------------------------------------------
...
''')

# create model folder and save placeholders in global variables
#------------------------------------------------------------------------------
preprocessor = PreProcessing()
model_folder_path = preprocessor.model_savefolder(GlobVar.models_path, 'SCADS')
model_folder_name = preprocessor.folder_name
GlobVar.model_folder_path = model_folder_path
GlobVar.model_folder_name = model_folder_name

# create subfolder for preprocessing data
#------------------------------------------------------------------------------
pp_path = os.path.join(model_folder_path, 'preprocessing')
if not os.path.exists(pp_path):
    os.mkdir(pp_path)

# load data from csv
#------------------------------------------------------------------------------
file_loc = os.path.join(GlobVar.data_path, 'SCADS_dataset.csv') 
df_adsorption = pd.read_csv(file_loc, sep = ';', encoding = 'utf-8')
file_loc = os.path.join(GlobVar.data_path, 'adsorbates_dataset.csv') 
df_adsorbates = pd.read_csv(file_loc, sep = ';', encoding = 'utf-8')
file_loc = os.path.join(GlobVar.data_path, 'adsorbents_dataset.csv') 
df_adsorbents = pd.read_csv(file_loc, sep = ';', encoding = 'utf-8')

# add molecular properties based on PUG CHEM API data
#------------------------------------------------------------------------------ 
print(f'''STEP 1 - Prepare raw dataset for preprocessing
      ''')
preprocessor = PreProcessing()
dataset = preprocessor.guest_properties(df_adsorption, df_adsorbates)

# filter experiments by allowed uptake units
#------------------------------------------------------------------------------ 
valid_units = ['mmol/g', 'mol/kg', 'mol/g', 'mmol/kg', 'mg/g', 'g/g', 
               'wt%', 'g Adsorbate / 100g Adsorbent', 'g/100g', 'ml(STP)/g', 
               'cm3(STP)/g']
dataset = dataset[dataset['adsorptionUnits'].isin(valid_units)]

# convert pressure and uptake to Pa (pressure) and mol/kg (uptake) 
#------------------------------------------------------------------------------ 
dataset['pressure_in_Pascal'] = dataset.progress_apply(lambda x : preprocessor.pressure_converter(x['pressureUnits'], x['pressure']), axis = 1)
dataset['uptake_in_mol/g'] = dataset.progress_apply(lambda x : preprocessor.uptake_converter(x['adsorptionUnits'], x['adsorbed_amount'], x['mol_weight']), axis = 1)
print()

# filter the dataset to remove experiments with units are outside desired boundaries, 
# such as experiments with negative values of temperature, pressure and uptake
#------------------------------------------------------------------------------ 
dataset = dataset[dataset['temperature'].astype(int) > 0]
dataset = dataset[dataset['pressure_in_Pascal'].astype(float).between(0.0, cnf.max_pressure)]
dataset = dataset[dataset['uptake_in_mol/g'].astype(float).between(0.0, cnf.max_uptake)]

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
dataset_grouped = dataset.groupby('filename', as_index = False).agg(aggregate_dict)
dataset_grouped.drop(columns='filename', axis=1, inplace=True)
total_num_exp = dataset_grouped.shape[0]

# remove series of pressure/uptake with less than X points, drop rows containing nan
# values and select a subset of samples for training
#------------------------------------------------------------------------------ 
dataset_grouped = dataset_grouped[dataset_grouped['pressure_in_Pascal'].apply(lambda x: len(x)) >= cnf.min_points]
dataset_grouped = dataset_grouped.dropna()
dataset_grouped = dataset_grouped.sample(n=cnf.num_samples, random_state=30).reset_index()

# [DATA PREPROCESSING]
#==============================================================================
# Preprocess data using different methods
#==============================================================================
print(f'''STEP 2 - Preprocess data for SCADS training
      ''')

# isolate data inputs and outputs 
#------------------------------------------------------------------------------ 
inputs = dataset_grouped[[x for x in dataset_grouped.columns if x != 'uptake_in_mol/g']]
labels = dataset_grouped['uptake_in_mol/g']

# split train and test dataset
#------------------------------------------------------------------------------ 
train_X, test_X, train_Y, test_Y = train_test_split(inputs, labels, test_size=cnf.test_size, 
                                                    random_state=cnf.seed, shuffle=True, stratify=None) 

# normalize all continuous variables (temperature, physicochemical properties,
# pressure, uptake)
#------------------------------------------------------------------------------
print(f'''STEP 3 - Normalizing continuous variables 
      ''') 
train_X, train_Y, test_X, test_Y = preprocessor.normalize_data(train_X, train_Y, test_X, test_Y)
features_normalizer = preprocessor.features_normalizer
pressure_normalizer = preprocessor.pressure_normalizer
uptake_normalizer = preprocessor.uptake_normalizer

# normalize all continuous variables (temperature, physicochemical properties,
# pressure, uptake)
#------------------------------------------------------------------------------
print(f'''STEP 4 - Encoding categorical variables 
      ''') 

unique_adsorbents = train_X['adsorbent_name'].nunique() + 1
unique_sorbates = train_X['adsorbates_name'].nunique() + 1

# encode categorical variables
#------------------------------------------------------------------------------ 
train_X, test_X = preprocessor.data_encoding(unique_adsorbents, unique_sorbates, train_X, test_X)
host_encoder = preprocessor.host_encoder
guest_encoder = preprocessor.guest_encoder

# apply encoding on the adsorbent and sorbates columns
#------------------------------------------------------------------------------ 
train_X['pressure_in_Pascal'] = train_X['pressure_in_Pascal'].apply(lambda x : preprocessor.sequence_padding(x, pad_length=cnf.pad_length, pad_value=cnf.pad_value))
test_X['pressure_in_Pascal'] = test_X['pressure_in_Pascal'].apply(lambda x : preprocessor.sequence_padding(x, pad_length=cnf.pad_length, pad_value=cnf.pad_value))
train_Y = [preprocessor.sequence_padding(x, pad_length=cnf.pad_length, pad_value=cnf.pad_value) for x in train_Y]
test_Y = [preprocessor.sequence_padding(x, pad_length=cnf.pad_length, pad_value=cnf.pad_value) for x in test_Y]

# [PRINT RESULTS AND SAVE DATA]
#==============================================================================
# Preprocess data using different methods
#==============================================================================
print(f'''
-------------------------------------------------------------------------------   
PREPROCESSING REPORT
-------------------------------------------------------------------------------
Number of experiments before filtering: {df_adsorption.groupby('filename').ngroup().nunique()}
Number of experiments upon filtering:   {dataset.groupby('filename').ngroup().nunique()}
Number of experiments removed:          {df_adsorption.groupby('filename').ngroup().nunique() - dataset.groupby('filename').ngroup().nunique()}
-------------------------------------------------------------------------------
Total number of experiments:             {total_num_exp}
Select number of experiments:            {cnf.num_samples}
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

# transform label datasets into pandas dataframe to save as .csv
#------------------------------------------------------------------------------
train_Y = pd.DataFrame(train_Y, columns=['uptake_in_mol/g'])
test_Y = pd.DataFrame(test_Y, columns=['uptake_in_mol/g'])

# save preprocessed data
#------------------------------------------------------------------------------
file_loc = os.path.join(pp_path, 'train_X.csv')
train_X.to_csv(file_loc, index=False, sep=';', encoding='utf-8')
file_loc = os.path.join(pp_path, 'train_Y.csv')
train_Y.to_csv(file_loc, index=False, sep=';', encoding='utf-8')
file_loc = os.path.join(pp_path, 'test_X.csv')
test_X.to_csv(file_loc, index=False, sep=';', encoding='utf-8')
file_loc = os.path.join(pp_path, 'test_Y.csv')
test_Y.to_csv(file_loc, index=False, sep=';', encoding='utf-8')

