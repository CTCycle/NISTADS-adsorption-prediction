import os
import sys
import numpy as np
import pandas as pd
import pickle

# set warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# add parent folder path to the namespace
#------------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 

# import modules and classes
#------------------------------------------------------------------------------
from components.data_assets import PreProcessing, DataValidation
import components.global_paths as globpt
import configurations as cnf

# specify relative paths from global paths and create subfolders
#------------------------------------------------------------------------------
eval_path = os.path.join(globpt.data_path, 'validation')
hist_path = os.path.join(eval_path, 'histograms')
os.mkdir(eval_path) if not os.path.exists(eval_path) else None
os.mkdir(hist_path) if not os.path.exists(hist_path) else None

# [PREPROCESS DATASET: ADD PHYSICOCHEMICAL PROPERTIES]
#==============================================================================
#==============================================================================
print(f'''
-------------------------------------------------------------------------------
NISTADS data validation
-------------------------------------------------------------------------------
...
''')

preprocessor = PreProcessing()

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
print('Adding physicochemical properties from guest species dataset\n')
dataset = preprocessor.add_guest_properties(df_adsorption, df_adsorbates)
dataset = dataset.dropna()

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
                  'uptake_in_mol_g' : list}
   
# group dataset by experiments and drop filename column as it is not necessary
dataset_grouped = dataset.groupby('filename', as_index=False).agg(aggregate_dict)
dataset_grouped.drop(columns='filename', axis=1, inplace=True)

# remove series of pressure/uptake with less than X points, drop rows containing nan
# values and select a subset of samples for training
#------------------------------------------------------------------------------ 
dataset_grouped = dataset_grouped[~dataset_grouped[preprocessor.P_col].apply(lambda x: all(elem == 0 for elem in x))]
dataset_grouped = dataset_grouped[dataset_grouped[preprocessor.P_col].apply(lambda x: len(x)) >= cnf.min_points]
dataset_grouped = dataset_grouped.dropna()

# check to avoid errors when selecting number of samples higher than effectively 
# available samples. If less are available, the entire dataset is selected
if cnf.num_samples < dataset_grouped.shape[0]:
    dataset_grouped = dataset_grouped.sample(n=cnf.num_samples, random_state=30).reset_index()

# preprocess sequences to remove leading 0 values (some experiments may have several
# zero measurements at the start), make sure that every experiment starts with pressure
# of 0 Pa and uptake of 0 mol/g (effectively converges to zero)
#------------------------------------------------------------------------------
dataset_grouped[[preprocessor.P_col, preprocessor.Q_col]] = dataset_grouped.apply(lambda row: 
                 preprocessor.remove_leading_zeros(row[preprocessor.P_col],
                 row[preprocessor.Q_col]), axis=1, result_type='expand')

# [VALIDATE DATA]
#==============================================================================
#==============================================================================

validator = DataValidation()

# print report with statistics and info about the non-grouped dataset
#------------------------------------------------------------------------------ 
print(f'''
Number of adsorption measurements:   {dataset.shape[0]}
Number of unique experiments:        {dataset_grouped.shape[0]}
Number of dataset features:          {dataset_grouped.shape[1]}
Average measurements per experiment: {dataset.shape[0]//dataset_grouped.shape[0]}
''')

# perform prelimiary analysis on the grouped, unsplit dataset
#------------------------------------------------------------------------------ 

# check columns with null values
print('Checking for missing values in the dataset:\n')
missing_values = validator.check_missing_values(dataset_grouped)  

# generate histograms of the grouped dataset features (only those that are continuous)
print('\nGenerating histograms for the grouped dataset\n')
validator.plot_histograms(dataset_grouped, eval_path)

# validate splitting based on random seed
#------------------------------------------------------------------------------ 
print('\nValidation best random seed for data splitting\n')
min_diff, best_seed, best_split = validator.data_split_validation(dataset, cnf.test_size, 500)
print(f'''\nBest split found with split_seed of {best_seed}, with total difference equal to {round(min_diff, 3)}
Mean and standard deviation differences per features (X and Y):''')
for key, val in best_split.items():
    print(f'{key} ---> mean difference = {val[0]}, STD difference = {val[1]}')

