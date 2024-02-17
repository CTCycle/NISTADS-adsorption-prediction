import os
import sys
import numpy as np
import pandas as pd
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
from modules.components.model_assets import Inference, ModelValidation
import modules.global_variables as GlobVar
import configurations as cnf

# [LOAD MODEL AND DATA]
#==============================================================================
# Load data and models
#==============================================================================

# identify columns
#------------------------------------------------------------------------------
features = ['temperature', 'mol_weight', 'complexity', 'covalent_units', 
            'H_acceptors', 'H_donors', 'heavy_atoms']
ads_col = ['adsorbent_name'] 
sorb_col = ['adsorbates_name']
P_col = 'pressure_in_Pascal'
Q_col = 'uptake_in_mol/g'

# load the model for inference and print summary
#------------------------------------------------------------------------------
inference = Inference() 
model, parameters = inference.load_pretrained_model(GlobVar.models_path)
model_path = inference.folder_path
model.summary(expand_nested=True)

# load preprocessed data
#------------------------------------------------------------------------------
pp_path = os.path.join(model_path, 'preprocessing')
filepath = os.path.join(pp_path, 'train_X.csv')                
train_X = pd.read_csv(filepath, sep= ';', encoding='utf-8')
filepath = os.path.join(pp_path, 'test_X.csv')                
test_X = pd.read_csv(filepath, sep= ';', encoding='utf-8')
filepath = os.path.join(pp_path, 'train_Y.csv')                
train_Y = pd.read_csv(filepath, sep= ';', encoding='utf-8')
filepath = os.path.join(pp_path, 'test_Y.csv')                
test_Y = pd.read_csv(filepath, sep= ';', encoding='utf-8')

# load encoders and normalizers
#------------------------------------------------------------------------------
filepath = os.path.join(pp_path, 'features_normalizer.pkl')
with open(filepath, 'rb') as file:
    feat_normalizer = pickle.load(file)
filepath = os.path.join(pp_path, 'pressure_normalizer.pkl')
with open(filepath, 'rb') as file:
    press_normalizer = pickle.load(file)
filepath = os.path.join(pp_path, 'uptake_normalizer.pkl')
with open(filepath, 'rb') as file:
    uptake_normalizer = pickle.load(file)
filepath = os.path.join(pp_path, 'guest_encoder.pkl')
with open(filepath, 'rb') as file:
    G_encoder = pickle.load(file)
filepath = os.path.join(pp_path, 'host_encoder.pkl')
with open(filepath, 'rb') as file:
    H_encoder = pickle.load(file)

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

# initialize custom classes
#------------------------------------------------------------------------------
preprocessor = PreProcessing()
validator = ModelValidation(model)

# create subfolder for evaluation data
#------------------------------------------------------------------------------
eval_path = os.path.join(model_path, 'evaluation') 
if not os.path.exists(eval_path):
    os.mkdir(eval_path)

# convert pressure and uptake sequences from strings to array of floats
#------------------------------------------------------------------------------
train_pressure = train_X[P_col].apply(lambda x: np.array([float(val) for val in x.split()]))
test_pressure = test_X[P_col].apply(lambda x: np.array([float(val) for val in x.split()]))
train_output = train_Y[Q_col].apply(lambda x: np.array([float(val) for val in x.split()]))
test_output = test_Y[Q_col].apply(lambda x: np.array([float(val) for val in x.split()]))

# Reshape sequences of pressure and uptakes to 2D arrays 
#------------------------------------------------------------------------------
train_pressure = np.stack(train_pressure.values)
test_pressure = np.stack(test_pressure.values)
train_output = np.stack(train_output.values)
test_output = np.stack(test_output.values)

# define train and test inputs
#------------------------------------------------------------------------------
train_inputs = [train_X[features], train_X[ads_col], train_X[sorb_col], train_pressure]
test_inputs = [test_X[features], test_X[ads_col], test_X[sorb_col], test_pressure]

# evaluate model performance on train and test datasets
#------------------------------------------------------------------------------
train_eval = model.evaluate(x=train_inputs, y=train_output, batch_size=512, 
                            verbose=1, workers=6, use_multiprocessing=True)
test_eval = model.evaluate(x=test_inputs, y=test_output, batch_size=512, 
                           verbose=1, workers=6, use_multiprocessing=True)

print(f'''
-------------------------------------------------------------------------------
MODEL EVALUATION
-------------------------------------------------------------------------------    
Train dataset:
- Loss:   {train_eval[0]}
- Metric: {train_eval[1]} 

Test dataset:
- Loss:   {test_eval[0]}
- Metric: {test_eval[1]}        
''')

# predict adsorption from train and test datasets input
#------------------------------------------------------------------------------
train_predictions = model.predict(train_inputs)
test_predictions = model.predict(test_inputs)

# remove padding and normalization from the original train and test pressure series,
# as well from the original train and test uptake series and the predicted values
#------------------------------------------------------------------------------
true_train_pressures = preprocessor.sequence_recovery(train_pressure, parameters['padding_value'], press_normalizer) 
true_test_pressures = preprocessor.sequence_recovery(test_pressure, parameters['padding_value'], press_normalizer)
true_train_uptakes = preprocessor.sequence_recovery(train_output, parameters['padding_value'], uptake_normalizer) 
true_test_uptakes = preprocessor.sequence_recovery(test_output, parameters['padding_value'], uptake_normalizer) 

# the lenght of the original uptake series (labels) is used to slice the predicted series,
# since the pad value cannot be used directly as a reference to remove unwanted values
predicted_train_uptakes = preprocessor.sequence_recovery(train_predictions, parameters['padding_value'], uptake_normalizer,
                                                         from_reference=True, reference=true_train_uptakes) 
predicted_test_uptakes = preprocessor.sequence_recovery(test_predictions, parameters['padding_value'], uptake_normalizer,
                                                        from_reference=True, reference=true_test_uptakes)

# perform visual validation by comparing true and predicted isotherms on both 
# the train and test datasets
#------------------------------------------------------------------------------
validator.visualize_predictions(true_train_pressures, true_train_uptakes, predicted_train_uptakes, 'train', eval_path)
validator.visualize_predictions(true_test_pressures, true_test_uptakes, predicted_test_uptakes, 'test', eval_path)



            


    
    




    
    










    




               

    












