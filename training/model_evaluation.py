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
from components.data_assets import PreProcessing
from components.model_assets import Inference, ModelValidation
import components.global_paths as globpt
import configurations as cnf

# specify relative paths from global paths and create subfolders
#------------------------------------------------------------------------------
cp_path = os.path.join(globpt.train_path, 'checkpoints')
os.mkdir(cp_path) if not os.path.exists(cp_path) else None

# [LOAD MODEL AND DATA]
#==============================================================================
#==============================================================================

# load the model for inference and print summary
#------------------------------------------------------------------------------
inference = Inference(cnf.seed) 
model, parameters = inference.load_pretrained_model(cp_path)
model_path = inference.folder_path
model.summary(expand_nested=True)

# load preprocessed data
#------------------------------------------------------------------------------
pp_path = os.path.join(model_path, 'preprocessing')

# load train data
train_parameters = np.load(os.path.join(pp_path, 'train_parameters.npy'))
train_hosts = np.load(os.path.join(pp_path, 'train_hosts.npy'))
train_guests = np.load(os.path.join(pp_path, 'train_guests.npy'))
train_pressures = np.load(os.path.join(pp_path, 'train_pressures.npy'))
train_uptakes = np.load(os.path.join(pp_path, 'train_uptakes.npy'))

# load test data
test_parameters = np.load(os.path.join(pp_path, 'test_parameters.npy'))
test_hosts = np.load(os.path.join(pp_path, 'test_hosts.npy'))
test_guests = np.load(os.path.join(pp_path, 'test_guests.npy'))
test_pressures = np.load(os.path.join(pp_path, 'test_pressures.npy'))
test_uptakes = np.load(os.path.join(pp_path, 'test_uptakes.npy'))

# create list of inputs for both train and test datasets
train_inputs = [train_parameters, train_hosts, train_guests, train_pressures] 
test_inputs = [test_parameters, test_hosts, test_guests, test_pressures] 
validation_data = (test_inputs, test_uptakes) 

# load encoders and normalizers
#------------------------------------------------------------------------------
filepath = os.path.join(pp_path, 'pressure_normalizer.pkl')
with open(filepath, 'rb') as file:
    press_normalizer = pickle.load(file)
filepath = os.path.join(pp_path, 'uptake_normalizer.pkl')
with open(filepath, 'rb') as file:
    uptake_normalizer = pickle.load(file)


# [PERFORM QUICK EVALUATION]
#==============================================================================
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
os.mkdir(eval_path) if not os.path.exists(eval_path) else None

# evaluate model performance on train and test datasets
#------------------------------------------------------------------------------
train_eval = model.evaluate(x=train_inputs, y=train_uptakes, batch_size=512, verbose=1)
test_eval = model.evaluate(x=test_inputs, y=test_uptakes, batch_size=512, verbose=1)

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
rec_train_P, rec_train_Q, pred_train_Q = inference.sequence_recovery(train_pressures,                                                                     
                                                                     train_uptakes,
                                                                     train_predictions,
                                                                     parameters['padding_value'],
                                                                     press_normalizer,
                                                                     uptake_normalizer)  

rec_test_P, rec_test_Q, pred_test_Q = inference.sequence_recovery(test_pressures,
                                                                  test_uptakes,
                                                                  test_predictions,                                                                  
                                                                  parameters['padding_value'],
                                                                  press_normalizer,
                                                                  uptake_normalizer)  

# perform visual validation by comparing true and predicted isotherms on both 
# the train and test datasets
#------------------------------------------------------------------------------
validator.visualize_predictions(rec_train_P, rec_train_Q, pred_train_Q, 'train', eval_path)
validator.visualize_predictions(rec_test_P, rec_test_Q, pred_test_Q, 'test', eval_path)



            


    
    




    
    










    




               

    












