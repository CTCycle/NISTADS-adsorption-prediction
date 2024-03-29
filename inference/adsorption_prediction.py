import os
import sys
import pandas as pd
import numpy as np
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
from utils.preprocessing import PreProcessing
from utils.inference import Inference
from utils.validation import ModelValidation
import utils.global_paths as globpt
import configurations as cnf

# specify relative paths from global paths and create subfolders
#------------------------------------------------------------------------------
cp_path = os.path.join(globpt.train_path, 'checkpoints')
os.mkdir(cp_path) if not os.path.exists(cp_path) else None

# [LOAD MODEL AND DATA]
#==============================================================================
#==============================================================================

# define column names
#------------------------------------------------------------------------------
valid_units = ['mmol/g', 'mol/kg', 'mol/g', 'mmol/kg', 'mg/g', 'g/g', 
               'wt%', 'g Adsorbate / 100g Adsorbent', 'g/100g', 'ml(STP)/g', 
               'cm3(STP)/g']
features = ['temperature', 'mol_weight', 'complexity', 'covalent_units', 
            'H_acceptors', 'H_donors', 'heavy_atoms']
ads_col, sorb_col  = ['adsorbent_name'], ['adsorbates_name'] 
P_col, Q_col  = 'pressure_in_Pascal', 'uptake_in_mol/g'
P_unit_col, Q_unit_col  = 'pressureUnits', 'adsorptionUnits'

# load the model for inference and print summary
#------------------------------------------------------------------------------
inference = Inference(cnf.seed) 
model, parameters = inference.load_pretrained_model(cp_path)
model_path = inference.folder_path
model.summary(expand_nested=True)

file_loc = os.path.join(globpt.inference_path, 'adsorption_inputs.csv') 
df_predictions = pd.read_csv(file_loc, sep=';', encoding='utf-8')
file_loc = os.path.join(globpt.inference_path, 'adsorbates_dataset.csv') 
df_adsorbates = pd.read_csv(file_loc, sep=';', encoding='utf-8')


