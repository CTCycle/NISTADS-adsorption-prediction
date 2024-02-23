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
from components.model_assets import Inference, ModelValidation
import components.global_paths as globpt
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




    
    




    
    










    




               

    












