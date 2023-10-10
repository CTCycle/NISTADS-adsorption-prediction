import os
import sys
import pandas as pd
import numpy as np
import pickle 
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# [IMPORT MODULES AND CLASSES]
#==============================================================================
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.components.data_classes import PreProcessing
from modules.components.training_classes import ModelTraining, ModelValidation
import modules.global_variables as GlobVar

