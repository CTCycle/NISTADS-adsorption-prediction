import os

# Define paths for the script
#------------------------------------------------------------------------------
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
pred_path = os.path.join(data_path, 'predictions')

# Create folders if they do not exist
#------------------------------------------------------------------------------
if not os.path.exists(data_path):
    os.mkdir(data_path)
if not os.path.exists(models_path):
    os.mkdir(models_path) 
if not os.path.exists(pred_path):
    os.mkdir(pred_path) 

# placeholder for model save folder
#------------------------------------------------------------------------------
model_folder_path = ''
model_folder_name = ''
