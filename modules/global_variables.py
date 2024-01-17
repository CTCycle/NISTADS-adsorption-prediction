import os

# Define paths for the script
#------------------------------------------------------------------------------
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
pred_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'predictions')
SCADS_path = os.path.join(data_path, 'SCADS preprocessing')
BMADS_path = os.path.join(data_path, 'BMADS preprocessing')


# Create folders if they do not exist
#------------------------------------------------------------------------------
if not os.path.exists(data_path):
    os.mkdir(data_path)
if not os.path.exists(model_path):
    os.mkdir(model_path) 
if not os.path.exists(pred_path):
    os.mkdir(pred_path) 
if not os.path.exists(SCADS_path):
    os.mkdir(SCADS_path) 
if not os.path.exists(BMADS_path):
    os.mkdir(BMADS_path) 

