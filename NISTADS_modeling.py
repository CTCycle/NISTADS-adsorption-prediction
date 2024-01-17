import sys

# set warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# import modules and classes
#------------------------------------------------------------------------------
from modules.components.data_classes import UserOperations

# [MAIN MENU]
#==============================================================================
# module for the selection of different operations
#==============================================================================
user_operations = UserOperations()
operations_menu = {'1' : 'SCADS framework: training and predictions',                   
                   '2' : 'BMADS framework: training and predictions',                                    
                   '3' : 'Exit and close'}

SCADS_menu = {'1' : 'Preprocess data for model training',
              '2' : 'Pretrain model',
              '3' : 'Validation of pretrained models',
              '4' : 'Predict adsorption of compounds',              
              '5' : 'Go back to main menu'}

while True:
    print('------------------------------------------------------------------------')
    print('NISTADS Project')
    print('------------------------------------------------------------------------')
    print()
    op_sel = user_operations.menu_selection(operations_menu)
    print()      
    if op_sel == 1:
        while True:
            sec_sel = user_operations.menu_selection(SCADS_menu)
            print()
            if sec_sel == 1:
                import modules.SCADS_preprocessing
                del sys.modules['modules.SCADS_preprocessing']
            elif sec_sel == 2:
                import modules.SCADS_training
                del sys.modules['modules.SCADS_training']
            elif sec_sel == 3:
                import modules.SCADS_validation
                del sys.modules['modules.SCADS_validation']
            elif sec_sel == 4:
                import modules.SCADS_predictions
                del sys.modules['modules.SCADS_predictions']
            elif sec_sel == 5:
                break
        
    elif op_sel == 2:
        pass 

    elif op_sel == 3:
        break      
    
 
