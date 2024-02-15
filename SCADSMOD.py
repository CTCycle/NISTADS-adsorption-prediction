import sys
import art

# set warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# import modules and classes
#------------------------------------------------------------------------------
from modules.components.data_assets import UserOperations

# welcome message
#------------------------------------------------------------------------------
ascii_art = art.text2art('SCADS modeling')
print(ascii_art)


# [MAIN MENU]
#==============================================================================
# module for the selection of different operations
#==============================================================================
user_operations = UserOperations()
operations_menu = {'1' : 'SCADS model training',                  
                   '3' : 'Model evaluation',  
                   '4' : 'Predict adsorption with pretrained model',                                   
                   '5' : 'Exit and close'}

while True:
    print('------------------------------------------------------------------------')
    print('MAIN MENU')
    print('------------------------------------------------------------------------')
    op_sel = user_operations.menu_selection(operations_menu)
    print()      
    if op_sel == 1:
        import modules.model_training
        del sys.modules['modules.model_training']               
    elif op_sel == 2:
        import modules.model_evaluation
        del sys.modules['modules.model_evaluation']    
    elif op_sel == 3:
        import modules.adsorption_prediction
        del sys.modules['modules.adsorption_prediction']
    elif op_sel == 4:
                break


 
