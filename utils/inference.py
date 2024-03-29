import os
import numpy as np
import json
import tensorflow as tf

    
# [TOOLKIT TO USE THE PRETRAINED MODEL]
#==============================================================================
# Custom training operations
#==============================================================================
class Inference:

    def __init__(self, seed):
        self.seed = seed
        np.random.seed(seed)
        tf.random.set_seed(seed)  

    #-------------------------------------------------------------------------- 
    def load_pretrained_model(self, path):

        '''
        Load pretrained keras model (in folders) from the specified directory. 
        If multiple model directories are found, the user is prompted to select one,
        while if only one model directory is found, that model is loaded directly.
        If `load_parameters` is True, the function also loads the model parameters 
        from the target .json file in the same directory. 

        Keyword arguments:
            path (str): The directory path where the pretrained models are stored.
            load_parameters (bool, optional): If True, the function also loads the 
                                              model parameters from a JSON file. 
                                              Default is True.

        Returns:
            model (keras.Model): The loaded Keras model.

        '''        
        model_folders = []
        for entry in os.scandir(path):
            if entry.is_dir():
                model_folders.append(entry.name)
        if len(model_folders) > 1:
            model_folders.sort()
            index_list = [idx + 1 for idx, item in enumerate(model_folders)]     
            print('Please select a pretrained model:') 
            print()
            for i, directory in enumerate(model_folders):
                print(f'{i + 1} - {directory}')        
            print()               
            while True:
                try:
                    dir_index = int(input('Type the model index to select it: '))
                    print()
                except:
                    continue
                break                         
            while dir_index not in index_list:
                try:
                    dir_index = int(input('Input is not valid! Try again: '))
                    print()
                except:
                    continue
            self.folder_path = os.path.join(path, model_folders[dir_index - 1])

        elif len(model_folders) == 1:
            self.folder_path = os.path.join(path, model_folders[0])                 
        
        self.model_path = os.path.join(self.folder_path, 'model') 
        model = tf.keras.models.load_model(self.model_path)
        path = os.path.join(self.folder_path, 'model_parameters.json')
        with open(path, 'r') as f:
            configuration = json.load(f)               
        
        return model, configuration    
   
    #--------------------------------------------------------------------------
    def sequence_recovery(self, pressure, true_Y, pred_Y, pad_value, 
                          pressure_normalizer, uptake_normalizer):
        
        indices_to_remove = [np.where(pressure[i] == pad_value)[0] for i in range(len(pressure))]        
        true_Y_recovered = [np.delete(true_Y[i], indices_to_remove[i]) for i in range(len(pressure))]
        pred_Y_recovered = [np.delete(pred_Y[i], indices_to_remove[i]) for i in range(len(pressure))]
        pressure_recovered = [np.delete(pressure[i], indices_to_remove[i]) for i in range(len(pressure))]
        true_Y_recovered = [uptake_normalizer.inverse_transform(x.reshape(-1, 1)) for x in true_Y_recovered]
        pred_Y_recovered = [uptake_normalizer.inverse_transform(x.reshape(-1, 1)) for x in pred_Y_recovered]
        pressure_recovered = [pressure_normalizer.inverse_transform(x.reshape(-1, 1)) for x in pressure_recovered]

        return pressure_recovered, true_Y_recovered, pred_Y_recovered
        
    
   

        
          
    
            
