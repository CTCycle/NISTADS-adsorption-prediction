import os
import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
from keras.api._v2.keras import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, OrdinalEncoder
from tqdm import tqdm
tqdm.pandas()


# [USER OPERATIONS]
#==============================================================================
# ...
#==============================================================================
class UserOperations:    
        
    # print custom menu on console and allows selecting an option
    #--------------------------------------------------------------------------
    def menu_selection(self, menu):        
        
        '''        
        menu_selection(menu)
        
        Presents a custom menu to the user and returns the selected option.
        
        Keyword arguments:                      
            menu (dict): A dictionary containing the options to be presented to the user. 
                         The keys are integers representing the option numbers, and the 
                         values are strings representing the option descriptions.
        
        Returns:            
            op_sel (int): The selected option number.
        
        '''        
        indexes = [idx + 1 for idx, val in enumerate(menu)]
        for key, value in menu.items():
            print('{0} - {1}'.format(key, value))            
        
        print()
        while True:
            try:
                op_sel = int(input('Select the desired operation: '))
            except:
                continue            
            
            while op_sel not in indexes:
                try:
                    op_sel = int(input('Input is not valid, please select a valid option: '))
                except:
                    continue
            break
        
        return op_sel   
         

# [DATA PREPROCESSING]
#==============================================================================
# preprocess adsorption data
#==============================================================================
class PreProcessing:   


    def __init__(self):
        self.valid_units = ['mmol/g', 'mol/kg', 'mol/g', 'mmol/kg', 'mg/g', 'g/g', 
                            'wt%', 'g Adsorbate / 100g Adsorbent', 'g/100g', 'ml(STP)/g', 
                            'cm3(STP)/g']
        self.parameters = ['temperature', 'mol_weight', 'complexity', 'covalent_units', 
                         'H_acceptors', 'H_donors', 'heavy_atoms']
        self.ads_col, self.sorb_col  = ['adsorbent_name'], ['adsorbates_name'] 
        self.P_col, self.Q_col  = 'pressure_in_Pascal', 'uptake_in_mol/g'
        self.P_unit_col, self.Q_unit_col  = 'pressureUnits', 'adsorptionUnits'  

    

    #--------------------------------------------------------------------------
    def pressure_converter(self, type, p_val):

        '''
        Converts pressure from the specified unit to Pascals.

        Keyword arguments:
            type (str): The original unit of pressure.
            p_val (int or float): The original pressure value.

        Returns:
            p_val (int): The pressure value converted to Pascals.

        '''         
        if type == 'bar':
            p_val = int(p_val * 100000)        
                
        return p_val

    #--------------------------------------------------------------------------
    def uptake_converter(self, q_unit, q_val, mol_weight):

        '''
        Converts the uptake value from the specified unit to moles per gram.

        Keyword arguments:
            q_unit (str):              The original unit of uptake.
            q_val (int or float):      The original uptake value.
            mol_weight (int or float): The molecular weight of the adsorbate.

        Returns:
            q_val (float): The uptake value converted to moles per gram

        '''        
        if q_unit in ('mmol/g', 'mol/kg'):
            q_val = q_val/1000         
        elif q_unit == 'mmol/kg':
            q_val = q_val/1000000
        elif q_unit == 'mg/g':
            q_val = q_val/1000/float(mol_weight)            
        elif q_unit == 'g/g':
            q_val = (q_val/float(mol_weight))                                   
        elif q_unit == 'wt%':                
            q_val = ((q_val/100)/float(mol_weight))          
        elif q_unit in ('g Adsorbate / 100g Adsorbent', 'g/100g'):              
            q_val = ((q_val/100)/float(mol_weight))                            
        elif q_unit in ('ml(STP)/g', 'cm3(STP)/g'):
            q_val = q_val/22.414      
                
        return q_val        
        
    #--------------------------------------------------------------------------
    def add_guest_properties(self, df_isotherms, df_adsorbates):

        '''
        Assigns properties to adsorbates based on their isotherm data.

        This function takes two pandas DataFrames: one containing isotherm data (df_isotherms)
        and another containing adsorbate properties (df_adsorbates). It merges the two DataFrames
        on the 'adsorbates_name' column, assigns properties to each adsorbate, and returns a new
        DataFrame containing the merged data with assigned properties.

        Keyword Arguments:
            df_isotherms (pandas DataFrame): A DataFrame containing isotherm data.
            df_adsorbates (pandas DataFrame): A DataFrame containing adsorbate properties.

        Returns:
            df_adsorption (pandas DataFrame): A DataFrame containing merged isotherm data
                                              with assigned adsorbate properties.

        '''
        df_isotherms['adsorbates_name'] = df_isotherms['adsorbates_name'].str.lower()
        df_adsorbates['adsorbates_name'] = df_adsorbates['name'].str.lower()        
        df_properties = df_adsorbates[['adsorbates_name', 'complexity', 'atoms', 'mol_weight', 
                                        'covalent_units', 'H_acceptors', 'H_donors', 'heavy_atoms']]        
        df_adsorption = pd.merge(df_isotherms, df_properties, on='adsorbates_name', how='inner')

        return df_adsorption
    
    #--------------------------------------------------------------------------
    def remove_leading_zeros(self, sequence_A, sequence_B):

        # Find the index of the first non-zero element or get the last index if all are zeros
        first_non_zero_index_A = next((i for i, x in enumerate(sequence_A) if x != 0), len(sequence_A) - 1)
        first_non_zero_index_B = next((i for i, x in enumerate(sequence_B) if x != 0), len(sequence_B) - 1)
            
        # Ensure to remove leading zeros except one, for both sequences
        processed_seq_A = sequence_A[max(0, first_non_zero_index_A - 1):]
        processed_seq_B = sequence_B[max(0, first_non_zero_index_B - 1):]        
        
        return processed_seq_A, processed_seq_B
    

    #--------------------------------------------------------------------------
    def split_dataset(self, dataset, test_size, seed=42):
        inputs = dataset[[x for x in dataset.columns if x != self.Q_col]]
        labels = dataset[self.Q_col]
        train_X, test_X, train_Y, test_Y = train_test_split(inputs, labels, test_size=test_size, 
                                                            random_state=seed, shuffle=True, 
                                                            stratify=None) 
        
        return train_X, test_X, train_Y, test_Y 
        

    # normalize sequences using a RobustScaler: X = X - median(X)/IQR(X)
    # flatten and reshape array to make it compatible with the scaler
    #--------------------------------------------------------------------------  
    def normalize_sequences(self, train, test, column):        
        
        normalizer = MinMaxScaler(feature_range=(0,1))
        sequence_array = np.array([item for sublist in train[column] for item in sublist]).reshape(-1, 1)         
        normalizer.fit(sequence_array)
        train[column] = train[column].apply(lambda x: normalizer.transform(np.array(x).reshape(-1, 1)).flatten())
        test[column] = test[column].apply(lambda x: normalizer.transform(np.array(x).reshape(-1, 1)).flatten())

        return train, test, normalizer
    
    # normalize parameters
    #--------------------------------------------------------------------------  
    def normalize_parameters(self, train_X, train_Y, test_X, test_Y):

        '''
        Normalize the input features and output labels for training and testing data.
        This method normalizes the input features and output labels to facilitate 
        better model training and evaluation.

        Keyword Arguments:
            train_X (DataFrame): DataFrame containing the features of the training data.
            train_Y (list): List containing the labels of the training data.
            test_X (DataFrame): DataFrame containing the features of the testing data.
            test_Y (list): List containing the labels of the testing data.

        Returns:
            Tuple: A tuple containing the normalized training features, normalized training labels,
                   normalized testing features, and normalized testing labels.
        
        '''
        
        # cast float type for both the labels and the continuous features columns 
        norm_columns = ['temperature', 'mol_weight', 'complexity', 'heavy_atoms']       
        train_X[norm_columns] = train_X[norm_columns].astype(float)        
        test_X[norm_columns] = test_X[norm_columns].astype(float)
        
        # normalize the numerical features (temperature and physicochemical properties)      
        self.param_normalizer = MinMaxScaler(feature_range=(0, 1))
        train_X[norm_columns] = self.param_normalizer.fit_transform(train_X[norm_columns])
        test_X[norm_columns] = self.param_normalizer.transform(test_X[norm_columns])        

        return train_X, train_Y, test_X, test_Y    
    
    
    # encode variables  
    #--------------------------------------------------------------------------  
    def GH_encoding(self, unique_adsorbents, unique_sorbates, train_X, test_X):

        '''
        Encode categorical features using ordinal encoding. This method encodes categorical 
        features in the training and testing data using ordinal encoding.

        Keyword Arguments:
            unique_adsorbents (int): Number of unique adsorbents.
            unique_sorbates (int): Number of unique sorbates.
            train_X (DataFrame): DataFrame containing the features of the training data.
            test_X (DataFrame): DataFrame containing the features of the testing data.

        Returns:
            Tuple: A tuple containing the encoded training features and encoded testing features.
        
        '''      
        self.host_encoder = OrdinalEncoder(categories='auto', handle_unknown='use_encoded_value', unknown_value=unique_adsorbents - 1)
        self.guest_encoder = OrdinalEncoder(categories='auto', handle_unknown='use_encoded_value',  unknown_value=unique_sorbates - 1)
        
        train_X[['adsorbent_name']] = self.host_encoder.fit_transform(train_X[['adsorbent_name']])
        train_X[['adsorbates_name']] = self.guest_encoder.fit_transform(train_X[['adsorbates_name']])
        test_X[['adsorbent_name']] = self.host_encoder.transform(test_X[['adsorbent_name']])
        test_X[['adsorbates_name']] = self.guest_encoder.transform(test_X[['adsorbates_name']])

        return train_X, test_X    
    
    #--------------------------------------------------------------------------  
    def sequence_padding(self, dataset, column, pad_value=-1, pad_length=50):

        '''
        Normalizes a series of values.
    
        Keyword arguments:
            series (list): A list of values to be normalized
    
        Returns:
            list: A list of normalized values
        
        '''        
        dataset[column] = preprocessing.sequence.pad_sequences(dataset[column], 
                                                               maxlen=pad_length, 
                                                               value=pad_value, 
                                                               dtype='float32', 
                                                               padding='post').tolist()           

        return dataset     
        
    #--------------------------------------------------------------------------
    def model_savefolder(self, path, model_name):

        '''
        Creates a folder with the current date and time to save the model.
    
        Keyword arguments:
            path (str):       A string containing the path where the folder will be created.
            model_name (str): A string containing the name of the model.
    
        Returns:
            str: A string containing the path of the folder where the model will be saved.
        
        '''        
        today_datetime = str(datetime.now())
        truncated_datetime = today_datetime[:-10]
        today_datetime = truncated_datetime.replace(':', '').replace('-', '').replace(' ', 'H') 
        self.folder_name = f'{model_name}_{today_datetime}'
        model_folder_path = os.path.join(path, self.folder_name)
        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path) 
                    
        return model_folder_path
          

