import os
import pandas as pd
import numpy as np
from datetime import datetime
from keras.api._v2.keras import preprocessing
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
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
    

    #--------------------------------------------------------------------------
    def zero_convergence(self, row, col_A, col_B):
        if row[col_A][0] == 0 and row[col_B][0] != 0:
            row[col_B][0] = 0         
        elif row[col_A][0] != 0 and row[col_B][0] != 0:
            row[col_A].insert(0, 0)
            row[col_B].insert(0, 0)

        return row

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
    def guest_properties(self, df_isotherms, df_adsorbates):

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
    
    # normalize data 
    #--------------------------------------------------------------------------  
    def normalize_data(self, train_X, train_Y, test_X, test_Y):

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
        columns = ['temperature', 'mol_weight', 'complexity', 'heavy_atoms']

        # cast float type for both the labels and the continuous features columns
        train_Y = [[float(value) for value in row] for row in train_Y]
        test_Y = [[float(value) for value in row] for row in test_Y]
        train_X[columns] = train_X[columns].astype(float)        
        test_X[columns] = test_X[columns].astype(float)
        
        # normalize the numerical features (temperature, physicochemical properties)      
        self.features_normalizer = MinMaxScaler(feature_range=(0, 1))
        train_X[columns] = self.features_normalizer.fit_transform(train_X[columns])
        test_X[columns] = self.features_normalizer.transform(test_X[columns])

        # normalize pressures of adsorption within the range 0 - 1
        # flatten and reshape array of arrays to make it compatible with the MinMaxScaler
        # use apply to transform each array
        column = 'pressure_in_Pascal'
        pressure_array = [item for sublist in train_X[column] for item in sublist]
        pressure_array = np.array(pressure_array).reshape(-1, 1)

        self.pressure_normalizer = MinMaxScaler(feature_range=(0, 1))
        self.pressure_normalizer.fit(pressure_array)
        train_X[column] = train_X[column].apply(lambda x: self.pressure_normalizer.transform(np.array(x).reshape(-1, 1)).flatten())
        test_X[column] = test_X[column].apply(lambda x: self.pressure_normalizer.transform(np.array(x).reshape(-1, 1)).flatten())

        # normalize uptake within the range 0 - 1
        # flatten and reshape array of arrays to make it compatible with the MinMaxScaler
        # use apply to transform each array
        column = 'uptake_in_mol/g'
        uptake_array = [item for sublist in train_Y for item in sublist]
        uptake_array = np.array(uptake_array).reshape(-1, 1)

        self.uptake_normalizer = MinMaxScaler(feature_range=(0, 1))
        self.uptake_normalizer.fit(uptake_array)
        train_Y = [self.uptake_normalizer.transform(np.array(x).reshape(-1, 1)).flatten() for x in train_Y]
        test_Y = [self.uptake_normalizer.transform(np.array(x).reshape(-1, 1)).flatten() for x in test_Y]

        return train_X, train_Y, test_X, test_Y    
    
    # encode variables  
    #--------------------------------------------------------------------------  
    def data_encoding(self, unique_adsorbents, unique_sorbates, train_X, test_X):

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
    
    # preprocessing model for tabular data using Keras pipeline    
    #--------------------------------------------------------------------------  
    def sequence_padding(self, sequence, pad_value=0, pad_length=50):

        '''
        Normalizes a series of values.
    
        Keyword arguments:
            series (list): A list of values to be normalized
    
        Returns:
            list: A list of normalized values
        
        '''
        processed_series = preprocessing.sequence.pad_sequences([sequence], maxlen=pad_length, value=pad_value, 
                                                                dtype='float32', padding = 'post')
        pp_seq = processed_series[0]        
        pp_seq = ' '.join([str(x) for x in pp_seq])

        return pp_seq 

    #--------------------------------------------------------------------------
    def sequence_recovery(self, sequences, pad_value, normalizer, 
                          from_reference=False, reference=None):

        def unpadding(seq, pad_value):
            pad_value = normalizer.inverse_transform(np.array([pad_value]).reshape(1, 1)) 
            length = len([x for x in seq if x != pad_value.item()])            
            return seq[:length]
        
        if from_reference==True and reference is not None:
            reference_lens = [len(x) for x in reference]
            scaled_sequences = normalizer.inverse_transform(sequences)
            unpadded_sequences = [x[:l] for x, l in zip(scaled_sequences, reference_lens)] 
        else:
            scaled_sequences = normalizer.inverse_transform(sequences)            
            unpadded_sequences = [unpadding(x, pad_value) for x in scaled_sequences]            
        
        return unpadded_sequences      
        
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
          

