import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from keras.api._v2.keras import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from tqdm import tqdm
tqdm.pandas()
      

# [DATA PREPROCESSING]
#==============================================================================
# preprocess adsorption data
#==============================================================================
class PreProcessing:  


    def __init__(self):        
        self.parameters = ['temperature', 'mol_weight', 'complexity', 'covalent_units', 
                         'H_acceptors', 'H_donors', 'heavy_atoms']
        self.ads_col, self.sorb_col  = ['adsorbent_name'], ['adsorbates_name'] 
        self.P_col, self.Q_col  = 'pressure_in_Pascal', 'uptake_in_mol_g'
        self.P_unit_col, self.Q_unit_col  = 'pressureUnits', 'adsorptionUnits'   

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
    
          
# [DATA VALIDATION]
#==============================================================================
# 
#==============================================================================
class DataValidation:   


    def __init__(self):
        
        self.parameters = ['temperature', 'mol_weight', 'complexity', 'covalent_units', 
                           'H_acceptors', 'H_donors', 'heavy_atoms']
        self.categoricals  = ['adsorbent_name', 'adsorbates_name'] 
        self.sequences  = ['pressure_in_Pascal', 'uptake_in_mol_g']        

    #--------------------------------------------------------------------------
    def check_missing_values(self, dataset):

        '''
        Checks for missing values in each column of the dataset 
        and prints a report of the findings.

        Keyword arguments:
            dataset (DataFrame): The dataset to be checked for missing values.

        Returns:
            Series: A pandas Series object where the index corresponds to the column names of the dataset and 
                    the values are the counts of missing values in each column.

        '''
        missing_values = dataset.isnull().sum()
        if missing_values.any():
            print(f'{len(missing_values)} columns have missing values:')
            print(missing_values[missing_values > 0])            
        else:
            print('No columns with missing values\n')

        return missing_values         

    #--------------------------------------------------------------------------
    def plot_histograms(self, dataset, path):

        '''
        Generates histograms for specified columns in a dataset and saves 
        them as JPEG files to a given directory. This function iterates through 
        a list of columns, generating a histogram for each. Each histogram is 
        saved with a filename indicating the column it represents.

        Keyword arguments:
            dataset (DataFrame): The dataset from which to generate histograms.
            path (str): The directory path where the histogram images will be saved.

        Return:
            None

        '''
        histogram_cols = self.parameters + self.sequences        
        for column in tqdm(histogram_cols):
            plot_path = os.path.join(path, f'{column}_histogram.jpeg')
            values = dataset[column].values
            if column in self.sequences:
                values = dataset[column].explode(column).values                          
            plt.figure(figsize=(8, 6))
            plt.hist(values, bins=20, color='skyblue', edgecolor='black')
            plt.title(f'histogram_{column}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')             
            plt.grid(True)                 
            plt.tight_layout()
            plt.savefig(plot_path, bbox_inches='tight', format='jpeg', dpi=300)
            plt.close()

    #--------------------------------------------------------------------------
    def features_comparison(self, train_X, test_X, train_Y, test_Y):

        '''
        Compares the statistical properties (mean and standard deviation) of training and testing 
        datasets for both features and labels.

        Keyword arguments:
            train_X (DataFrame): The training set features.
            test_X (DataFrame): The testing set features.
            train_Y (Series/DataFrame): The training set labels.
            test_Y (Series/DataFrame): The testing set labels.

        Returns:
            dict: A dictionary containing the mean and standard deviation differences for each column in the features, 
                and for the labels, under the key 'Y'. Each entry is a dictionary with keys 'mean_diff' and 'std_diff'.

        '''
        stats = {}  
        features_cols = self.parameters + [self.sequences[0]]   
        
        for col in features_cols:
            if col == self.sequences[0]:
                train_X[col] = train_X[col].explode(col) 
                test_X[col] = test_X[col].explode(col) 
            train_mean, test_mean = train_X[col].mean(), test_X[col].mean()
            train_std, test_std = train_X[col].std(), test_X[col].std()
            mean_diff = abs(train_mean - test_mean)
            std_diff = abs(train_std - test_std)
            stats[col] = [mean_diff, std_diff]
            stats[col] = {'mean_diff': mean_diff, 'std_diff': std_diff}

        train_Y, test_Y = train_Y.explode(), test_Y.explode()
        train_mean_Y, test_mean_Y = train_Y.mean(), test_Y.mean()
        train_std_Y, test_std_Y = train_Y.std(), test_Y.std()
        mean_diff_Y = abs(train_mean_Y - test_mean_Y)
        std_diff_Y = abs(train_std_Y - test_std_Y)
        stats['Y'] = {'mean_diff': mean_diff_Y, 'std_diff': std_diff_Y}

        return stats
    
    #--------------------------------------------------------------------------
    def data_split_validation(self, dataset, test_size, range_val):

        '''
        Evaluates various train-test splits to find the one where the difference in statistical properties (mean and standard deviation) 
        between the training and testing sets is minimized for both features and labels.

        Keyword arguments:
            dataset (DataFrame): The dataset to be split into training and testing sets.
            test_size (float): The proportion of the dataset to include in the test split.
            range_val (int): The number of different splits to evaluate.

        Returns:
            tuple: Contains the minimum difference found across all splits, the seed for the best split, and the statistics 
                for this split. The statistics are a dictionary with keys for each feature and 'Y' for labels, 
                where each entry is a dictionary with 'mean_diff' and 'std_diff'.
        '''
        inputs = dataset[[x for x in dataset.columns if x != self.sequences[1]]]
        labels = dataset[self.sequences[1]]

        min_diff = float('inf')
        best_i = None
        best_stats = None
        for i in tqdm(range(range_val)):
            train_X, test_X, train_Y, test_Y = train_test_split(inputs, labels, test_size=test_size, 
                                                                random_state=i+1, shuffle=True, 
                                                                stratify=None)
            # function call to compare columns by mean and standard deviation
            stats = self.features_comparison(train_X, test_X, train_Y, test_Y)
            # Calculate total difference for this split
            total_diff = sum([sum(values.values()) for key, values in stats.items()])
            # Update values only if the difference is lower in this iteration
            if total_diff < min_diff:
                min_diff = total_diff
                best_seed = i + 1
                best_stats = stats        

        return min_diff, best_seed, best_stats