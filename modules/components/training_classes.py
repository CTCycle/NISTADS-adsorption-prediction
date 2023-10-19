import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras import layers 
from tensorflow.keras.preprocessing.sequence import pad_sequences  
                   
# [CALLBACK FOR REAL TIME TRAINING MONITORING]
#==============================================================================
#==============================================================================
#==============================================================================
class RealTimeHistory(keras.callbacks.Callback):
    
    """ 
    A class including the callback to show a real time plot of the training history. 
      
    Methods:
        
    __init__(plot_path): initializes the class with the plot savepath       
    
    """   
    def __init__(self, plot_path, validation=True):        
        super().__init__()
        self.plot_path = plot_path
        self.epochs = []
        self.loss_hist = []
        self.metric_hist = []
        self.loss_val_hist = []        
        self.metric_val_hist = []
        self.validation = validation            
    #--------------------------------------------------------------------------
    def on_epoch_end(self, epoch, logs = {}):
        if epoch % 2 == 0:                    
            self.epochs.append(epoch)
            self.loss_hist.append(logs[list(logs.keys())[0]])
            self.metric_hist.append(logs[list(logs.keys())[1]])
            if self.validation==True:
                self.loss_val_hist.append(logs[list(logs.keys())[2]])            
                self.metric_val_hist.append(logs[list(logs.keys())[3]])
        if epoch % 50 == 0:            
            #------------------------------------------------------------------
            fig_path = os.path.join(self.plot_path, 'training_history.jpeg')
            plt.subplot(2, 1, 1)
            plt.plot(self.epochs, self.loss_hist, label = 'training loss')
            if self.validation==True:
                plt.plot(self.epochs, self.loss_val_hist, label = 'validation loss')
                plt.legend(loc = 'best', fontsize = 8)
            plt.title('Loss plot')
            plt.ylabel('MSE')
            plt.xlabel('epoch')
            plt.subplot(2, 1, 2)
            plt.plot(self.epochs, self.metric_hist, label = 'train metrics') 
            if self.validation==True: 
                plt.plot(self.epochs, self.metric_val_hist, label = 'validation metrics') 
                plt.legend(loc = 'best', fontsize = 8)
            plt.title('metrics plot')
            plt.ylabel('MAE')
            plt.xlabel('epoch')       
            plt.tight_layout()
            plt.savefig(fig_path, bbox_inches = 'tight', format = 'jpeg', dpi = 300)
            plt.show(block = False)
            plt.close()      

# [CUSTOM DATA GENERATOR FOR TRAINING]
#==============================================================================
#==============================================================================
#==============================================================================
class DataGenerator(keras.utils.Sequence):

    def __init__(self, dataframe_X, dataframe_Y, pad_length, pad_value, batch_size, shuffle=True):        
        self.dataframe_X = dataframe_X  
        self.dataframe_Y = dataframe_Y       
        self.num_of_samples = dataframe_X.shape[0]
        self.ads_col = 'adsorbent_name'
        self.sorb_col = 'adsorbates_name'
        self.P_col = 'pressure_in_Pascal'
        self.Y_col = 'uptake_in_mol/g'
        self.feat_cols = ['temperature', 'mol_weight', 'complexity', 
                          'covalent_units', 'H_acceptors', 'H_donors', 
                          'heavy_atoms']       
        
        self.batch_size = batch_size  
        self.batch_index = 0 
        self.pad_length = pad_length 
        self.pad_value = pad_value            
        self.shuffle = shuffle
        self.on_epoch_end()       

    # define length of the custom generator      
    #--------------------------------------------------------------------------
    def __len__(self):
        return int(np.ceil(self.num_of_samples/self.batch_size))
    
    # define method to get X and Y data through custom functions, and subsequently
    # create a batch of data converted to tensors
    #--------------------------------------------------------------------------
    def __getitem__(self, idx): 
        features_batch = self.dataframe_X[self.feat_cols].iloc[idx * self.batch_size:(idx + 1) * self.batch_size].values        
        adsorbent_batch = self.dataframe_X[self.ads_col][idx * self.batch_size:(idx + 1) * self.batch_size]
        sorbates_batch = self.dataframe_X[self.sorb_col][idx * self.batch_size:(idx + 1) * self.batch_size]
        pressure_batch = self.dataframe_X[self.P_col][idx * self.batch_size:(idx + 1) * self.batch_size]
        uptake_batch = self.dataframe_Y[self.Y_col][idx * self.batch_size:(idx + 1) * self.batch_size]        
        x1_batch = [self.__features_generation(feat) for feat in features_batch]
        x2_batch = [self.__token_generation(token) for token in adsorbent_batch]
        x3_batch = [self.__token_generation(token) for token in sorbates_batch]
        x4_batch = [self.__series_generation(series) for series in pressure_batch] 
        y_batch = [self.__series_generation(series) for series in uptake_batch]
        X1_tensor = tf.convert_to_tensor(x1_batch)
        X2_tensor = tf.convert_to_tensor(x2_batch)
        X3_tensor = tf.convert_to_tensor(x3_batch)
        X4_tensor = tf.convert_to_tensor(x4_batch)
        Y_tensor = tf.convert_to_tensor(y_batch)

        return (X1_tensor, X2_tensor, X3_tensor, X4_tensor), Y_tensor
    
    # define method to perform data operations on epoch end
    #--------------------------------------------------------------------------
    def on_epoch_end(self):        
        self.indexes = np.arange(self.num_of_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    # define method to load images and perform data augmentation    
    #--------------------------------------------------------------------------
    def __features_generation(self, features):
        features = np.array(features, dtype=np.float32)         

        return features   

    # define method to load images and perform data augmentation    
    #--------------------------------------------------------------------------
    def __series_generation(self, sequence):
        pp_seq = sequence.replace('[', '').replace(']', '')
        series = [float(x) for x in pp_seq.split(', ')]
        max_val = max([float(g) for g in series])
        if max_val == 0.0:
            max_val = 10e-14
        norm_series = [x/max_val for x in series]
        padded_series = pad_sequences([[norm_series]], maxlen = self.pad_length, value = self.pad_value, 
                                      dtype = 'float32', padding = 'post')

        padded_series = np.array(padded_series[0], dtype=np.float32)

        return padded_series   
    
    # define method to load labels    
    #--------------------------------------------------------------------------
    def __token_generation(self, token):
        token = np.array(token, dtype=np.float32)        

        return token
    
    # define method to call the elements of the generator    
    #--------------------------------------------------------------------------
    def next(self):
        next_index = (self.batch_index + 1) % self.__len__()
        self.batch_index = next_index

        return self.__getitem__(next_index)
  
             
# Class for preprocessing tabular data prior to GAN training 
#==============================================================================
#==============================================================================
#==============================================================================
class SCADSModel:

    def __init__(self, learning_rate, num_features, pad_length, pad_value, adsorbent_dims, 
                 adsorbates_dims, embedding_dims, XLA_acceleration=False):

        self.learning_rate = learning_rate
        self.num_features = num_features
        self.pad_length = pad_length
        self.pad_value = pad_value
        self.adsorbent_dims = adsorbent_dims
        self.adsorbates_dims = adsorbates_dims 
        self.embedding_dims = embedding_dims
        self.XLA_state = XLA_acceleration

    #--------------------------------------------------------------------------
    def ContFeatModel(self):

        input_layer = layers.Input(shape = (self.num_features, ), name = 'continuous_input')

        dense1 = layers.Dense(128, activation = 'relu')(input_layer)
        #----------------------------------------------------------------------        
        dense2 = layers.Dense(256, activation = 'relu')(dense1)
        #----------------------------------------------------------------------        
        dense3 = layers.Dense(512, activation = 'relu')(dense2)
        #----------------------------------------------------------------------
        dense4 = layers.Dense(760, activation = 'relu')(dense3)
        #----------------------------------------------------------------------       
        output = layers.Dense(1024, activation = 'relu')(dense4)

        submodel = Model(inputs = input_layer, outputs = output, name = 'CF_model')

        return submodel 

    #--------------------------------------------------------------------------    
    def EmbeddingModel(self, input_dims): 
   
        embedding_inputs = layers.Input(shape = (1, ), name = 'embedded_input')

        embedding = layers.Embedding(input_dim = input_dims, output_dim = self.embedding_dims)(embedding_inputs)
        #----------------------------------------------------------------------
        flatten = layers.Flatten()(embedding)
        #----------------------------------------------------------------------
        dense1 = layers.Dense(1024, activation = 'relu')(flatten)        
        #----------------------------------------------------------------------
        dense2 = layers.Dense(512, activation = 'relu')(dense1)
        #----------------------------------------------------------------------
        dense3 = layers.Dense(512, activation = 'relu')(dense2)   
        #----------------------------------------------------------------------            
        dense4 = layers.Dense(256, activation = 'relu')(dense3)
        #----------------------------------------------------------------------        
        output = layers.Dense(256, activation = 'relu')(dense4)

        submodel = Model(inputs = embedding_inputs, outputs = output)

        return submodel
    
    #--------------------------------------------------------------------------
    def PressureModel(self):       

        pressure_inputs = layers.Input(shape = (self.pad_length, ), name = 'pressure_input')
           
        mask = layers.Masking(mask_value=self.pad_value)(pressure_inputs)
        #----------------------------------------------------------------------
        reshape = layers.Reshape((-1, self.pad_length))(mask)        
        #----------------------------------------------------------------------
        conv1 = layers.Conv1D(256, kernel_size = 6, padding='same', activation='relu')(reshape)        
        #----------------------------------------------------------------------
        pool1 = layers.AveragePooling1D(pool_size = 2, strides=None, padding='same')(conv1)
        #----------------------------------------------------------------------        
        conv2 = layers.Conv1D(512, kernel_size = 6, padding='same', activation='relu')(pool1)        
        #----------------------------------------------------------------------
        pool2 = layers.AveragePooling1D(pool_size = 2, strides=None, padding='same')(conv2)
        #----------------------------------------------------------------------
        conv3 = layers.Conv1D(1024, kernel_size = 6, padding='same', activation='relu')(pool2)        
        #----------------------------------------------------------------------
        flatten = layers.Flatten()(conv3)
        #----------------------------------------------------------------------
        dense1 = layers.Dense(1024, activation = 'relu')(flatten) 
        #----------------------------------------------------------------------        
        dense2 = layers.Dense(512, activation = 'relu')(dense1) 
        #----------------------------------------------------------------------               
        dense3 = layers.Dense(512, activation = 'relu')(dense2) 
        #----------------------------------------------------------------------       
        dense4 = layers.Dense(256, activation = 'relu')(dense3) 
        #----------------------------------------------------------------------
        output = layers.Dense(256, activation = 'relu')(dense4)

        submodel = Model(inputs = pressure_inputs, outputs = output, name = 'PS_submodel')

        return submodel    
    
    #--------------------------------------------------------------------------
    def SCADS(self):

        ContFeat_submodel = self.ContFeatModel()
        adsorbent_submodel = self.EmbeddingModel(self.adsorbent_dims)
        adsorbate_submodel = self.EmbeddingModel(self.adsorbates_dims)        
        pressure_submodel = self.PressureModel() 

        continuous_inputs = layers.Input(shape = (self.num_features, ), name = 'continuous_input')
        adsorbent_inputs = layers.Input(shape = (1, ), name = 'adsorbents_input')
        adsorbate_inputs = layers.Input(shape = (1, ), name = 'sorbates_input')
        pressure_inputs = layers.Input(shape = (self.pad_length, ), name = 'pressure_input')

        features_block = ContFeat_submodel(continuous_inputs)
        adsorbent_block = adsorbent_submodel(adsorbent_inputs)
        adsorbate_block = adsorbate_submodel(adsorbate_inputs)
        pressure_block = pressure_submodel(pressure_inputs) 
        
        
        embedding_concat = layers.Concatenate()([adsorbent_block, adsorbate_block])             
        #----------------------------------------------------------------------         
        dense1 = layers.Dense(1024, activation = 'relu')(embedding_concat)        
        #----------------------------------------------------------------------          
        dense2 = layers.Dense(512, activation = 'relu')(dense1)
        #----------------------------------------------------------------------
        dense3 = layers.Dense(512, activation = 'relu')(dense2)
        #----------------------------------------------------------------------
        model_concat = layers.Concatenate()([dense3, features_block, pressure_block, pressure_inputs])
        #----------------------------------------------------------------------
        dense4 = layers.Dense(1024, activation = 'relu')(model_concat)
        #----------------------------------------------------------------------        
        dense5 = layers.Dense(512, activation = 'relu')(dense4)
        #----------------------------------------------------------------------        
        dense6 = layers.Dense(256, activation = 'relu')(dense5)
        #----------------------------------------------------------------------          
        output = layers.Dense(self.pad_length, activation = 'relu')(dense6) 
        #----------------------------------------------------------------------

        inputs = [continuous_inputs, adsorbent_inputs, adsorbate_inputs, pressure_inputs]    
        model = Model(inputs = inputs, outputs = output, name = 'SCADS')       
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = keras.losses.MeanSquaredError()
        metrics = keras.metrics.MeanAbsoluteError()
        model.compile(loss = loss, optimizer = opt, metrics = metrics, run_eagerly=False,
                      jit_compile=self.XLA_state)            
        
        return model
    

              

# define model class
#==============================================================================
#==============================================================================
#==============================================================================
class ModelTraining:
    
    """     
    A class for training operations. Includes many different methods that can 
    be used in sequence to build a functional
    preprocessing pipeline.
      
    Methods:
        
    __init__(df_SC, df_BN): initializes the class with the single component 
                            and binary mixrture datasets
    
    training_logger(path, model_name):     write the training session info in a txt file
    prevaluation_model(model, n_features): evaluates the model with dummy datasets 
              
    """    
    def __init__(self, device = 'default', seed=42):                   
        np.random.seed(seed)
        tf.random.set_seed(seed)         
        self.available_devices = tf.config.list_physical_devices()
        self.available_devices = tf.config.list_physical_devices()
        print('----------------------------------------------------------------')
        print('The current devices are available: ')
        for dev in self.available_devices:
            print()
            print(dev)
        print()
        print('----------------------------------------------------------------')
        if device == 'GPU':
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)   
            self.physical_devices = tf.config.list_physical_devices('GPU')
            tf.config.set_visible_devices(self.physical_devices[0], 'GPU')
            print('GPU is set as active device')
            print('----------------------------------------------------------------')
            print()        
        elif device == 'CPU':
            tf.config.set_visible_devices([], 'GPU')
            print('CPU is set as active device')
            print('----------------------------------------------------------------')
            print()     

    #========================================================================== 
    def model_parameters(self, parameters_dict, savepath): 
        path = os.path.join(savepath, 'model_parameters.txt')      
        with open(path, 'w') as f:
            for key, value in parameters_dict.items():
                f.write(f'{key}: {value}\n')
    
    #========================================================================== 
    def load_pretrained_model(self, path):
        
        model_folders = []
        for entry in os.scandir(path):
            if entry.is_dir():
                model_folders.append(entry.name)
        model_folders.sort()
        index_list = [idx + 1 for idx, item in enumerate(model_folders)]     
        print('Please select a pretrained model:') 
        print()
        for i, directory in enumerate(model_folders):
            print('{0} - {1}'.format(i + 1, directory))        
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
           
        model_path = os.path.join(path, model_folders[dir_index - 1])
        model = keras.models.load_model(model_path)        
        
        return model
    
# define class for trained model validation and data comparison
#============================================================================== 
#==============================================================================
#==============================================================================
class ModelValidation: 

    def __init__(self, model):

        self.model = model     
    
    # comparison of data distribution using statistical methods 
    #==========================================================================     
    def SCADS_validation(self, X, Y_real, Y_predicted, plot_path):       

        fig_path = os.path.join(plot_path, 'validation_SCADS.jpeg')
        fig, axs = plt.subplots(2, 2)       
        axs[0, 0].plot(X[0], Y_predicted[0], label='Predicted')
        axs[0, 0].plot(X[0], Y_real[0], label='Actual')         
        axs[0, 1].plot(X[1], Y_predicted[1])
        axs[0, 1].plot(X[1], Y_real[1])        
        axs[1, 0].plot(X[2], Y_predicted[2])
        axs[1, 0].plot(X[2], Y_real[2])        
        axs[1, 1].plot(X[3], Y_predicted[3])
        axs[1, 1].plot(X[3], Y_real[3])        
        for ax in axs.flat:
            ax.set_ylabel('mol/g adsorbed')
            ax.set_xlabel('pressure (Pa)')
        fig.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches='tight', format='jpeg', dpi=400)
        plt.show(block=False)
        plt.close()          

        
          
    
            
