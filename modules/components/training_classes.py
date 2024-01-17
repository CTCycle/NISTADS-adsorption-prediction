import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Embedding, Dropout, Dense, Conv1D, LayerNormalization, BatchNormalization, Add

                   

# [CALLBACK FOR REAL TIME TRAINING MONITORING]
#==============================================================================
#==============================================================================
#==============================================================================
class RealTimeHistory(keras.callbacks.Callback):
    
    ''' 
    A class including the callback to show a real time plot of the training history. 
      
    Methods:
        
    __init__(plot_path): initializes the class with the plot savepath       
    
    ''' 
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
        if epoch % 10 == 0:                    
            self.epochs.append(epoch)
            self.loss_hist.append(logs[list(logs.keys())[0]])
            self.metric_hist.append(logs[list(logs.keys())[1]])
            if self.validation==True:
                self.loss_val_hist.append(logs[list(logs.keys())[2]])            
                self.metric_val_hist.append(logs[list(logs.keys())[3]])
        if epoch % 40 == 0:            
            #------------------------------------------------------------------
            fig_path = os.path.join(self.plot_path, 'training_history.jpeg')
            plt.subplot(2, 1, 1)
            plt.plot(self.epochs, self.loss_hist, label='training loss')
            if self.validation==True:
                plt.plot(self.epochs, self.loss_val_hist, label='validation loss')
                plt.legend(loc = 'best', fontsize = 8)
            plt.title('Loss plot')
            plt.ylabel('MSE')
            plt.xlabel('epoch')
            plt.subplot(2, 1, 2)
            plt.plot(self.epochs, self.metric_hist, label='train metrics') 
            if self.validation==True: 
                plt.plot(self.epochs, self.metric_val_hist, label='validation metrics') 
                plt.legend(loc = 'best', fontsize = 8)
            plt.title('metrics plot')
            plt.ylabel('MAE')
            plt.xlabel('epoch')       
            plt.tight_layout()
            plt.savefig(fig_path, bbox_inches = 'tight', format = 'jpeg', dpi = 300)
            plt.close()     

# [PARAMETRIZER BLOCK]
#==============================================================================
# Parametrizer custom layer
#==============================================================================
class Parametrizer(keras.Layer):
    def __init__(self):
        super(Parametrizer, self).__init__()        
        self.dense1 = Dense(128, activation = 'relu', kernel_initializer='he_uniform')              
        self.dense2 = Dense(256, activation = 'relu', kernel_initializer='he_uniform')              
        self.dense3 = Dense(512, activation = 'relu', kernel_initializer='he_uniform')
        self.dense4 = Dense(512, activation='relu', kernel_initializer='he_uniform')          
        self.layernorm = LayerNormalization()                 

    # implement parametrizer through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs):
        layer = self.dense1(inputs)       
        layer = self.dense2(layer)       
        layer = self.dense3(layer)
        output = self.layernorm(inputs + layer)
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(Parametrizer, self).get_config()
        config.update({})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        

# [BATCH NORMALIZED FFW]
#==============================================================================
# Custom layer
#============================================================================== 
class BNFeedForward(keras.Layer):
    def __init__(self, units, seed=42, dropout=0.1):
        super(BNFeedForward, self).__init__()
        self.units = units   
        self.seed = seed  
        self.dropout = dropout
        self.BN = BatchNormalization(axis=-1, epsilon=0.001)  
        self.drop = Dropout(rate=dropout, seed=seed)      
        self.dense = Dense(units, activation='relu', kernel_initializer='he_uniform')
        
    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, training):        
        layer = self.dense(inputs)
        layer = self.BN(layer)       
        output = self.drop(layer, training=training)                
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(BNFeedForward, self).get_config()
        config.update({'units': self.units,
                       'seed': self.seed,
                       'dropout': self.dropout})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)      

# [GUEST-HOST ENCODER]
#==============================================================================
# Custom layer
#============================================================================== 
class GHEncoder(keras.Layer):
    def __init__(self, gvocab_size, hvocab_size, embedding_dims, seed=42, dropout=0.1):
        super(GHEncoder, self).__init__()
        self.gvocab_size = gvocab_size
        self.hvocab_size = hvocab_size
        self.embedding_dims = embedding_dims
        self.seed = seed  
        self.dropout = dropout        
        self.embedding_G = Embedding(input_dim=gvocab_size, output_dim=self.embedding_dims)
        self.embedding_H = Embedding(input_dim=hvocab_size, output_dim=self.embedding_dims)              
        self.dense1 = Dense(256, activation = 'relu', kernel_initializer='he_uniform')    
        self.dense2 = Dense(512, activation = 'relu',kernel_initializer='he_uniform') 
        self.resdense_G = Dense(512, activation = 'relu', kernel_initializer='he_uniform')    
        self.resdense_H = Dense(512, activation = 'relu',kernel_initializer='he_uniform')
        self.layernorm1 = LayerNormalization() 
        self.layernorm2 = LayerNormalization()         
        self.bnffn = BNFeedForward(1024, self.seed, dropout)             
        
        
    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, guests, hosts, training):        
        G_emb = self.embedding_G(guests)
        H_emb = self.embedding_G(hosts)
        layernorm1 = self.layernorm1(G_emb + H_emb)
        layer = self.dense1(layernorm1)
        layer = self.dense2(layer)   
        reslayer_G = self.resdense_G(G_emb)  
        reslayer_H = self.resdense_H(H_emb)
        layernorm2 = self.layernorm1(reslayer_G + reslayer_H)
        residual = layer + layernorm2
        output = self.bnffn(residual, training)                      
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(GHEncoder, self).get_config()
        config.update({'gvocab_size': self.gvocab_size,
                       'hvocab_size': self.hvocab_size,
                       'embedding_dims': self.embedding_dims,
                       'seed': self.seed,
                       'dropout': self.dropout})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config) 
    
# [GUEST-HOST ENCODER]
#==============================================================================
# Custom layer
#============================================================================== 
class PressureEncoder(keras.Layer):
    def __init__(self, gvocab_size, hvocab_size, embedding_dims, seed=42, dropout=0.1):
        super(PressureEncoder, self).__init__()
        self.gvocab_size = gvocab_size
        self.hvocab_size = hvocab_size
        self.embedding_dims = embedding_dims
        self.seed = seed  
        self.dropout = dropout        
        self.embedding_G = Embedding(input_dim=gvocab_size, output_dim=self.embedding_dims)
        self.embedding_H = Embedding(input_dim=hvocab_size, output_dim=self.embedding_dims)              
        self.dense1 = Dense(256, activation = 'relu', kernel_initializer='he_uniform')    
        self.dense2 = Dense(512, activation = 'relu',kernel_initializer='he_uniform') 
        self.resdense_G = Dense(512, activation = 'relu', kernel_initializer='he_uniform')    
        self.resdense_H = Dense(512, activation = 'relu',kernel_initializer='he_uniform')
        self.layernorm1 = LayerNormalization() 
        self.layernorm2 = LayerNormalization()         
        self.bnffn = BNFeedForward(1024, self.seed, dropout)             
        
        
    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, guests, hosts, training):        
        G_emb = self.embedding_G(guests)
        H_emb = self.embedding_G(hosts)
        layernorm1 = self.layernorm1(G_emb + H_emb)
        layer = self.dense1(layernorm1)
        layer = self.dense2(layer)   
        reslayer_G = self.resdense_G(G_emb)  
        reslayer_H = self.resdense_H(H_emb)
        layernorm2 = self.layernorm1(reslayer_G + reslayer_H)
        residual = layer + layernorm2
        output = self.bnffn(residual, training)                      
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(GHEncoder, self).get_config()
        config.update({'gvocab_size': self.gvocab_size,
                       'hvocab_size': self.hvocab_size,
                       'embedding_dims': self.embedding_dims,
                       'seed': self.seed,
                       'dropout': self.dropout})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config) 
    
# [COLOR CODE MODEL]
#==============================================================================
# collection of model and submodels
#==============================================================================
class SCADSModel:

    def __init__(self, learning_rate, num_features, pad_length, pad_value, adsorbent_dims, 
                 adsorbates_dims, embedding_dims, seed, XLA_acceleration=False):

        self.learning_rate = learning_rate
        self.num_features = num_features
        self.pad_length = pad_length
        self.pad_value = pad_value
        self.adsorbent_dims = adsorbent_dims
        self.adsorbates_dims = adsorbates_dims 
        self.embedding_dims = embedding_dims
        self.seed = seed
        self.XLA_state = XLA_acceleration 

        self.parametrizer = Parametrizer()
        self.embedder = GHEncoder(adsorbates_dims, adsorbent_dims, embedding_dims, seed, dropout=0.2)

        

    # build model given the architecture
    #--------------------------------------------------------------------------
    def build(self):
        
        
        continuous_inputs = Input(shape = (self.num_features, ), name = 'continuous_input')
        adsorbent_inputs = Input(shape = (1,), name = 'adsorbents_input')
        adsorbate_inputs = Input(shape = (1,), name = 'sorbates_input')
        pressure_inputs = Input(shape = (self.pad_length, ), name = 'pressure_input')
        #----------------------------------------------------------------------
        parameters = self.parametrizer(continuous_inputs)
        GH_encoding = self.embedder(adsorbent_inputs, adsorbate_inputs)        
        pressure_encoding = pressure_submodel(pressure_inputs) 
        #----------------------------------------------------------------------       
        embedding_concat = Concatenate()([adsorbent_block, adsorbate_block])                 
        dense1 = Dense(1024, activation = 'relu')(embedding_concat)                 
        dense2 = Dense(512, activation = 'relu')(dense1)       
        dense3 = Dense(512, activation = 'relu')(dense2)       
        model_concat = Concatenate()([dense3, features_block, pressure_block, pressure_inputs])        
        dense4 = Dense(1024, activation = 'relu')(model_concat)                
        dense5 = Dense(512, activation = 'relu')(dense4)                
        dense6 = Dense(256, activation = 'relu')(dense5)
        #----------------------------------------------------------------------        
        output = Dense(self.pad_length, activation = 'relu')(dense6) 
        

        inputs = [continuous_inputs, adsorbent_inputs, adsorbate_inputs, pressure_inputs]    
        model = Model(inputs = inputs, outputs = output, name = 'SCADS')       
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = keras.losses.MeanSquaredError()
        metrics = keras.metrics.MeanAbsoluteError()
        model.compile(loss = loss, optimizer = opt, metrics = metrics, run_eagerly=False,
                      jit_compile=self.XLA_state)            
        
        return model

# Class for preprocessing tabular data prior to GAN training 
#==============================================================================
#==============================================================================
#==============================================================================
class REFERENCE:

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

    
    def ContFeatModel(self):

        input_layer = Input(shape = (self.num_features, ), name = 'continuous_input')
        #----------------------------------------------------------------------
        dense1 = Dense(128, activation = 'relu')(input_layer)                
        dense2 = Dense(256, activation = 'relu')(dense1)                
        dense3 = Dense(512, activation = 'relu')(dense2)        
        dense4 = Dense(760, activation = 'relu')(dense3) 
        #----------------------------------------------------------------------              
        output = Dense(1024, activation = 'relu')(dense4)

        submodel = Model(inputs = input_layer, outputs = output, name = 'CF_model')

        return submodel 

       
    def EmbeddingModel(self, input_dims): 
   
        embedding_inputs = Input(shape = (1, ), name = 'embedded_input')
        #----------------------------------------------------------------------
        embedding = Embedding(input_dim = input_dims, output_dim = self.embedding_dims)(embedding_inputs)        
        flatten = Flatten()(embedding)        
        dense1 = Dense(1024, activation = 'relu')(flatten)       
        dense2 = Dense(512, activation = 'relu')(dense1)        
        dense3 = Dense(512, activation = 'relu')(dense2)                    
        dense4 = Dense(256, activation = 'relu')(dense3)                
        output = Dense(256, activation = 'relu')(dense4)
        #----------------------------------------------------------------------
        submodel = Model(inputs = embedding_inputs, outputs = output)

        return submodel
    
   
    def PressureModel(self):       

        pressure_inputs = Input(shape = (self.pad_length, ), name = 'pressure_input')
        #----------------------------------------------------------------------           
        mask = Masking(mask_value=self.pad_value)(pressure_inputs)        
        reshape = Reshape((-1, self.pad_length))(mask)                
        conv1 = Conv1D(256, kernel_size = 6, padding='same', activation='relu')(reshape)       
        pool1 = AveragePooling1D(pool_size = 2, strides=None, padding='same')(conv1)                
        conv2 = Conv1D(512, kernel_size = 6, padding='same', activation='relu')(pool1)       
        pool2 = AveragePooling1D(pool_size = 2, strides=None, padding='same')(conv2)        
        conv3 = Conv1D(1024, kernel_size = 6, padding='same', activation='relu')(pool2)  
        #----------------------------------------------------------------------     
        flatten = Flatten()(conv3)        
        dense1 = Dense(1024, activation = 'relu')(flatten)               
        dense2 = Dense(512, activation = 'relu')(dense1)                        
        dense3 = Dense(512, activation = 'relu')(dense2)                
        dense4 = Dense(256, activation = 'relu')(dense3)         
        output = Dense(256, activation = 'relu')(dense4)
        #----------------------------------------------------------------------
        submodel = Model(inputs = pressure_inputs, outputs = output, name = 'PS_submodel')

        return submodel    
    
    
    def SCADS(self):

        ContFeat_submodel = self.ContFeatModel()
        adsorbent_submodel = self.EmbeddingModel(self.adsorbent_dims)
        adsorbate_submodel = self.EmbeddingModel(self.adsorbates_dims)        
        pressure_submodel = self.PressureModel() 
        #----------------------------------------------------------------------
        continuous_inputs = Input(shape = (self.num_features, ), name = 'continuous_input')
        adsorbent_inputs = Input(shape = (1, ), name = 'adsorbents_input')
        adsorbate_inputs = Input(shape = (1, ), name = 'sorbates_input')
        pressure_inputs = Input(shape = (self.pad_length, ), name = 'pressure_input')
        #----------------------------------------------------------------------
        features_block = ContFeat_submodel(continuous_inputs)
        adsorbent_block = adsorbent_submodel(adsorbent_inputs)
        adsorbate_block = adsorbate_submodel(adsorbate_inputs)
        pressure_block = pressure_submodel(pressure_inputs) 
        #----------------------------------------------------------------------       
        embedding_concat = Concatenate()([adsorbent_block, adsorbate_block])                 
        dense1 = Dense(1024, activation = 'relu')(embedding_concat)                 
        dense2 = Dense(512, activation = 'relu')(dense1)       
        dense3 = Dense(512, activation = 'relu')(dense2)       
        model_concat = Concatenate()([dense3, features_block, pressure_block, pressure_inputs])        
        dense4 = Dense(1024, activation = 'relu')(model_concat)                
        dense5 = Dense(512, activation = 'relu')(dense4)                
        dense6 = Dense(256, activation = 'relu')(dense5)
        #----------------------------------------------------------------------        
        output = Dense(self.pad_length, activation = 'relu')(dense6) 
        

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

        
          
    
            
