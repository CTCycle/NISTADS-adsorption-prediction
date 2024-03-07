import os
import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import Model
from keras import layers
from IPython.display import display
from ipywidgets import Dropdown
                   

# [CALLBACK FOR REAL TIME TRAINING MONITORING]
#==============================================================================
# ... 
#==============================================================================
class RealTimeHistory(keras.callbacks.Callback):    
    
    def __init__(self, plot_path, validation=True):        
        super().__init__()
        self.plot_path = plot_path
        self.epochs = []
        self.loss_hist = []
        self.metric_hist = []
        self.loss_val_hist = []        
        self.metric_val_hist = []
        self.validation = validation            
    
    def on_epoch_end(self, epoch, logs = {}):
        if epoch % 5 == 0:                    
            self.epochs.append(epoch)
            self.loss_hist.append(logs[list(logs.keys())[0]])
            self.metric_hist.append(logs[list(logs.keys())[1]])
            if self.validation==True:
                self.loss_val_hist.append(logs[list(logs.keys())[2]])            
                self.metric_val_hist.append(logs[list(logs.keys())[3]])
        if epoch % 20 == 0:           
            fig_path = os.path.join(self.plot_path, 'training_history.jpeg')
            plt.subplot(2, 1, 1)
            plt.plot(self.epochs, self.loss_hist, label='trains')
            if self.validation==True:
                plt.plot(self.epochs, self.loss_val_hist, label='validation')
                plt.legend(loc = 'best', fontsize = 8)
            plt.title('Loss plot')
            plt.ylabel('MSE')
            plt.xlabel('epoch')
            plt.subplot(2, 1, 2)
            plt.plot(self.epochs, self.metric_hist, label='train') 
            if self.validation==True: 
                plt.plot(self.epochs, self.metric_val_hist, label='validation') 
                plt.legend(loc = 'best', fontsize = 8)
            plt.title('metrics plot')
            plt.ylabel('MAE')
            plt.xlabel('epoch')       
            plt.tight_layout()
            plt.savefig(fig_path, bbox_inches='tight', format='jpeg', dpi=300)
            plt.close()   


# [PARAMETRIZER BLOCK]
#==============================================================================
# Parametrizer custom layer
#==============================================================================
@keras.utils.register_keras_serializable(package='CustomLayers', name='Parametrizer')
class Parametrizer(layers.Layer):
    def __init__(self, seq_length, seed=42, **kwargs):
        super(Parametrizer, self).__init__(**kwargs)
        self.seq_length = seq_length
        self.seed = seed        
        self.dense1 = layers.Dense(256, activation='tanh', kernel_initializer='glorot_uniform')              
        self.dense2 = layers.Dense(368, activation='tanh', kernel_initializer='glorot_uniform')              
        self.dense3 = layers.Dense(512, activation='tanh', kernel_initializer='glorot_uniform')       
        self.dense4 = layers.Dense(512, activation='tanh', kernel_initializer='glorot_uniform')
        self.bnorm = layers.BatchNormalization() 
        self.drop = layers.Dropout(seed=seed, rate=0.2)               

    # implement parametrizer through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, training=True):
        layer = self.dense1(inputs)       
        layer = self.dense2(layer)        
        layer = self.dense3(layer)       
        layer = self.bnorm(layer)
        layer = self.dense4(layer)
        output = self.drop(layer)
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(Parametrizer, self).get_config()
        config.update({'seq_length': self.seq_length,
                       'seed': self.seed})
        return config

    # deserialization method  
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        

# [BATCH NORMALIZED FFW]
#==============================================================================
# Custom layer
#============================================================================== 
@keras.utils.register_keras_serializable(package='CustomLayers', name='BNFeedForward')
class BNFeedForward(layers.Layer):
    def __init__(self, units, seed=42, dropout=0.1, **kwargs):
        super(BNFeedForward, self).__init__(**kwargs)
        self.units = units         
        self.seed = seed  
        self.dropout = dropout
        self.BN = layers.BatchNormalization(axis=-1, epsilon=0.001)  
        self.drop = layers.Dropout(rate=dropout, seed=seed)      
        self.dense = layers.Dense(units, activation='relu', kernel_initializer='he_uniform')
        
    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, training=True):        
        layer = self.dense(inputs)
        layer = self.BN(layer, training=training)       
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

    # deserialization method  
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)      
    

# [GUEST-HOST ENCODER]
#==============================================================================
# Custom layer
#============================================================================== 
@keras.utils.register_keras_serializable(package='Encoders', name='GHEncoder')
class GHEncoder(layers.Layer):
    def __init__(self, seq_length, gvocab_size, hvocab_size, embedding_dims, seed=42, **kwargs):
        super(GHEncoder, self).__init__(**kwargs)
        self.seq_length = seq_length 
        self.gvocab_size = gvocab_size
        self.hvocab_size = hvocab_size
        self.embedding_dims = embedding_dims
        self.seed = seed             
        self.embedding_G = layers.Embedding(input_dim=gvocab_size, output_dim=self.embedding_dims)
        self.embedding_H = layers.Embedding(input_dim=hvocab_size, output_dim=self.embedding_dims)              
        self.dense1 = layers.Dense(256, activation='relu', kernel_initializer='he_uniform')    
        self.dense2 = layers.Dense(368, activation='relu',kernel_initializer='he_uniform') 
        self.dense3 = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')        
        self.layernorm = layers.LayerNormalization()
        self.pooling = layers.GlobalAveragePooling1D()              
        self.bnffn = BNFeedForward(512, self.seed, 0.2)        
        
    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, guests, hosts, training=True):        
        G_emb = self.embedding_G(guests)        
        H_emb = self.embedding_H(hosts)                 
        layer = self.layernorm(G_emb + H_emb)
        layer = self.pooling(layer)        
        layer = self.dense1(layer)              
        layer = self.dense2(layer)
        layer = self.dense3(layer)           
        output = self.bnffn(layer, training=training)                     
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(GHEncoder, self).get_config()
        config.update({'seq_length': self.seq_length,
                       'gvocab_size': self.gvocab_size,
                       'hvocab_size': self.hvocab_size,
                       'embedding_dims': self.embedding_dims,
                       'seed': self.seed})
        return config

    # deserialization method  
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
     
    
# [PRESSURE ENCODER]
#==============================================================================
# Custom layer
#============================================================================== 
@keras.utils.register_keras_serializable(package='Encoders', name='PressureEncoder')
class PressureEncoder(layers.Layer):

    def __init__(self, pad_value=-1, seed=42, **kwargs):
        super(PressureEncoder, self).__init__(**kwargs)
        self.pad_value = pad_value        
        self.seed = seed               
        self.conv1 = layers.Conv1D(64, 6, padding='same', activation='relu', kernel_initializer='he_uniform')
        self.conv2 = layers.Conv1D(128, 6, padding='same', activation='relu', kernel_initializer='he_uniform')
        self.conv3 = layers.Conv1D(256, 6, padding='same', activation='relu', kernel_initializer='he_uniform')
        self.dense1 = layers.Dense(256, activation='relu', kernel_initializer='he_uniform')    
        self.dense2 = layers.Dense(368, activation='relu', kernel_initializer='he_uniform')        
        self.bnffn1 = BNFeedForward(512, self.seed, 0.2) 
        self.bnffn2 = BNFeedForward(512, self.seed, 0.3)
        self.pooling = layers.GlobalAveragePooling1D()                   
        self.supports_masking = True

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, training=True):

        mask = self.compute_mask(inputs)
        inputs = inputs * tf.cast(mask, dtype=inputs.dtype)       
        inputs = tf.expand_dims(inputs, axis=-1)                          
        layer = self.conv1(inputs)  
        layer = self.conv2(layer) 
        layer = self.conv3(layer)       
        layer = self.dense1(layer)
        layer = self.dense2(layer)
        layer = self.pooling(layer)                                
        layer = self.bnffn1(layer, training=training)
        output = self.bnffn2(layer, training=training)                             
        
        return output
    
    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def compute_mask(self, inputs, mask=None):
        mask = tf.math.not_equal(inputs, self.pad_value)       

        return mask     
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(PressureEncoder, self).get_config()        
        config.update({'pad_value': self.pad_value,
                       'seed': self.seed})
        return config

    # deserialization method  
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
     
    
# [UPTAKE DECODER]
#==============================================================================
# Custom layer
#============================================================================== 
@keras.utils.register_keras_serializable(package='Decoder', name='QDecoder')
class QDecoder(layers.Layer):
    def __init__(self, seq_length, seed=42, **kwargs):
        super(QDecoder, self).__init__(**kwargs)
        self.seq_length = seq_length        
        self.seed = seed           
        self.dense1 = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')                 
        self.dense2 = layers.Dense(512, activation='relu', kernel_initializer='he_uniform') 
        self.dense3 = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')        
        self.bnffn1 = BNFeedForward(768, self.seed, 0.1) 
        self.bnffn2 = BNFeedForward(1024, self.seed, 0.2) 
        self.bnffn3 = BNFeedForward(1024, self.seed, 0.2)        
        self.denseout = layers.Dense(seq_length, activation='softplus', dtype='float32') 
        self.layernorm = layers.LayerNormalization()        
        self.supports_masking = True

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, features, embeddings, sequences, mask=None, training=True):
        layer1 = self.dense1(features)
        layer2 = self.dense2(embeddings)
        layer3 = self.dense3(sequences)
        layer = self.layernorm(layer1 + layer2 + layer3)       
        layer = self.bnffn1(layer, training=training)
        layer = self.bnffn2(layer, training=training)
        layer = self.bnffn3(layer, training=training)        
        output = self.denseout(layer)                    
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(QDecoder, self).get_config()        
        config.update({'seq_length': self.seq_length,
                       'seed': self.seed})
        return config

    # deserialization method  
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config) 
    
    
# [SCADS MODEL]
#==============================================================================
# collection of model and submodels
#==============================================================================
class SCADSModel:

    def __init__(self, learning_rate, num_features, sequence_length, pad_value, adsorbent_dims, 
                 adsorbates_dims, embedding_dims, seed=42, XLA_acceleration=False):

        self.learning_rate = learning_rate
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.pad_value = pad_value
        self.adsorbent_dims = adsorbent_dims
        self.adsorbates_dims = adsorbates_dims 
        self.embedding_dims = embedding_dims
        self.seed = seed
        self.XLA_state = XLA_acceleration
        self.parametrizer = Parametrizer(sequence_length, seed)
        self.embedder = GHEncoder(sequence_length, adsorbates_dims, adsorbent_dims, embedding_dims, seed)
        self.encoder = PressureEncoder(pad_value, seed)
        self.decoder = QDecoder(sequence_length, seed)
        
    # build model given the architecture
    #--------------------------------------------------------------------------
    def get_model(self, summary=True):       
       
        # define model inputs using input layers
        feat_inputs = layers.Input(shape = (self.num_features, ))
        host_inputs = layers.Input(shape = (1,))
        guest_inputs = layers.Input(shape = (1,))
        pressure_inputs = layers.Input(shape = (self.sequence_length, ))
               
        parametrizer = self.parametrizer(feat_inputs)
        GH_encoder = self.embedder(host_inputs, guest_inputs)        
        pressure_encoder = self.encoder(pressure_inputs)
        decoder = self.decoder(parametrizer, GH_encoder, pressure_encoder)        
        
        model = Model(inputs=[feat_inputs, host_inputs, guest_inputs, pressure_inputs],
                      outputs=decoder, name='SCADS')
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = MaskedMeanSquaredError(self.pad_value)
        metrics = MaskedMeanAbsoluteError(self.pad_value)
        model.compile(loss=loss, optimizer=opt, metrics=metrics, run_eagerly=False,
                      jit_compile=self.XLA_state)     
        if summary==True:
            model.summary(expand_nested=True)

        return model
                 

# [CUSTOM MASKED LOSS]
#==============================================================================
# collection of model and submodels
#==============================================================================
@keras.utils.register_keras_serializable(package='CustomLoss', name='MaskedMeanSquaredError')
class MaskedMeanSquaredError(keras.losses.Loss):
    def __init__(self, pad_value, reduction=keras.losses.Reduction.AUTO, name='MaskedMeanSquaredError', **kwargs):
        super(MaskedMeanSquaredError, self).__init__(reduction=reduction, name=name, **kwargs)
        self.pad_value = pad_value
        self.mse = keras.losses.MeanSquaredError(reduction=reduction)

    # implement call method 
    #--------------------------------------------------------------------------
    def call(self, y_true, y_pred):
        mask = tf.not_equal(y_true, self.pad_value)       
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)       
        loss = self.mse(y_true_masked, y_pred_masked)

        return loss
    
    # serialize loss for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(MaskedMeanSquaredError, self).get_config()        
        config.update({'pad_value': self.pad_value})
        return config

    # deserialization method  
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)

# [CUSTOM MASKED METRIC]
#==============================================================================
# collection of model and submodels
#==============================================================================    
@keras.utils.register_keras_serializable(package='CustomMetric', name='MaskedMeanAbsoluteError')
class MaskedMeanAbsoluteError(keras.metrics.Metric):
    def __init__(self, pad_value, name='MaskedMeanAbsoluteError', **kwargs):
        super(MaskedMeanAbsoluteError, self).__init__(name=name, **kwargs)
        self.pad_value = pad_value
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    # update metric status
    #--------------------------------------------------------------------------
    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.not_equal(y_true, self.pad_value)
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)
        
        error = tf.abs(y_true_masked - y_pred_masked)
        self.total.assign_add(tf.reduce_sum(error))
        self.count.assign_add(tf.cast(tf.size(y_true_masked), tf.float32))

    # results
    #--------------------------------------------------------------------------
    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    # The state of the metric will be reset at the start of each epoch
    #--------------------------------------------------------------------------
    def reset_states(self):        
        self.total.assign(0)
        self.count.assign(0)

    # serialize loss for saving  
    #--------------------------------------------------------------------------
    def get_config(self):       
        config = super(MaskedMeanAbsoluteError, self).get_config()
        config.update({'pad_value': self.pad_value})
        return config

    # deserialization method  
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):        
        return cls(**config)
    
    
# [TRAINING OPTIONS]
#==============================================================================
# Custom training operations
#==============================================================================
class ModelTraining:

    '''
    ModelTraining - A class for configuring the device and settings for model training.

    Keyword Arguments:
        device (str):                         The device to be used for training. 
                                              Should be one of ['default', 'GPU', 'CPU'].
                                              Defaults to 'default'.
        seed (int, optional):                 The seed for random initialization. Defaults to 42.
        use_mixed_precision (bool, optional): Whether to use mixed precision for improved training performance.
                                              Defaults to False.
    
    '''      
    def __init__(self, device='default', seed=42, use_mixed_precision=False):                            
        np.random.seed(seed)
        tf.random.set_seed(seed)         
        self.available_devices = tf.config.list_physical_devices()
        print('-------------------------------------------------------------------------------')        
        print('The current devices are available: ')
        print('-------------------------------------------------------------------------------')
        for dev in self.available_devices:
            print()
            print(dev)
        print()
        print('-------------------------------------------------------------------------------')
        if device == 'GPU':
            self.physical_devices = tf.config.list_physical_devices('GPU')
            if not self.physical_devices:
                print('No GPU found. Falling back to CPU')
                tf.config.set_visible_devices([], 'GPU')
            else:
                if use_mixed_precision == True:
                    policy = keras.mixed_precision.Policy('mixed_float16')
                    keras.mixed_precision.set_global_policy(policy) 
                tf.config.set_visible_devices(self.physical_devices[0], 'GPU')
                os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'                 
                print('GPU is set as active device')
            print('-------------------------------------------------------------------------------')
            print()        
        elif device == 'CPU':
            tf.config.set_visible_devices([], 'GPU')
            print('CPU is set as active device')
            print('-------------------------------------------------------------------------------')
            print()

    #--------------------------------------------------------------------------
    def model_parameters(self, parameters_dict, savepath):

        '''
        Saves the model parameters to a JSON file. The parameters are provided 
        as a dictionary and are written to a file named 'model_parameters.json' 
        in the specified directory.

        Keyword arguments:
            parameters_dict (dict): A dictionary containing the parameters to be saved.
            savepath (str): The directory path where the parameters will be saved.

        Returns:
            None       

        '''
        path = os.path.join(savepath, 'model_parameters.json')      
        with open(path, 'w') as f:
            json.dump(parameters_dict, f)   
    
    
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
        model = tf.keras.models.load_model(self.model_path, compile=True)
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
        
    
# [MODEL VALIDATION]
#============================================================================== 
# ...
#==============================================================================
class ModelValidation: 

    def __init__(self, model):

        self.model = model     
    
    # comparison of data distribution using statistical methods 
    #--------------------------------------------------------------------------     
    def visualize_predictions(self, X, Y_real, Y_predicted, name='Series', plot_path=None):       

        fig_path = os.path.join(plot_path, f'Visual_validation_{name}.jpeg')
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
        plt.show()       
        plt.close()          

        
          
    
            
