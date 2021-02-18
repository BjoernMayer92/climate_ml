import xarray as xr
import os
import pickle
import numpy as np
import innvestigate
import datetime
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys
import pathlib
import matplotlib.pyplot as plt
import logging

from .ml_stat import *

def align_coords(data_origin_coords, data):
    n_x = data_origin_coords["lon"].sizes["x"]
    n_y = data_origin_coords["lon"].sizes["y"]
                    
    lat = data_origin_coords["lat"].assign_coords(x = range(n_x)).assign_coords(y= range(n_y))
    lon = data_origin_coords["lon"].assign_coords(x = range(n_x)).assign_coords(y= range(n_y))

                    #return(weights_temp)
    data_changed = data.reindex({"x": range(n_x),"y":range(n_y)}, fill_value=np.nan).assign_coords(lat=(("y","x"),lat.values),lon=(("y","x"),lon.values))
    return data_changed
        
    

    
    

class ml_model():
    """
    
    """
    def __init__(self):
        pass
        
    def save_model(self,path,name):
        model_path      = os.path.join(path,name)
        self.model_path = model_path
        os.system("mkdir -p " +model_path)
        config_filename = os.path.join(model_path, "config.pkl")
        
        dictionary = self.__dict__.copy()
        del dictionary["model"]
        del dictionary["dataset"]
        del dictionary["optimizer"]
        del dictionary["layers"]
        
        with open(config_filename, 'wb') as handle:
            pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        self.model.save(os.path.join(model_path,"model"))
    
    def load_model(self,path,name):
        model_path      = os.path.join(path,name)
        self.model_path = model_path
        config_filename = os.path.join(model_path, "config.pkl")
        with open(config_filename, 'rb') as handle:
            dictionary = pickle.load(handle)
        
        
        self.__dict__ = dictionary
        self.model = tf.keras.models.load_model(os.path.join(model_path,"model"))
    
 
    def define_layers(self,
                     dataset,
                     n_neurons,
                     activations,
                     regularizers, 
                     optimizer,
                     output_activation,
                     loss,
                     name = ""):
        # Assign Values
        self.dataset      = dataset

        self.n_neurons    = n_neurons
        self.activations  = activations
        self.regularizers = regularizers
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.loss = loss
        self.input_data_coords = self.dataset.input_data.coords
        
        # Get number of remaining features after dropping Nan
        self.n_feature = self.dataset.feature.size
        self.n_output  = self.dataset.output.size
        # Initialize  Input Layer with number of features 
        input_layer = tf.keras.Input(shape = (self.n_feature,))

        # Start List of Layers
        layers = []
        layers.append(input_layer)

        for i_layer, n_neuron in enumerate(self.n_neurons):
            if (i_layer==0):
                # Initialize first hidden layer
                layers.append(tf.keras.layers.Dense(n_neuron,
                                                 activation = self.activations[i_layer],
                                                 kernel_initializer='he_normal',
                                                 bias_initializer='he_normal', 
                                                 kernel_regularizer = tf.keras.regularizers.l2(self.regularizers[i_layer]))(input_layer))
            else:
                layers.append(tf.keras.layers.Dense(n_neuron,
                                                 activation = self.activations[i_layer],
                                                 kernel_initializer='he_normal',
                                                 bias_initializer='he_normal', 
                                                 kernel_regularizer = tf.keras.regularizers.l2(self.regularizers[i_layer]))(layers[i_layer]))
        
        " Initialize output layer"
        layers.append(tf.keras.layers.Dense(self.n_output,activation= output_activation)(layers[-1]))
        #self.layers = layers
        setattr(self,"layers"+name, layers)
        
    def define_model(self, custom=True, name = ""):
        
        layers = getattr(self,"layers"+name)
        
        setattr(self,"model"+name, tf.keras.models.Model(layers[0],layers[-1]))
       
            
    def compile(self, metrics = [tf.keras.metrics.categorical_accuracy], name = ""):
        
        model = getattr(self,"model"+name)
        model.compile(optimizer = self.optimizer, loss= self.loss, metrics= metrics)
        
    def fit(self,epochs, batch_size , shuffle=True, validation_split = 0.3, verbose = 2,custom_callbacks = [], callbacks = []):

        """
        Fits the model given the hyperparameters supplied to this function
        
        Parameters:
        -----------
        epochs: Integer
            Number of epochs
        
        batch_size: Integer
            Size of batches in training
        
        validation_split: Float
            Ratio of test samples to total number of samples
        
        verbose: Integer
            Determines verbose of output
        
        callbacks: Array of callback functions
            Determines which callbacks are called during training
            
        Returns:
        -------
        
        """
        
        self.fit_epochs           = epochs 
        self.fit_shuffle          = shuffle
        self.fit_validation_split = validation_split
        self.batch_size           = batch_size
        
        
        custom_callback_histories = []
        
        for callback in custom_callbacks:
            custom_callback_histories.append(callback(self.dataset.input_data_stack.values, self.dataset.label_data_stack.values, self.dataset.label_data_stack.coords))
            
        callback_histories = []
        for callback in callbacks:
            callback_histories.append(callback)
            
        
            
        if(validation_split != 0):
            x_train, x_val, y_train, y_val = train_test_split(self.dataset.input_data_stack.values,self.dataset.label_data_stack.values, test_size = self.fit_validation_split)
        
        
        callback_histories_combined = np.concatenate([custom_callback_histories, callback_histories]).tolist()
        
        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x_val,y_val),
            verbose = verbose,
            callbacks = callback_histories_combined)
        
        
        self.history = history_xr(history)
        
        # Combine callback histories into one file
        arr =[]
        for callback in callback_histories:
            arr.append(callback.data_xr)
        self.callback_data = xr.merge(arr)
        
        
        
    def check_layer_output(self, layer_index=1):
        """
        Returns the output of a given layer in a model given some input data
        Parameters
        ----------
        model: keras.model
            Model for which the output of a given layer should be calculated
        input_data: np.array
            Input Array of data in the shape of the input layer
        layer_index: int
            Index of the layer. Can not be larger than the number of layers in the model and not zero

        Returns
        -------
        intermediate_output np.array
        Output of the given layer after activation function has been applied

        """

        intermediate_layer_model = tf.keras.models.Model(inputs=self.model.input, outputs=self.model.layers[layer_index].output)
        intermediate_output      = intermediate_layer_model.predict(self.input)

        return intermediate_output

    
    def transform_weights(self,curvi_input = True,curvi_output=False):
        """
        Transform weights into xarray and assigns physical input dimensions for the input layer
        
        Parameters:
        -----------
        curvi_input: boolean
            Whether input is in a curvilinear grid or not. Determines whether coordinates need to be aligned 
        
        curvi_output: boolean
            Whether output is in a curvilinear grid or not. Determines whether coordinates need to be aligned 
            
        name: string
            Name of the model.
        
        """
                
        weights_layers = []
        biases_layers  = []

        model = getattr(self,"model")

        
        # Loop through layers
        for i_layer, n_neuron in enumerate(self.n_neurons):
            
            # Get weights and biases
            tmp_weights = model.layers[i_layer + 1].get_weights()[0]
            tmp_biases  = model.layers[i_layer + 1].get_weights()[1]
            

            if (i_layer==0):
                weights_temp = xr.DataArray(tmp_weights,
                                            dims=("feature","hidden_layer_"+str(i_layer)),
                                            coords=(self.dataset.feature,np.arange(self.n_neurons[i_layer]))).unstack(dim="feature").rename("weights_layer"+str(i_layer)) 

                if(curvi_input == True):
                    weights_temp = align_coords(self.input_data_coords, weights_temp)
                    
                weights_layers.append(weights_temp)
                biases_layers.append(xr.DataArray(tmp_biases,
                                                  dims=("hidden_layer_"+str(i_layer)),
                                                  coords=(np.arange(self.n_neurons[i_layer]),)).rename("bias_layer_"+str(i_layer)))
                
            else:
                weights_layers.append(xr.DataArray(tmp_weights,
                                                   dims   = ("hidden_layer_"+str(i_layer-1),"hidden_layer_"+str(i_layer)),
                                                   coords = (np.arange(self.n_neurons[i_layer-1]),np.arange(self.n_neurons[i_layer]))).rename("weights_layer"+str(i_layer)))
                biases_layers.append(xr.DataArray(tmp_biases,
                                                  dims   = ("hidden_layer_"+str(i_layer)),
                                                  coords = (np.arange(self.n_neurons[i_layer]),)).rename("bias_layer_"+str(i_layer)))

        ## Output Layer
        
        tmp_weights = self.model.layers[-1].get_weights()[0]
        tmp_biases  = self.model.layers[-1].get_weights()[1]
        
        
        weights_temp = xr.DataArray(tmp_weights,
                                    dims = ("hidden_layer_"+str(i_layer),"output"),
                                    coords = (np.arange(self.n_neurons[i_layer]),self.dataset.output)).unstack(dim="output").rename("weights_layer_output")
         
        
        # 
        if(curvi_output == True):
                    weights_temp = align_coords(self.input_data_coords, weights_temp)
        
        
        weights_layers.append(xr.DataArray(weights_temp))
         
        biases_layers.append(xr.DataArray(tmp_biases,
                                         dims = ("output"),
                                         coords = (self.dataset.output,)).rename("bias_layer_output").unstack(dim="output"))
       
        weights = xr.merge(weights_layers)
        biases  = xr.merge(biases_layers)

        parameters = xr.merge([weights,biases])
        
        
        self.parameters = parameters
        
    
        
        
    def LRP(self, input_data_stack, analyzer = "deep_taylor",softmax = True):
        """
        Calculates Layer-Wise-Relevance Propagation
        """
        # 
        import keras
        weights_path = os.path.join(self.model_path,"weights.h5")
        self.model.save_weights(weights_path)
        print("weights saved")
        
        # rebuild Model in keras:
        
        
        # Initialize  Input Layer with number of features 
        input_layer = keras.Input(shape = (self.n_feature,))

        # Start List of Layers
        layers = []
        layers.append(input_layer)

        for i_layer, n_neuron in enumerate(self.n_neurons):
            if (i_layer==0):
                # Initialize first hidden layer
                layers.append(keras.layers.Dense(n_neuron,
                                                 activation = self.activations[i_layer],
                                                 kernel_initializer='he_normal',
                                                 bias_initializer='he_normal', 
                                                 kernel_regularizer = keras.regularizers.l2(self.regularizers[i_layer]))(input_layer))
            else:
                layers.append(keras.layers.Dense(n_neuron,
                                                 activation = self.activations[i_layer],
                                                 kernel_initializer='he_normal',
                                                 bias_initializer='he_normal', 
                                                 kernel_regularizer = keras.regularizers.l2(self.regularizers[i_layer]))(layers[i_layer]))
        
        " Initialize output layer"
        layers.append(keras.layers.Dense(self.n_output,activation= self.output_activation)(layers[-1]))
        layers = layers
        
        model = keras.models.Model(layers[0],layers[-1])
        model.load_weights(weights_path)
        
      
        
        
        
        
        
        
        
        
        
        #Remove the softmax layer from the model if the last layer is a softmax layer
        if(softmax):
            innvestigate_model = innvestigate.utils.model_wo_softmax(model)
        else:
            innvestigate_model = model
            
        
        #Create the "analyzer", or the object that will generate the LRP heatmaps given an input sample
        analyzer = innvestigate.create_analyzer(analyzer, innvestigate_model)
        
        #Create empty array to store all heatmaps
        LRP_heatmaps_all = []
        
        input_stacked = input_data_stack.transpose("sample","feature")
        
        LRP_dir = os.path.join(self.model_path,"LRP")
        os.system("mkdir -p "+ LRP_dir)
        
        #We will process all of the samples, although another good option is to only process the validation samples
        for sample_ind in range(input_stacked.sizes["sample"]):

            #Make a print statement so we know how long it is taking to process the samples
            if (sample_ind%100 == 0) & (sample_ind > 0):
                #print(sample_ind, np.nanmax)
                print(str(sample_ind).zfill(3),end="\r")
    
            LRP_heatmap = analyzer.analyze(input_stacked.isel(sample=[sample_ind]))
            
            sample = input_stacked.isel(sample=[sample_ind]).coords["sample"]#.isel(sample = sample_ind)
            #return sample
            #return np.array(LRP_heatmap)
            LRP_heatmap_da = xr.DataArray(np.array(LRP_heatmap), 
                                          dims=("sample","feature"),
                                          coords=({"sample":sample,"feature": input_stacked.coords["feature"]})).unstack(dim="feature").unstack(dim="sample")
            
            LRP_heatmap_da.to_netcdf(os.path.join(LRP_dir,"sample_"+str(sample_ind)+".nc"))
            
        LRP_heatmap_da = xr.open_mfdataset(os.path.join(LRP_dir,"sample_*.nc"), use_cftime=True)
        
        LRP_heatmap_da_coords = align_coords(self.input_data_coords,LRP_heatmap_da)
        
        heatmaps_filename = os.path.join(LRP_dir,"heatmaps.nc")
        LRP_heatmap_da_coords.to_netcdf(heatmaps_filename)
        
        os.system("rm "+os.path.join(LRP_dir,"sample_*.nc"))
        
        self.heatmaps_filename = heatmaps_filename 
        def bias_variance_decomposition_reg(self,regularizers, epochs, batch_size, custom_callbacks, callbacks, train_test_split = 0.3, n_kfold = 2, N=5, seed = None):
            """
            #Calculates the bias variance composition for the given network

            """
            import time


            # Split Train Test 
            x_train, x_test, y_train, y_test = kfold_train_test_split(input_data = self.dataset.input_data_stack,
                                                                      label_data = self.dataset.label_data_stack,
                                                                      train_test_split = train_test_split,
                                                                      seed = seed,
                                                                      n_kfold = n_kfold,
                                                                      N=N)


            n_kfold = x_train.sizes["kfold"]

            data_arr = []
            pred_arr = []
            weig_arr = []
            hist_arr = []
            call_arr = []


            for regularizer_index in regularizers.regularizer_index:
                start = time.time()

                pred = []
                weig = []
                hist = []
                call = []

                for kfold in range(n_kfold):

                    # Compiles the model again to reinitiaize the weights
                    self.define_layers(self.dataset,
                                       self.n_neurons,
                                       self.activations,
                                       regularizers.isel(regularizer_index = regularizer_index).values,
                                       self.optimizer, 
                                       self.output_activation,
                                       self.loss,
                                       name = "_kfold")

                    # Defines the model with standard parameters
                    self.define_model(name = "_kfold")

                    # Compiles the model
                    self.compile(name = "_kfold")

                    # Select the training set
                    x_train_tmp = x_train.isel(kfold=kfold).dropna(dim="sample")
                    y_train_tmp = y_train.isel(kfold=kfold).dropna(dim="sample")



                    # Prepare Callbacks
                    custom_callback_histories = []
                    callback_histories = []

                    # Start the callbacks
                    for callback in custom_callbacks:
                        custom_callback_histories.append(callback(x_train_tmp.values,
                                                           y_train_tmp.values,
                                                           y_train_tmp.coords))
                    for callback in callbacks:
                        callback_histories.append(callback)



                    # Fit the model
                    callback_histories_combined = np.concatenate([custom_callback_histories, callback_histories]).tolist()
                    history = self.model_kfold.fit(x_train_tmp.values,
                                                   y_train_tmp.values,
                                                   epochs=epochs,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   validation_data=(x_test.values,y_test.values),
                                                   verbose = 0,
                                                   callbacks = callback_histories_combined)

                    # Tramsform history back

                    hist_tmp = transform_history_xarray(history)

                    # Get predicitons on test set
                    pred_tmp = self.model_kfold.predict(x_test.values)
                    pred_tmp = xr.DataArray(pred_tmp, dims = ["sample","output"], coords = {"sample":x_test.coords["sample"],"output": y_test.coords["output"] })

                    # Get weights
                    weig_tmp = self.transform_weights(self,name = "_kfold")

                    # Get cusom callback data
                    tmp = []
                    for callback in custom_callback_histories:
                        tmp.append(callback.data_xr)
                    call_tmp = xr.merge(tmp)

                    # Gather data into arrays
                    pred.append(pred_tmp.assign_coords(kfold=kfold))
                    weig.append(weig_tmp.assign_coords(kfold=kfold))
                    hist.append(hist_tmp.data.assign_coords(kfold=kfold))
                    call.append(call_tmp.assign_coords(kfold=kfold))



                # combine predictions
                pred_kfold = xr.concat(pred, dim = "kfold", coords = {"kfold" : x_train.kfold}).assign_coords(regularizer_index = regularizer_index)
                weig_kfold = xr.concat(weig, dim = "kfold", coords = {"kfold" : x_train.kfold}).assign_coords(regularizer_index = regularizer_index)
                hist_kfold = xr.concat(hist, dim = "kfold", coords = {"kfold" : x_train.kfold}).assign_coords(regularizer_index = regularizer_index)

                # Average the predictions
                predictions_average_pdf = kfold_predictions_average_pdf(pred_kfold, category_dim="output")

                # Calculate the crossentropy
                cce  = entropy(y_test, pred_kfold, category_dim="output").mean(dim="kfold").rename("cross_entropy")

                # Calculate Bias and Variances

                bias = kullback_leibler_divergence(y_test, predictions_average_pdf,     category_dim="output").rename("bias")
                vari = kullback_leibler_divergence(predictions_average_pdf, pred_kfold, category_dim = "output").mean(dim="kfold").rename("vari")

                data_kfold = xr.merge([cce,bias,vari])


                layer_names = []
                for hidden in range(regularizers.sizes["layer"]):
                    name = "hidden_layer_"+str(hidden)
                    data_kfold = data_kfold.assign_coords({"hidden_layer_"+str(hidden) : regularizers.isel(layer = hidden).isel(regularizer_index=regularizer_index)})
                    layer_names.append(name)

                pred_arr.append(pred_kfold)
                weig_arr.append(weig_kfold)
                data_arr.append(data_kfold)
                hist_arr.append(hist_kfold)

                print("regularization term "+str(regularizer_index)+"/"+str(len(regularizers.regularizer_index))+" done  in "+ str(time.time()-start) +" s")

            return_pred = xr.concat(pred_arr, dim="regularizer_index")
            return_weig = xr.concat(weig_arr, dim="regularizer_index")
            return_data = xr.concat(data_arr, dim="regularizer_index") 
            return_hist = xr.concat(hist_arr, dim="regularizer_index")
            #return_call = xr.concat(call_arr, dim="regularizer_index")

            return return_data, return_weig, return_pred, return_hist, x_train, x_test, y_train, y_test 
    

        
        
                
class bias_variance_decomposition_test():
    def __init__(self, ml_model):
        logging.info("create Instance of bias variance decomposition")
        self.parent_model = ml_model
        
    def copy_parameters(self):
        self.dataset           = self.parent_model.dataset
        self.n_neurons         = self.parent_model.n_neurons
        self.activations       = self.parent_model.activations
        self.output_activation = self.parent_model.output_activation
        self.optimizer         = self.parent_model.optimizer
        self.loss              = self.parent_model.loss
        self.input_data_coords = self.parent_model.dataset.input_data.coords
        
    def define_models(self, regularizers):
        
        self.regularizers = regularizers
        
        data_arr = []
        pred_arr = []
        weig_arr = []
        hist_arr = []
        call_arr = []
        
        mlmo_arr = []
        
        for regularizer_index in self.regularizers.regularizer_index:
            #start = time.time()
            
            
            pred = []
            weig = []
            hist = []
            call = []
            mlmo = []
            
            regularizer = self.regularizers.isel(regularizer_index=regularizer_index).values
            
            for kfold in range(self.n):
                
                model = ml_model()
                model.define_layers(dataset           = self.dataset,
                                    n_neurons         = self.n_neurons,
                                    activations       = self.activations,
                                    regularizers      = regularizer,
                                    optimizer         = self.optimizer,
                                    output_activation = self.output_activation,
                                    loss              = self.loss)
                model.define_model()
                model.compile()
                mlmo.append(model)
            mlmo_arr.append(mlmo)
            
        self.models = mlmo_arr
        return mlmo_arr
    
    
    def train_test_split(self, n_kfold = 2, train_test_split=0.2, N=5, seed = None):
        """
        
        
        """
        
        x_train, x_test, y_train, y_test = kfold_train_test_split(input_data = self.dataset.input_data_stack,
                                                                  label_data = self.dataset.label_data_stack,
                                                                  train_test_split = train_test_split,
                                                                  seed = seed,
                                                                  n_kfold = n_kfold,
                                                                  N=N)
        
        self.x_train = x_train
        self.y_train = y_train
        
        self.y_test  = y_test
        self.x_test  = x_test
        
        self.n = len(x_train.kfold)
        
    
    def train_models(self,callbacks = [], custom_callbacks =[],epochs=200, batch_size=32, learning_rate =0.01):
        
        data_arr = []
        pred_arr = []
        weig_arr = []
        hist_arr = []
        call_arr = []
        
        mlmo_arr = []
        
        for regularizer_index in self.regularizers.regularizer_index.values:
            #start = time.time()
            
            
            pred = []
            weig = []
            hist = []
            call = []
            mlmo = []
            
            regularizer = self.regularizers.isel(regularizer_index=regularizer_index).values
            
            for kfold in range(self.n):
                        
                    # select the model
                    model = self.models[regularizer_index][kfold]
                    
                    # Select the training set
                    x_train_tmp = self.x_train.isel(kfold=kfold).dropna(dim="sample")
                    y_train_tmp = self.y_train.isel(kfold=kfold).dropna(dim="sample")


                    # Prepare Callbacks
                    custom_callback_histories = []
                    callback_histories = []

                    # Start the callbacks
                    for callback in custom_callbacks:
                        custom_callback_histories.append(callback(x_train_tmp.values,
                                                           y_train_tmp.values,
                                                           y_train_tmp.coords))
                    for callback in callbacks:
                        callback_histories.append(callback)



                    # Fit the model
                    callback_histories_combined = np.concatenate([custom_callback_histories, callback_histories]).tolist()
                    
                    history = model.model.fit(x_train_tmp.values,
                              y_train_tmp.values,
                              epochs=epochs,
                              batch_size=batch_size,
                              shuffle=True,
                              validation_data=(self.x_test.values,self.y_test.values),
                              verbose = 1,
                              callbacks = callback_histories_combined)

                    hist_tmp = history_xr(history)
                    #hist_tmp = model.history
                    
                    # Get predicitons on test set
                    pred_tmp = model.model.predict(self.x_test.values)
                    pred_tmp = xr.DataArray(pred_tmp, dims = ["sample","output"], coords = {"sample":self.x_test.coords["sample"],"output": self.y_test.coords["output"] })

                    # Get weights
                    model.transform_weights(curvi_input=False)
                    weig_tmp = model.parameters
                    
                    # Get cusom callback data
                    tmp = []
                    for callback in custom_callback_histories:
                        tmp.append(callback.data_xr)
                    call_tmp = xr.merge(tmp)

                    # Gather data into arrays
                    pred.append(pred_tmp.assign_coords(kfold=kfold))
                    weig.append(weig_tmp.assign_coords(kfold=kfold))
                    hist.append(hist_tmp.data.assign_coords(kfold=kfold))
                    call.append(call_tmp.assign_coords(kfold=kfold))
                
            # combine predictions
            pred_kfold = xr.concat(pred, dim = "kfold", coords = {"kfold" : self.x_train.kfold}).assign_coords(regularizer_index = regularizer_index)
            weig_kfold = xr.concat(weig, dim = "kfold", coords = {"kfold" : self.x_train.kfold}).assign_coords(regularizer_index = regularizer_index)
            hist_kfold = xr.concat(hist, dim = "kfold", coords = {"kfold" : self.x_train.kfold}).assign_coords(regularizer_index = regularizer_index)
            call_kfold = xr.concat(call, dim = "kfold", coords = {"kfold" : self.x_train.kfold}).assign_coords(regularizer_index = regularizer_index)
                
            # Average the predictions
            predictions_average_pdf = kfold_predictions_average_pdf(pred_kfold, category_dim="output")

            # Calculate the crossentropy
            cce  = entropy(self.y_test, pred_kfold, category_dim="output").mean(dim="kfold").rename("cross_entropy")

            # Calculate Bias and Variances

            bias = kullback_leibler_divergence(self.y_test,             predictions_average_pdf,     category_dim="output").rename("bias")
            vari = kullback_leibler_divergence(predictions_average_pdf, pred_kfold,                  category_dim = "output").mean(dim="kfold").rename("vari")

            data_kfold = xr.merge([cce,bias,vari]).assign_coords(regularizer_index = regularizer_index)


            #layer_names = []
            #for hidden in range(self.regularizers.sizes["layer"]):
            #    name = "hidden_layer_"+str(hidden)
            #    data_kfold = data_kfold.assign_coords({"hidden_layer_"+str(hidden) : regularizers.isel(layer = hidden).isel(regularizer_index=regularizer_index)})
            #    layer_names.append(name)

            pred_arr.append(pred_kfold)
            weig_arr.append(weig_kfold)
            data_arr.append(data_kfold)
            hist_arr.append(hist_kfold)

        return_pred = xr.concat(pred_arr, dim="regularizer_index")
        return_weig = xr.concat(weig_arr, dim="regularizer_index")
        return_data = xr.concat(data_arr, dim="regularizer_index") 
        return_hist = xr.concat(hist_arr, dim="regularizer_index")
        return_call = xr.concat(call_arr, dim="regularizer_index")
            
        print("regularization term "+str(regularizer_index)+"/"+str(len(regularizers.regularizer_index))+" done  in "+ str(time.time()-start) +" s")
        
        return return_data, return_weig, return_pred, return_hist, return_call, 
                
class history_xr():
    """
    Class that takes in keras history objects and saves them in an xarray
    """
    
    
    def __init__(self,history):
        self.data = self.transform_history_xarray(history)
        
        
    def transform_history_xarray(self, history):
        
        N_epoch = history.params["epochs"]
        
        tmp  = []
        for key in history.history.keys():
            tmp.append( xr.DataArray(history.history[key], dims = "epoch", coords = {"epoch": range(N_epoch)}).rename(key))
            
        data = xr.merge(tmp)
            
        return data.assign_coords(history.params)
    
    
    def plot(self, ax= None, metrics=["loss","categorical_accuracy"], colors = ["Red","Blue"], linestyles = ["-","--"]):
        """
        Plots two metrics for validation and loss
        
        Right now works only with two metrics. More metrics might be implemented later with more y axis
        
        Parameters:
        -----------
        ax: matplotlib.axes
            Axes where plot should be plotted. If None function creates automatically new figure
        metrics: list of strings
            Metrics that are plotted
        colors: list of strings
            Colors for each metric
        linestyles: list of strings
            linestyles for training and validation
            
        Returns:
        --------
        
        """
        
        if ax ==None:
            fig, ax1 = plt.subplots(1,1,figsize=(10,10))
        else:
            ax1 = ax
            
        ax2 = ax1.twinx()
        
        ax =[ax1,ax2]
        ax[0].set_xlabel("epoch")
        
        for i, metric in enumerate(metrics):
            ax[i].plot(self.data[        metric], color = colors[i], linestyle = linestyles[0], label = "training" )
            ax[i].plot(self.data["val_"+ metric], color = colors[i], linestyle = linestyles[1], label = "validation")
            ax[i].set_ylabel(metric)
            
        ax[0].legend()
        
        
        
       
