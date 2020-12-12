import xarray as xr
import os
import pickle
import keras
import numpy as np
import plotting

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
        os.system("mkdir " +model_path)
        config_filename = os.path.join(model_path, "config.pkl")
        
        dictionary = self.__dict__.copy()
        del dictionary["model"]

        print(dictionary)
        with open(config_filename, 'wb') as handle:
            pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        self.model.save(model_path)
    
    def load_model(self,path,name):
        model_path      = os.path.join(path,name)
        config_filename = os.path.join(model_path, "config.pkl")
        with open(config_filename, 'rb') as handle:
            dictionary = pickle.load(handle)
        
        
        self.__dict__ = dictionary
        self.model = keras.models.load_model(model_path)
    
    
 
    def define_model(self,
                     dataset,
                     n_neurons,
                     activations,
                     regularizers, 
                     optimizer,
                     output_activation,
                     loss):
        # Assign Values
        self.dataset      = dataset

        self.n_neurons    = n_neurons
        self.activations  = activations
        self.regularizers = regularizers
        
        self.optimizer = optimizer
        self.loss = loss
        
        
        # Get number of remaining features after dropping Nan
        n_feature = self.dataset.feature.size
        n_output  = self.dataset.output.size
        # Initialize  Input Layer with number of features 
        input_layer = keras.Input(shape = (n_feature,))

        # Start List of Layers
        layers = []

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
                                                 kernel_regularizer = keras.regularizers.l2(self.regularizers[i_layer]))(layers[i_layer-1]))
        
        " Initialize output layer"
        layers.append(keras.layers.Dense(n_output,activation= output_activation)(layers[-1]))

        # Combine Model
        self.model = keras.models.Model(input_layer,layers[-1])
        
    def compile(self):
        self.model.compile(optimizer = self.optimizer, loss= self.loss, metrics=['accuracy'])
        
    def fit(self,epochs, batch_size , shuffle=True, validation_split = 0.3, verbose = 2):
        self.fit_epochs           = epochs 
        self.fit_shuffle          = shuffle
        self.fit_validation_split = validation_split
        self.batch_size           = batch_size
            
            
        self.model.fit(
            self.dataset.input_data_stack.transpose("sample","feature").values,
            self.dataset.label_data_stack.transpose("sample","output").values,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=validation_split,
            verbose = verbose)
            
        
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

        intermediate_layer_model = keras.models.Model(inputs=self.model.input, outputs=self.model.layers[layer_index].output)
        intermediate_output      = intermediate_layer_model.predict(self.input)

        return intermediate_output

    
    def transform_weights(self,curvi_input = True,curvi_output=False):
        """
        Transform weights into xarray and assigns physical input dimensions for the input layer
        """
                
        weights_layers = []
        biases_layers  = []


        for i_layer, n_neuron in enumerate(self.n_neurons):
            
            tmp_weights = self.model.layers[i_layer + 1].get_weights()[0]
            tmp_biases  = self.model.layers[i_layer + 1].get_weights()[1]
                #return tmp_biases

            if (i_layer==0):
                weights_temp = xr.DataArray(tmp_weights,
                                            dims=("feature","hidden_layer_"+str(i_layer)),
                                            coords=(self.dataset.feature,np.arange(self.n_neurons[i_layer]))).unstack(dim="feature").rename("weights_layer"+str(i_layer)) 

                if(curvi_input == True):
                    weights_temp = align_coords(self.dataset.input_data.coords, weights_temp)
                    
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
        #return tmp_biases
        
        weights_temp = xr.DataArray(tmp_weights,
                                    dims = ("hidden_layer_"+str(i_layer),"output"),
                                    coords = (np.arange(self.n_neurons[i_layer]),self.dataset.output)).unstack(dim="output").rename("weights_layer_output")
         
        
        if(curvi_output == True):
                    weights_temp = align_coords(self.dataset.input_data.coords, weights_temp)
        
        
        weights_layers.append(xr.DataArray(weights_temp))
         
        biases_layers.append(xr.DataArray(tmp_biases,
                                         dims = ("output"),
                                         coords = (self.dataset.output,)).rename("bias_layer_output").unstack(dim="output"))
       
        return weights_layers, biases_layers
        weights = xr.merge(weights_layers)
        biases  = xr.merge(biases_layers)

        parameters = xr.merge([weights,biases])
        
        self.parameters = parameters
    
        
        
    def LRP(self, analyzer = "deep_taylor",softmax = True):
        #Remove the softmax layer from the model
        if(softmax):
            innvestigate_model = innvestigate.utils.model_wo_softmax(self.model)
        else:
            innvestigate_model = self.model
        #Create the "analyzer", or the object that will generate the LRP heatmaps given an input sample
        analyzer = innvestigate.create_analyzer(analyzer, innvestigate_model)
        
        #First, create some lists to hold the output
        LRP_heatmaps_all = []
        
        self.dataset.label_data_stack.transpose("sample","output").values
        #We will process all of the samples, although another good option is to only process the validation samples
        for sample_ind in range(data_stacked_anomal.sizes["sample"]):

              #Make a print statement so we know how long it is taking to process the samples
            if (sample_ind%1000 == 0) & (sample_ind > 0):
                print(sample_ind, np.nanmax)
    
                print(str(sample_ind).zfill(3),end="\r")
    
            LRP_heatmap = analyzer.analyze(data_stacked_anomal.isel(sample=[sample_ind]))
    
        LRP_heatmaps_all.append(LRP_heatmap)
        