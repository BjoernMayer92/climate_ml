import xarray as xr
import pickle
import logging

class ml_dataset():
    """
    Machine learning dataset saving all the important parameters for transforming a physical xarray dataset into a netcdf file with
    features, sample and output as dimension
    """
    
    def __init__(self, 
                 raw_input_filename,
                 raw_label_filename,
                 var_input_names,
                 var_label_names,
                 feature_dims,
                 sample_dims,
                 output_dims= None):
        
        """
        Parameters:
        ----------
        
        raw_input_filename: str
            Location of the input dataset
        raw_label_filename: str 
            Location of the label dataset
        var_input_names: array strings
            All variable names that are used as for the input. Example : ["density","temperature"]
        var_label_names: array string
            All variable names that are to be predicted. Example: ["pcs",]
        feature_dims: str tuple
            All dimensions of the input array that are combined to features
        sample_dims: str tuple
            All dimensions of input and label that are combined into a sample dimension
        
        
        
        """
        logging.info("Start Function")

        self.input_data_filename = raw_input_filename
        self.label_data_filename = raw_label_filename
        
        self.feature_dims = feature_dims
        self.sample_dims  = sample_dims
        self.output_dims  = output_dims
        
        self.var_input_names = var_input_names
        self.var_label_names = var_label_names
        
        self.excluded_keys = []
        logging.info("End Function")

        
    
    def initialize(self, chunks_dict = {"time":1}):
        """
        Opens the data
        
        Parameters:
        chunks_dict: dictionary
            Dictionary corrersponding to the chunk dimensions
        
        
        """
        logging.info("Start Function")

        
        
        self.chunks_dict = chunks_dict
        self.input_data = xr.open_dataset(self.input_data_filename, use_cftime=True, chunks = self.chunks_dict)[self.var_input_names].to_array(dim="variable")
        self.label_data = xr.open_dataset(self.label_data_filename, use_cftime=True, chunks = self.chunks_dict)[self.var_label_names].to_array(dim="variable")
        
        # Save the coordinates
        self.input_data_coords  = self.input_data.coords  
        self.label_data_coords  = self.label_data.coords
        
        self.excluded_keys.append("input_data")
        self.excluded_keys.append("label_data")
        logging.info("End Function")
        
    def sel_input(self, **kwargs):
        logging.info("Start Function")

        self.initialize()
        self.input_data = self.input_data.sel(kwargs)
        self.sel_input_kwargs = kwargs
        self.input_data_coords = self.input_data.coords
        logging.info("End Function")
        
    def isel_input(self, **kwargs):
        logging.info("Start Function")

        self.initialize()
        self.input_data = self.input_data.isel(kwargs)
        self.isel_input_kwargs = kwargs
        self.input_data_coords = self.input_data.coords
        logging.info("End Function")
        
    def sel_label(self, **kwargs):
        logging.info("Start Function")

        self.initialize()
        self.label_data = self.label_data.sel(kwargs)
        self.sel_label_kwargs = kwargs
        self.label_data_coords = self.label_data.coords
        logging.info("End Function")
        
    def stack_dimensions(self):
        logging.info("Start Function")
        self.input_data_stack = self.input_data.stack(feature = self.feature_dims).dropna(dim="feature").stack(sample = self.sample_dims).transpose("sample","feature")
        
        if(self.output_dims):
            self.label_data_stack = self.label_data.stack(sample  = self.sample_dims).stack(output=self.output_dims).dropna(dim="output").transpose("sample","output")
        else:
            self.label_data_stack = self.label_data.stack(sample  = self.sample_dims).expand_dims(dim="output").transpose("sample","output")
        self.feature  = self.input_data_stack.feature
        self.sample   = self.label_data_stack.sample
        self.output   = self.label_data_stack.output
        
        self.excluded_keys.append("input_data_stack")
        self.excluded_keys.append("label_data_stack")
        logging.info("End Function")
    
    def save(self, filename):
        logging.info("Start Function")
        self.filename = filename
        #config_filename = os.path.join(filename)
        dictionary = self.__dict__.copy()
        
        for key in self.excluded_keys:
            if key in dictionary:
                del dictionary[key]
       
        with open(filename, 'wb') as handle:
            pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info("End Function")
        
        
    def init_input(self):
        logging.info("Start Function")
        self.input_data = xr.open_dataset(self.input_data_filename, use_cftime=True, chunks = self.chunks_dict)[self.var_input_names].to_array()
        self.label_data = xr.open_dataset(self.label_data_filename, use_cftime=True, chunks = self.chunks_dict)[self.var_label_names].to_array()
        logging.info("End Function")
        
        
        
    def load(self, filename):
        logging.info("Start Function")
        with open(filename, 'rb') as handle:
            dictionary = pickle.load(handle)
        
        self.__dict__ = dictionary
        logging.info("End Function")
    
    def calc_training_data(self):
        logging.info("Start Function")
        self.input_data_stack = self.input_data_stack.compute()
        self.label_data_stack = self.label_data_stack.compute()
        logging.info("End Function")
       
