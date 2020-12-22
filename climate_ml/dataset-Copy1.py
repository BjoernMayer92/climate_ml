import xarray as xr
import pickle


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
        
        self.input_data_filename = raw_input_filename
        self.label_data_filename = raw_label_filename
        
        self.feature_dims = feature_dims
        self.sample_dims  = sample_dims
        self.output_dims  = output_dims
        
        self.var_input_names = var_input_names
        self.var_label_names = var_label_names
        
        self.excluded_keys = []
    
    def initialize(self, chunks_dict = {"time":1}):
        self.input_data = xr.open_dataset(self.input_data_filename, use_cftime=True, chunks = self.chunks_dict)[self.var_input_names].to_array()
        self.label_data = xr.open_dataset(self.label_data_filename, use_cftime=True, chunks = self.chunks_dict)[self.var_label_names].to_array()
    
        self.input_data_coords  = self.input_data.coords  
        self.label_data_coords  = self.label_data.coords
        
        self.excluded_keys = self.excluded_keys.append("input_data")
        self.excluded_keys = self.excluded_keys.append("label_data")
        
    def sel_input(self, **kwargs):
        self.input_data = self.input_data.sel(kwargs)
        self.sel_input = kwargs
        
    def sel_label(self, **kwargs):
        self.label_data = self.label_data.sel(kwargs)
        self.sel_label = kwargs
        
    def stack_dimensions(self):
        self.input_data_stack = 
        self.input_data.stack(feature = self.feature_dims).dropna(dim="feature").stack(sample = self.sample_dims)
        
        if(self.output_dims):
            self.label_data_stack = 
            self.label_data.stack(sample  = self.sample_dims).stack(output=self.output_dims).dropna(dim="output")
        else:
            self.label_data_stack = self.label_data.stack(sample  = self.sample_dims).expand_dims(dim="output")
        self.feature  = self.input_data_stack.feature
        self.sample   = self.label_data_stack.sample
        self.output   = self.label_data_stack.output
        
        
        self.excluded_keys = self.excluded.append("input_data_stack")
        self.excluded_keys = self.excluded.append("label_data_stack")
    
    
    def save(self, filename):
        self.filename = filename
        #config_filename = os.path.join(filename)
        dictionary = self.__dict__.copy()
        
        for key in self.excluded_keys:
            if key in dictionary:
                del dictionary[key]
       
        with open(filename, 'wb') as handle:
            pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    def init_input(self):
        
        self.input_data = 
        xr.open_dataset(self.input_data_filename, use_cftime=True, chunks = self.chunks_dict)[self.var_input_names].to_array()
        self.label_data = 
        xr.open_dataset(self.label_data_filename, use_cftime=True, chunks = self.chunks_dict)[self.var_label_names].to_array()
          
        
        
        
    def load(self, filename):
        with open(filename, 'rb') as handle:
            dictionary = pickle.load(handle)
        
        self.__dict__ = dictionary
    
    
    def calc_training_data(self):
        self.input_data_stack = self.input_data_stack.compute()
        self.label_data_stack = self.label_data_stack.compute()
        
       
