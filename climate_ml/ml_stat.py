import numpy as np
import xarray as xr
import logging
logging.basicConfig(format='%(asctime)s %(message)s')


def kfold_train_test_split(input_data, label_data, train_test_split = 0.3, seed = None, n_kfold= 5, N=5):
    """
    Splits a given input and label variable in test dataset as well as kfold sample
    
    Parameters:
    -----------
    
    input_data: xarray dataarray object
        Input data in the form (sample,feature)
        
    label_data: xarray dataarray object
        Label data in the form (sample, output)
        
    train_test_split: float
        Defining the ratio of test to overall data
        
    seed: int
        Integer defining the seed to ensure reproducibility
        
    n_kofld: int
        Number of Splits of the training set each permutation
    
    N: int
        Number of permutations of the dataset
        
    
    Returns:
    --------
    
    train_x_split: xarray dataarray object
        Input data for training in the form (sample, feature, kfold)
    
    test_x: xarray dataarray object
        Input data for testing in the form (sample, feature)
        
    train_y_split: xarray dataarray object
        Label data for training in the form (sample, output, kfold)
    
    test_y: xarray dataarray object
        Label data for testing in the form (sample,output)
    
    
    """
    
    input_data_length = input_data.sizes["sample"]
    
    # Create a random permutation of the data in the set
    permutation = np.random.RandomState(seed=None).permutation(input_data_length)
    
    # Calculate index at which the dataset is split into training and testing
    split_index = int(input_data_length*(1-train_test_split))
    
    # Split the dataarrays
    train_x = input_data.isel(sample = permutation[:split_index])
    test_x  = input_data.isel(sample = permutation[split_index:])
    
    train_y = label_data.isel(sample = permutation[:split_index])
    test_y  = label_data.isel(sample = permutation[split_index:])
    
    # Calculate the length of indices in each kfold split
    kfold_length  = int(split_index/n_kfold)
    
    # Initialize Empty arrays for training values
    nfold_x = []
    nfold_y = []
    
    # Initialize empty array of permutations
    permutations = []
    
    # Get the length of the training set
    length_train = train_x.sizes["sample"]
    
    
    # Loop through permutation number
    for n in range(N):
        
        # Create random permuation of the data in the training set
        permutation = np.random.permutation(length_train)
        
        # Select the permutations
        train_x_shuffled = train_x.isel(sample = permutation)
        train_y_shuffled = train_y.isel(sample = permutation)
        
        # Initialize empty array of kfold split
        kfold_x = []
        kfold_y = []
   
        # Loop through kfolds:
        for k in range(n_kfold):
            
            # Select subsets of training samples
            kfold_x.append(train_x.isel(sample = np.arange(kfold_length*k,kfold_length*(k+1))).assign_coords(k=k))
            kfold_y.append(train_y.isel(sample = np.arange(kfold_length*k,kfold_length*(k+1))).assign_coords(k=k))

        # Combine kfolds
        kfold_x_combined = xr.concat(kfold_x, dim ="k")
        kfold_y_combined = xr.concat(kfold_y, dim ="k")
        
        # Add kfolds to nfolds
        nfold_x.append(kfold_x_combined.assign_coords(N=n))
        nfold_y.append(kfold_y_combined.assign_coords(N=n))
        
    # combine nfolds
    nfold_x_combined = xr.concat(nfold_x, dim = "N")
    nfold_y_combined = xr.concat(nfold_y, dim = "N")
    
    train_x_split = nfold_x_combined.stack(kfold = ("N","k"))
    train_y_split = nfold_y_combined.stack(kfold = ("N","k"))
    
    return train_x_split, test_x, train_y_split, test_y