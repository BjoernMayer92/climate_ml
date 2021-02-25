import numpy as np
import xarray as xr
import logging
logging.basicConfig(format='%(asctime)s %(message)s')


def kfold_predictions_average_pdf(predictions, category_dim = "output"):
    """
    Calculates the average of multiple model predictions over a given category dimension
    
    
    Parameters:
    -----------
    
    predictions: Xarray DataArray
        DataArray of the predictions must have at least the dimension corresponding to category_dim and a dimension "kfold"
        
    category_dim: String
        Name of the dimension corresponding to the category of the prediciton problem
        
    Returns:
    --------
    result: Xarray DataArray
        Mean prediction over kfold dimension.
    
    """
    
    prob_logs          = xr.ufuncs.log(predictions,where=(predictions!=0))
    prob_log_kmean     = prob_logs.mean(dim="kfold")
    prob_log_kmean_exp = xr.ufuncs.exp(prob_log_kmean)
    
    
    return  prob_log_kmean_exp/prob_log_kmean_exp.sum(dim=category_dim)
    

def block_train_test_split(input_data, label_data, blocksize, train_test_split=0.2, seed = None):
    """
    Creates a block train test_split
    
    
    """
    
    
    RandomState = np.random.RandomState(seed = seed)
        
    input_data_coords = input_data.coords["sample"]
    label_data_coords = label_data.coords["sample"]
    
    index_old = 0
    input_arr = []
    label_arr = []
    
    Nsample = input_data.sizes["sample"]
    
    
    for i,index in enumerate(np.arange(0, Nsample, blocksize)[1:]):
        indices = range(index_old,index)
        
        input_arr.append(input_data_coords.isel(sample=indices).assign_coords(block=i))
        label_arr.append(label_data_coords.isel(sample=indices).assign_coords(block=i))
        
        index_old = index
    
    # make the rest that didnt fit into a block to the last block    
    if index<Nsample-1:
        indices = range(index,Nsample)
        input_arr.append(input_data_coords.isel(sample=indices).assign_coords(block=i+1))
        label_arr.append(label_data_coords.isel(sample=indices).assign_coords(block=i+1))
        
    input_blocked = xr.concat(input_arr, dim="block")
    label_blocked = xr.concat(label_arr, dim="block")
    
    Nblock = input_blocked.sizes["block"]
    
    
    split_index = int(Nblock*(1-train_test_split))
    
    permutation = np.random.permutation(range(Nblock))
    
    
    train_block_indices = permutation[range(0,split_index)]
    test_block_indices  = permutation[range(split_index,Nblock)]
    
    train_coords = input_blocked.isel(block=train_block_indices[0]).dropna(dim="sample")
    for i in range(1,len(train_block_indices)):
        train_coords = train_coords.combine_first(input_blocked.isel(block=train_block_indices[i]).dropna(dim="sample"))
        
    test_coords = input_blocked.isel(block=test_block_indices[0]).dropna(dim="sample")
    for i in range(1,len(test_block_indices)):
        test_coords = test_coords.combine_first(input_blocked.isel(block=test_block_indices[i]).dropna(dim="sample"))
    
    
    train_x = input_data.sel(sample = train_coords)
    train_y = label_data.sel(sample = train_coords)
    test_x  = input_data.sel(sample = test_coords)
    test_y  = label_data.sel(sample = test_coords)

    return train_x, test_x, train_y, test_y
        
    
    
    
    
    


def kfold_train_test_split(input_data, label_data, train_test_split = 0.3, seed = None, n_kfold= 5, N=5, initial_shuffle = True):
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
        
    initial_shuffle: boolean
        Determines whether the dataset is shuffled before the training/test split or not
    
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
    
    # Create a random permutation of the data in the set if initial_shuffle is True. If not use a normal range as the first permutation, i.e. the dataset is not shuffled before the training test_split#

    RandomState = np.random.RandomState(seed=seed)
    
    if(initial_shuffle == True):
        permutation = RandomState.permutation(input_data_length)
    else: 
        permuation = np.range(input_data_length)
        
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
        permutation = RandomState.permutation(length_train)
        
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
        

    
    
    if ((n_kfold==1)&(N==1)):
        return train_x_split.isel(kfold=0), test_x, train_y_split.isel(kfold=0), test_y
    else:
        return train_x_split, test_x, train_y_split, test_y

def entropy(x,y, category_dim):
    """
    Calculates the entropy of two dataarray x and y over a common dimension
    
    If y = 0 at some point the entropy is assumed to be zero
    
    Parameters:
    ----------
    x: xarray DataArray
        First DataArray must have at least dimension category_dim
    y; xarray DataArray
        First DataArray must have at least dimension category_dim
    
    Returns:
    --------
    entropy : xarray DataArray
        DataArray containing all the dimensions of the orgiginal DataArray except category_dim
    """
    
    entropy = -xr.dot(x,xr.ufuncs.log(y,where=(y!=0)), dims = category_dim)
    
    return entropy


def kullback_leibler_divergence(x,y, category_dim):
    """
    Calculates the Kullback Leibler Divergence of two dataarrays over a given category dimension
    
    
    Parameters:
    -----------
    x: xarray DataArray
        First DataArray must have at least dimension category_dim
    y; xarray DataArray
        First DataArray must have at least dimension category_dim
    
    Returns:
    --------
    KL: xarray DataArray
        Kullback Leiber Divergence containing all the dimensions of the original dataarray except category_dim
    
    """
    
    KL = -entropy(x,x, category_dim = category_dim)+entropy(x,y, category_dim = category_dim)
    
    return KL



def bias_variance_decomposition(label, pred, category_dim = "output"):
    """
    
    
    """
    
    pred_avg = kfold_predictions_average_pdf(pred, category_dim = category_dim)
    
    cce = entropy(label, pred).mean(dim="kfold").rename("cross_entropy")
    bia = kullback_leibler_divergence(label,    pred_avg, category_dim = category_dim).rename("bias")
    var = kullback_leibler_divergence(pred_avg, pred,     category_dim = category_dim).mean(dim="kfold").rename("variance")
    
    xr.merge([cce,bias,vari])