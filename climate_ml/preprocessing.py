import numpy as np
import xarray as xr

def sellonlatbox(data, lon_min, lon_max, lat_min, lat_max):
    """
    Selects a longitude-latitude box from a dataset
    Parameters:
    -----------
    data: xarray dataset or dataarray
        Input field, must have at least the dimension (lat,lon)
    lon_min: float
        Minimum longitude
    lon_max: float
        Maximum longitude
    lat_min: float
        Minimum latitude
    lat_max: float
        Maximum latitude
        
    Output:
    -------
    data_cropped: xarray dataset or dataarray
        Cropped Field
    """
    data_cropped = data.where( (data.lat<lat_max)&(data.lat>lat_min)&(data.lon<lon_max)&(data.lon>lon_min)  )
             
    return data_cropped

def min_max_scalar(data, dims):
    data_min = data.min(dim=dims)
    data_max = data.max(dim=dims)
    
    return (data - data_min) / (data_max -data_min)


def anomaly(data, dims={"time"}):
    """
    Calculates the anomaly with respect to the specified dimensions
    
    Parameters
    ----------
    data: xarray dataarray or dataset
        Input field over which the anomalies are calculated
    dims: dictionary of dimensions with respect to which anomalies are calculated
    Return
    ------
    data_anom: xarray dataarray or dataset
        Calculated Anomalieshttps://monitoring.dkrz.de/d/000000008/home?orgId=5&refresh=10s
    """
    
    return data-climatology(data,dims=dims)

def normalize(data, dims={"time"}):
    """
    Normalizes  with respect to the specified dimensions
    
    Parameters
    ----------
    data: xarray dataarray or dataset
        Input field over which the anomalies are calculated
    dims: dictionary of dimensions with respect to which anomalies are calculated
    Return
    ------
    data_anom: xarray dataarray or dataset
        Calculated Anomalies
    """
    return anomaly(data, dim=dims)/data.std(dim=dims)

    
def climatology(data,dims={"time"}):
    """
    Calculates Climatology with respect to the specified dimensions
    
    Parameters
    ----------
    data: xarray dataarray or dataset
        Input field for which Climatology ia about to be calculated
    dims: dictionary of dimensions with respect to which the climatology is calculated
    Return
    ------
    data_anom: xarray dataarray or dataset
        Calculated Climatology
    """
    
    return data.mean(dim=dims)
    
    
    
def sellonlatbox(data, lon_min, lon_max, lat_min, lat_max):
    """
    Selects a longitude-latitude box from a dataset
    Parameters:
    -----------
    data: xarray dataset or dataarray
        Input field, must have at least the dimension (lat,lon)
    lon_min: float
        Minimum longitude
    lon_max: float
        Maximum longitude
    lat_min: float
        Minimum latitude
    lat_max: float
        Maximum latitude
        
    Output:
    -------
    data_cropped: xarray dataset or dataarray
        Cropped Field
    """
    data_cropped = data.where( (data.lat<lat_max)&(data.lat>lat_min)&(data.lon<lon_max)&(data.lon>lon_min)  )
             
    return data_cropped



def one_hot_encoder_xarray(data, encoding_dim, drop_dims=None):
    """
    Encodes a dimension of a dataarray into a OneHot representation
    
    Parameters
    ----------
    data: xarray Dataarray
        Input Datarray
    encoding_dim: string
        Dimension of data over which the encoding is calculated
    drop_dims: string tuple
        Tuple of dimension names which should be not considered (e.g. feature dimensions)
    Return:
    data_transformed_xr : xarray Datarray
        Returns a dataarray with one additional dimension corresponding to the OneHotRepresentation
    """
    
    from sklearn.preprocessing import OneHotEncoder
    
    enc = OneHotEncoder()
    if(drop_dims):
        dictionary = dict(zip(np.array(drop_dims),np.zeros(len(drop_dims),dtype=int)))
        data = data.isel(dictionary)
    
    data_stacked = data.stack(stacked = data.dims)
    stacked_coords = data_stacked.coords["stacked"]
    
    
    data_transformed = enc.fit_transform(data_stacked.coords["stacked"].coords[encoding_dim].values.reshape(-1,1)).toarray()
    
    data_transformed_xr =  xr.DataArray(data_transformed, dims=["stacked", "dimension"], coords={"stacked":stacked_coords, "dimension": enc.get_feature_names()})
    data_transformed_xr_unstack = data_transformed_xr.unstack(dim="stacked")
    return data_transformed_xr_unstack.to_dataset(dim="variable")



    
def covariance (data_1, data_2, dims = ("time",)):
    """
    Calculates the covariance between two datasets over given common dimensions
    
    Parameters:
    -----------
    data_1: xarray object
        xarray object with at least the dimension given by dims
    data_2: xarray object
        xarray object with at least the dimension given by dims
    dims: string tuple
        tuple of dimension names over which the covariance is calculated
        
    Returns:
    --------
    cov: xarray object
        covariance between the two datasets over common dimension
    """
    
    cov = xr.cov(data_1,data_2, dim=dims)
    
    return cov

def correlation(data_1, data_2, dims= {"time"}):
    """
    Calculates correlation coefficient between two datasets over given common dimension
    
    Parameters:
    -----------
    data_1: xarray object
        xarray object with at least the dimension given by dims
    data_2: xarray object
        xarray object with at least the dimension given by dims
    dims: string tuple
        tuple of dimension names over which the covariance is calculated
        
    Returns:
    --------
    corr: xarray object
        correlation between the two datasets over common dimension
    """
    
    cov = covariance(data_1, data_2, dims = dims)
    
    
    # Calculates the size of the dimension over which the 
    data_1_std = data_1.std(dim=dims)
    data_2_std = data_2.std(dim=dims)
    
    corr = cov/(data_1_std*data_2_std)
    
    return corr




def self_correlation(data ,dims = ("time",), ignore_dims = ()):
    """
    Calculates the correlation of a data object over given dimensions, while the remaining dimensions are shifted through all possible combinations
    
    Parameters:
    -----------
    data: xarray object
        xarray object over which the self correlation should be calculated
    dims: string tuple
        tuole of dimension names over which correlation is calculated
    
    Returns:
    --------
    corr: xarray object
        correlation values with all dimension that were not in the dims tuple doubled and shifted
    """
    
    # create sets from tuples
    dims_full_set = set(data.dims)
    dims_set      = set(dims)
    
    print(dims_full_set)
    print(dims_set)
    
    #subtract the dims_set from the full set
    dims_remaining = tuple(dims_full_set.difference(set.union(dims_set,ignore_dims)))
    
    print(dims_remaining)
    
    # create dictionary for renaming dimensions
    rename_dict  = {}
    for dim in dims_remaining:
        rename_dict[dim] = dim+"_shifted"
        
    corr = correlation(data,data.rename(rename_dict), dims = dims)
    
    return corr
    
    