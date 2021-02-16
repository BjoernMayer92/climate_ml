import numpy as np
import xarray as xr



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
    return anomaly(data, dims=dims)/data.std(dim=dims)

    
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
    
    feature_names = []
    for name in enc.get_feature_names():
        feature_names.append(name[3:])
    
    data_transformed_xr =  xr.DataArray(data_transformed, dims=["stacked", "dimension"], coords={"stacked":stacked_coords, "dimension": feature_names})
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




def self_correlation(data , dims = ("time",), ignore_dims = ()):
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
    
    
def eof_weight_latitude(field):
    """
    Weight fields with corresponding Latitude using the sqrt such that the covariance in the calculation of the eof is     
    weighted with  cosine of latitude
    
    Parameters:
    -----------
    field: xarray object
        Input field
    
    Returns:
    --------
    field_weightes : xarray obbject
        Weighted Field
    
    """
    
    weights = xr.ufuncs.sqrt( xr.ufuncs.cos ( xr.ufuncs.deg2rad(field.lat) ) )
    
    return field * weights


def eof(data, neof=3, ensemble=False, norm=True, ddof=1):
    """
    Calculates Empirical orthogonal functions and principal components 
    
    Parameters:
    -----------
    data: xarray DataObject
        Dataobject must have at least a time dimension
        
    neof: integer
        Number of EOFs that are calculated
        
    ensemble: boolean
        If True data must have also a dimension called "ens". The dimensions "time" and "ens" will be stacked and 
        calculation of EOF is done over this stacked dimension.
    norm  boolean
        Whether the corresponding PCs are normalized or not
    ddof: integer
        No Effect yet
    
    
    Returns:
    --------
    data: xarray Dataset
        dataset containing "EOF", "PCs"
    
    solver: 
    
    """
    from eofs.xarray import Eof    
    
    
    # If ensemble == True first stack the ens and time dimension together
    if ensemble==True:
        field = stack_ensemble(data)
    if ensemble==False:
        field = data
    
    
    # preprocessing: Calculate Anomalies and Area Weight
    field_anom        = anomaly(field, dims="time")
    field_anom_weight = eof_weight_latitude(field_anom).transpose("time",...)
    
    #Calculate the Eof using xarray EOF package
    solver            = Eof(field_anom_weight)
    
    # Save pcs and eof and explained variance in variables
    pcs = solver.pcs(npcs=neof,pcscaling=0)
    eof = solver.eofs(neofs=neof)
    exp = solver.varianceFraction(neigs=neof)
    

    if ensemble==False:
        if norm==True:
            pcs=normalize(pcs, dims="time")
        data = xr.Dataset({'PCs': pcs, "EOF": eof})
            
    # if ensemble == True return the pcs after unstacking and renaming the dimension
    if ensemble==True:
        pcs_unstacked = pcs.unstack(dim="time").rename({"real_time":"time"})
        if norm==True:
            pcs_unstacked_norm = normalize(dims=("time","ens"))
        data = xr.Dataset({'PCs': pcs_unstacked_norm, "EOF": eof})
    
    return data, solver


def stack_ensemble(field):
    """
    Stacks time and ensemble dimension into a new dimension
    
    Parameters:
    -----------
    field: xarray DataObject
        Input field must have at least dimension ("time","ens")
    
    Returns:
    field_stacked: xarray DataObject
        Stacked Field with a multiindex "time" summarizing the original time dimension which has been renamed to 
        "real_time" and "ens"
    --------
    
    """

    return field.rename({"time":"real_time"}).stack(time=("real_time","ens")).transpose("time","lat","lon")


def area_weighted_mean(data, dims):
    """
    Calculates Area weighted mean of an xarray by multiplying each gridpoint with the cosine of latitude beforehand
    
    Parameters:
    -----------
    data: xarray DataArray
        Dataarray containing the values must have at least dimension "lat"
    dims: str or sequence of strings
        Dimensions over which mean is taken
    
    Returns:
    data_weighted: xarray Dataarray
        DataArray containing the weighted mean over given dimensions
    --------
    
    
    """
    
    weights = xr.ufuncs.cos(np.deg2rad(data.lat))
    weights.name = "weights"
    
    data_weighted = (data*weights).mean(dim=dims)
    
    return data_weighted
    
    
def split_year_season(data):
    """
    Splits the time dimension of an xarray dataObject into year and Season
    
    Parameters:
    -----------
    data: xarray DataObject
        Must have at least a time dimension with datetime type
        
        
    Returns:
    --------
    
    data_season: xarray DataObject
        Original data with year and month dimension instead of time
    
    xarray DataObject
        
    """
    
    season_dict = {"DJF":1,"MAM":4,"JJA":7,"SON":10}
    
    tmp_arr = []
    for season in season_dict:
        season_month = season_dict[season]
    
        tmp = data.where(data.time.dt.month==season_month,drop="True")
        tmp = tmp.assign_coords(time=tmp.time.dt.year).rename({"time":"year"}).assign_coords(season=season)
        tmp_arr.append(tmp)
    
    data_season = xr.concat(tmp_arr,dim = "season")
    
    return data_season