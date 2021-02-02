import xarray as xr


def calculate_confusion_matrix(prediction, true_dim = "variable", pred_dim = "dimension", samp_dim = "time"):
    """
    Calculates the confusion matrix for a given prediction_matrix
    
    Parameters:
    -----------
    prediction: xarray Dataarray
        Dataarray containing categorical predictions 
    true_dim: string
        Name of the dimension which corresponds to the true values
    pred_dim: string
        Name of the dimension 
        
    
    Returns:
    --------

    """
    
    prediction_idx = prediction.idxmax(dim=pred_dim)
    
    
    true_arr = []
    
    for coord in list(prediction.coords[true_dim].values):
    
        pred_arr = []
        for coord_pred in list(prediction.coords[pred_dim].values):
            pred_arr.append( (prediction_idx.sel(variable= coord) == coord_pred).sum(dim=samp_dim).assign_coords(pred_variable = coord_pred).rename({"variable":"true_variable"}))
        true_arr.append(xr.concat(pred_arr, dim = "pred_variable"))
        
    return xr.concat(true_arr, dim ="true_variable")


    
def cumulative_accuracy(prediction, true_dim = "variable", pred_dim = "dimension", cumsum_dim = "time"):
    """
    Calculates the cumulative accuracy of a prediction matrix over a certain dimension
    
    Parameters:
    -----------
    
    prediction: xarray DataArray
        prediction DataArray with at least three dimension.
    
    true_dim: string
        Variable in which coordinates corresponds to the real values
    
    pred_dim: string
        Dimension which saves the predicted probabilities for each possible output
        
    cumsum_dim: string
        Dimension over which 
    """
    
    # Get the correct predictions 
    prediction_idxmax = prediction.idxmax(dim=pred_dim)
    cumsum = []
    
    for coord in prediction_idxmax.coords[true_dim].values:
        coord_pred = "x0_" + coord
        
        cumsum.append((prediction_idxmax.sel(variable=coord) ==coord_pred).cumsum(dim=cumsum_dim).rename(coord))
    
    length_cumsum_dim = prediction.sizes[cumsum_dim]
    
    cumsum_xr = xr.concat(cumsum, dim = true_dim)
    
    cumsum_sum = xr.DataArray(np.ones(length_cumsum_dim),
                 dims=[cumsum_dim],
                 coords={cumsum_dim:getattr(cumsum_xr,cumsum_dim)}).cumsum(dim=cumsum_dim)
    
    return cumsum_xr/cumsum_sum