import tensorflow as tf
import xarray as xr


def regression_loss(model):
    """
    Metric for a tensorflow model which tracks the loss due to regularization terms in all layers
    
    Disclaimer: The metrics are evaluated at the end of the training after parameters have been updated. Therefore the regularization loss for the training is shifted
    by a factor of 1 epoch.
    
    
    Parameters:
    -----------
    model: tf.keras.Model
        keras model which regularization terms are analysed
    
    Returns:
    --------
    regression_loss_term: tf.Tensor
        Sum of the regularization terms in all layers
    """
    
    
    
    def regression_loss_term(y_true, y_pred):
        sum_reg = []
        for layer in model.layers:
            if(layer.losses):
                sum_reg.append(layer.losses[0])
        
        total = tf.add_n(sum_reg)
        return total
    
    
    return regression_loss_term





class callback_monitor_regularization(tf.keras.callbacks.Callback):
    """
    
    
    """
    
    def __init__(self, input_data, label_data, label_data_coords):
        self.regression_loss = []
    
    def on_train_begin(self, logs = None):
        self.regression_loss.append(  xr.DataArray(regression_loss(self.model)(0,0).numpy()).assign_coords({"epoch":"start"}))
                
    def on_epoch_end(self, epoch, logs=None):
        self.regression_loss.append(  xr.DataArray(regression_loss(self.model)(0,0).numpy()).assign_coords({"epoch":epoch}))
        
    def on_train_end(self, logs=None):
        self.data_xr = xr.concat(self.regression_loss, dim = "epoch").rename("regularization")
        
        

        
class callback_save_prediction(tf.keras.callbacks.Callback):
    """
    
    
    
    """
    
    def __init__(self, input_data, label_data, label_data_coords):
        self.input_data = input_data
        self.label_data = label_data
        self.label_data_coords = label_data_coords
        
    def on_train_begin(self, logs= None):
        self.prediction = []
        prediction = self.model.predict(self.input_data)
        #Save the prediction with random initialized weights at the beginning of the training:
        self.prediction.append(xr.DataArray(prediction,coords = self.label_data_coords).assign_coords({"epoch":"start"} )  )
        
    def on_epoch_end(self, epoch, logs =None):
        prediction = self.model.predict(self.input_data)
        self.prediction.append(xr.DataArray(prediction,coords = self.label_data_coords).assign_coords({"epoch": epoch } )  )
        
    def on_train_end(self, logs=None):
        self.data_xr = xr.concat(self.prediction, dim = "epoch").rename("prediction")

        
class callback_metrics_end_epoch(tf.keras.callbacks.Callback):
    """
    
    
    
    """
    def __init__(self, input_data, label_data, label_data_coords):
        self.input_data = input_data
        self.label_data = label_data
        self.label_data_coords = label_data_coords
        
    def on_train_begin(self, logs= None):
        label_pred = self.model.predict(self.input_data)
        print(self.model.metrics)
        pass
        
        for metric in self.model.metrics:
            name = metric.__name__
            setattr(self, name, [])
            metric_da = xr.DataArray(metric(self.label_data,label_pred).numpy(), coords = self.label_data_coords["sample"].coords)
            
            getattr(self,name).append(metric_da.assign_coords({"epoch":"start"}))
    
        
    def on_epoch_end(self, epoch, logs = None):
        label_pred = self.model.predict(self.input_data)
        
        for metric in self.model.metrics:
            name = metric.__name__
            
            metric_da = xr.DataArray(metric( self.label_data, label_pred).numpy(), coords = self.label_data_coords["sample"].coords)
            getattr(self,name).append(metric_da.assign_coords({"epoch":epoch}))
            
            
    def on_train_end(self,logs=None):
        metrics_arr = []
        for metric in self.model.metrics:
            name_or = metric.__name__
            name_xr = name_or+"_xr"
            
            metrics_arr.append( xr.concat(getattr(self,name_or), dim= "epoch").rename(name_or) )
            setattr(self, name_xr, xr.concat( getattr(self,name_or), dim="epoch") )
            
        self.data_xr = xr.merge(metrics_arr)#.rename("metrics")
        
        
        
        
class TimeHistory(tf.keras.callbacks.Callback):
    
    def __init__(self, input_data, label_data, label_data_coords):
        self.setattr(self,start,0)
        self.setattr(self,duration,0)

    def on_train_begin(self, logs=None):
        self.start = time.time()
     
    def on_train_end(self, logs = None):
        self.duration = time.time() - self.start