import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib
import matplotlib.cm
import os
import math
import itertools
import matplotlib.pyplot as plt
from IPython.display import Image
import matplotlib.image as mpimg
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets




class map_widgets():
    """
    Class that creates map for a given dataarray. 
    
    """
    
    
    
    def __init__(self,data,path,panel_dim):
        """
        Initializises the maps
        
        Parameters:
        -----------
        data: xarray dataarray
            Dataarray which is visualized  needs to have a variable name
        path: str
            Path were figures should be stored and saved
        panel_dim: str
            Dimension name of the datarray over which panels are created
        
        Returns:
        --------
        
        """
        self.plotting_done = False
        self.path = path
        self.data = data
        self.name = data.name 
        self.panel_dim = panel_dim
            
        if(panel_dim):
            pass
        else:
            panel_dim = "panel_dim"
            self.data = self.data.expand_dims("panel_dim")
        
        self.dims_list = list(self.data.dims)
        self.dims_list.remove(self.panel_dim)
        self.dims_list.remove("x")
        self.dims_list.remove("y")
        
        
        
        list_coords = []
        dict_coords = {}
        
        for dim in self.dims_list:
            dict_coords[dim] = self.data.coords[dim].values
            list_coords.append(self.data.coords[dim].values)
        
        self.list_coords = list_coords
        self.show_dict   = dict_coords
        
        self.imagepath =  os.path.join(path,self.name)
        
        
        os.system("mkdir -p "+ self.imagepath)
        
    
    def create_plots(self,vmin=None, vmax=None, extent=[-90,20,0,90],figsize=(15,15),projection=ccrs.LambertConformal(), n_cols=2, cmap = matplotlib.cm.Reds, bad_color="gray", extend="both"):
        """
        Creates the plots for the map
        """
        
        self.n_cols  = n_cols
        self.n_panel = self.data.sizes[self.panel_dim]
        self.n_rows  = math.ceil(self.n_panel/self.n_cols)
    
        # Remove all Dimensions that are needed for the plot
        

    
        if not (vmin):
            vmin = self.data.min().values
        if not (vmax):
            vmax = self.data.max().values

        
        lon = self.data.lon
        lat = self.data.lat
    
        cmap = cmap
        cmap.set_bad(bad_color,1.)
         
        
        figures =[]
        
        
        # Generate all possible parameter combinations
        self.list_parameter = list(itertools.product(*self.list_coords))
        
        
        # Loop through all paramter combinations
        for parameter_tuple in self.list_parameter:
            dictionary_var = dict(zip(self.dims_list, parameter_tuple))
        
            fig,axes = plt.subplots(nrows = self.n_rows,
                                    ncols = self.n_cols,
                                    subplot_kw = {'projection': projection},
                                    figsize = figsize)
        
            cbar_ax  = fig.add_axes([0.15, 0.05, 0.7, 0.05])
    
            ax_ravel = np.ravel(axes)
            
            data_parameter_tuple = self.data.sel(dictionary_var)
            

            panel_list = self.data.coords[self.panel_dim].values
            for i, panel_coord in enumerate(panel_list):
                
                ax_ravel[i].set_extent(extent, crs = ccrs.PlateCarree())
                ax_ravel[i].coastlines(resolution="50m")
                
                data_tmp  = data_parameter_tuple.sel({self.panel_dim:panel_coord}).transpose("y","x")
                data_plot = np.ma.masked_where(np.isnan(data_tmp), data_tmp)
           
                mesh = ax_ravel[i].pcolormesh(lon, lat, data_plot,vmin= vmin, vmax=vmax, cmap=cmap, transform=ccrs.PlateCarree(), antialiased=True, edgecolor='k', linewidth=0.3)
               
                
                ax_ravel[i].set_title(panel_coord)
                ax_ravel[i].coastlines()
            
            cbar=plt.colorbar(mesh, orientation='horizontal', shrink=1, cax = cbar_ax)
            
            
            # Generate filenames
            filename = "figure_"
            for i,dims in enumerate(self.dims_list):
                filename = filename + str(dims) +"_" +str(parameter_tuple[i])+"_"
            plt.style.use(("dark_background"))
            
            # Save figure
            fig.savefig(os.path.join(self.imagepath,filename+".png"))
            plt.close()
        print("Plotting done!")
        self.plotting_done = True
        
    def interactive_map(self):
        """
        Generates interactive visualization of the saved plots
        
        Parameters:
        -----------
        
        """
        
        if(self.plotting_done==False):
            print("Error plots need to be created first")
            return 0 
            
        widgets_arr = []
        parameter_dictionary = {}
        
        for dim in self.show_dict.keys():
            widgets_arr.append(widgets.SelectionSlider(description = dim, options=self.show_dict[dim]))
            parameter_dictionary[dim] = widgets_arr[-1]
        
        out = widgets.interactive_output(self.show_plot, parameter_dictionary)
        
        return out, widgets_arr
        
    def show_plot(self, **kwargs):
        filename = "figure_"
        for dim, parameter in kwargs.items():
            filename = filename + str(dim)+"_"+str(parameter)+"_"
        filename = os.path.join(self.imagepath,filename+".png")    
        display(Image(filename))
        
        
        

        
def interactive_line_plot(data, panel_dim, line_dim, x_dim, n_cols):
    dims_list = list(data.dims)
    
    # Remove all dimensions visualized in the later line plot
    if (panel_dim): dims_list.remove(panel_dim)
    if (line_dim):  dims_list.remove(line_dim)
    dims_list.remove(x_dim)
    
    #print(dims_list)
    
    parameter_dictionary = {}
    
    parameter_dictionary["data"]       = widgets.fixed(data)
    parameter_dictionary["panel_dim"]  = widgets.fixed(panel_dim)
    parameter_dictionary["line_dim"]   = widgets.fixed(line_dim)
    parameter_dictionary["n_cols"]     = widgets.fixed(n_cols)
    
    widgets_arr = []
    
    for i,dim in enumerate(dims_list):
        widgets_arr.append(widgets.SelectionSlider(description = dim, options=data.coords[dim].values))
        parameter_dictionary[dim] = widgets_arr[i]
    
    out = widgets.interactive_output(line_plot, parameter_dictionary)
    return widgets_arr, out



def line_plot(data, panel_dim, line_dim, n_cols, **kwargs):
    """
    
    """
    import math
    
    
    # Selects the subarray according to additional keyword args
    for key, value in kwargs.items():
        data = data.sel({key:value})
    
    
    if(panel_dim):
        n_panels = data.sizes[panel_dim]
    else:
        panel_dim = "panel_dim"
        data = data.expand_dims(dim=panel_dim)
        
    if(line_dim):
        n_lines = data.sizes[line_dim]
    else:
        line_dim = "line_dim"
        data = data.expand_dims(dim=line_dim)
    
    n_panels = data.sizes[panel_dim]
    n_lines = data.sizes[line_dim]
    
    n_rows = math.ceil(n_panels/n_cols)
    
    
    fig, ax = plt.subplots(n_rows,n_cols,figsize=(15,15))
    ax_ravel = np.ravel(ax)
    
    for i, panelname in enumerate(data[panel_dim]):
        for j, linename in enumerate(data[line_dim]):
            data_tmp = data.sel({panel_dim:panelname,line_dim:linename})
            ax_ravel[i].set_title(panelname.values)
            ax_ravel[i].plot(data_tmp, label = linename.values)
            ax_ravel[i].set_ylabel(data_tmp.name)
            ax_ravel[i].set_xlabel(data_tmp.dims[0])
            
            ax_ravel[i].legend()
            
    plt.style.use(("dark_background"))