import pandas as pd
import numpy as np
from osgeo import gdal
from osgeo.gdalconst import *
import os
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from datetime import datetime

class Image_to_array():
             
    def __init__ (self, 
                  raster_path,
                  raster_name,
                  raster_extension,
                  ws = os.getcwd()):
        print('''
        args: raster_path,
        raster_name,
        raster_extension
              ''')

        raster_output_path = os.path.join(ws, 
                      'output_rs_learn',
                      'raster_output')
        
    
        if not os.path.exists(raster_output_path):
            os.makedirs(raster_output_path)  
        
        self.ws = ws  

        self.raster_output_path = os.path.join(self.ws, 
                              'output_rs_learn',
                              'raster_output')
                                     
            
        self.raster_path = raster_path
        self.file_name = raster_name
        self.file_extension = raster_extension 
        self.raster_file = os.path.join(raster_path,'%s.%s'%(self.file_name,
                                                            self.file_extension))
        print('Opening %s raster'%self.raster_file)
        print()
        
        while True:
            
            try:
                
                if os.path.exists(self.raster_file) is True:
                    
                    self.data_source = gdal.Open(self.raster_file,
                                              GA_ReadOnly) 
                    self.gt = self.data_source.GetGeoTransform()  
                    self.driver = self.data_source.GetDriver()
                    self.n_cols = self.data_source.RasterXSize
                    self.n_rows = self.data_source.RasterYSize
                    self.pixel = self.gt[1]
                    self.n_band = self.data_source.RasterCount
                    self.projection = self.data_source.GetProjection()
                    
                    print('Rows: %s' %self.n_rows)
                    print('Cols: %s' %self.n_cols)
                    print('Pixel size: %s' %self.pixel)
                    print('Number of bands: %s'%self.n_band)
                    print('Projection: %s' %self.projection)

                    break
                    
                else:
                    
                    print ("Path does not exists")
                    break     
                    
            except IndexError:
                
                print ("File extension is incorrect")
                break  
    
                        
    def convert_to_array(self):

        band = self.data_source.GetRasterBand(1)
        band.ComputeStatistics(0)
        no_data = band.GetNoDataValue()
        array = band.ReadAsArray(0, 
                               0, 
                               self.n_cols, 
                               self.n_rows)  

        array = np.where(array == no_data,
                          np.nan,
                         array)
        
        return array
    
    def binarize_array(self,
                       array,
                       filename,
                       title_name):
    
        graph_output_path = os.path.join(os.getcwd(), 
                    'output_rs_learn',
                    'graphs')
        
        f, ax = plt.subplots(figsize = (8,8))
    
        if not os.path.exists(graph_output_path):
            os.makedirs(graph_output_path)
    

        #get threshold
        
        array = np.where(array == 0,
                  np.nan,
                 array)
        clean_arr = array[~np.isnan(array)]
        thresh = threshold_otsu(clean_arr)
        print()
        print()
        print(f'Threshold is {thresh}')
        print()
        print()
        plt.hist(clean_arr.ravel(), 
                 bins = 100,
                 color = 'grey',
                 alpha = 0.8)

        array = np.where((array < thresh) | (array is np.nan), 1, 0)
        
        plt.title(f'Histogram of {title_name}', 
                  fontsize = 15,
                  loc = 'left')
        plt.xlabel('Values',
                   fontsize = 15)
        plt.ylabel('Count', 
                   fontsize = 15)

        plt.axvline(thresh, 
                    color='k',
                    label = f'threshold: {np.round(thresh,4)}')
        plt.legend(fontsize = 15)
        plt.tight_layout()
        plt.savefig(os.path.join(graph_output_path,
                                f'{filename}_thresh.png'),
                    dpi = 300,
                    edgecolor = 'none')  
        plt.show()
        
        return array, thresh

    def convert_to_image(self, 
                  array, 
                  output_filename):
        
        date_time_1 = str(datetime.now())
        date_time_2 = date_time_1.replace(":",
                                      "-").replace(' ',
                                                   '-').split('.')[0]        
        
        self.raster_output_path_name = os.path.join(self.raster_output_path,
                                  '%s_%s_%s.tif'%\
                                  (output_filename,
                                   self.file_name,
                                   date_time_2))
#                
        new_source = self.driver.Create(self.raster_output_path_name, 
                        self.n_cols, 
                        self.n_rows,
                        1, 
                        GDT_Float32)
#
#        print('+++++++++++++++++++++')
#        print(self.gt)
        new_source.SetGeoTransform(self.gt) 
        new_source.SetProjection(self.projection)
        
        output_raster = new_source.GetRasterBand(1)
        output_raster.ComputeStatistics(0)
        
        output_raster.WriteArray(array, 
                             0, 
                             0)
        
        
        new_source.FlushCache()
        output_raster.FlushCache()  
        
        plt.imshow(array, 
                   cmap = 'Spectral_r')
        plt.colorbar()
        plt.axis('off')
        
        plt.savefig(os.path.join(self.raster_output_path,
                                '%s.png'%output_filename),
                    dpi = 300,
                    edgecolor = 'none')     
        
        del output_raster
        del new_source
        del self.data_source

#ia = Image_to_array(
#        raster_path = r'C:\Users\Dlaniger\Anaconda3\Lib\site-packages\rs_learn',
#        raster_name = 'test',
#        raster_extension = 'tif',
#        ws = os.getcwd()
#              )
#
#arr = ia.convert_to_array()
#binary, thresh = ia.binarize_array(arr,
#                           'test',
#                           'test')
#
#ia.convert_to_image(binary,
#                 'test')
