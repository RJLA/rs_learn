import numpy as np
from osgeo import gdal
from osgeo.gdalconst import *
import os
import matplotlib.pyplot as plt

class Raster_to_array():
             
    def __init__ (self, 
                  raster_path,
                  raster_name,
                  raster_extension,
                  ws = os.getcwd()):
        '''
        Keyword arguments: 
        
            raster_path,
            raster_name,
            raster_extension
        
        Transform rasters into dask arrays
        '''

        self.misc_output_path = os.path.join(os.getcwd(), 
                          'output_rs_learn',
                          'misc')     
    
        if not os.path.exists(self.misc_output_path):
            os.makedirs(self.misc_output_path)    
            
            
        self.raster_output_path = os.path.join(ws, 
                      'output_rs_learn',
                      'raster_output')

        if not os.path.exists(self.raster_output_path):
            os.makedirs(self.raster_output_path)  
            
        self.ws = ws  

        self.raster_output_path = os.path.join(self.ws, 
                              'output_rs_learn',
                              'raster_output')
            
        self.raster_path = raster_path
        self.file_name = raster_name
        self.file_extension = raster_extension 
        self.raster_file = os.path.join(self.raster_path,
                                    f'{self.file_name}.{self.file_extension}')

        print(f'Opening {self.raster_file} raster')
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
                    
                    print(f'Rows: {self.n_rows}')
                    print(f'Cols: {self.n_cols}')
                    print(f'Pixel size: {self.pixel}')
                    print(f'Number of bands: {self.n_band}')
                    print(f'Projection: {self.projection}')

                    break
                    
                else:
                    
                    print ("Path does not exists")
                    break     
                    
            except IndexError:
                
                print ("File extension is incorrect")
                break  
            
    def make_array_table(self,
                         file_name):
            
        def make_flat_array(band_number):

            band = self.data_source.GetRasterBand(band_number)
            band.ComputeStatistics(0)
            no_data = band.GetNoDataValue()
            
            if no_data == None:
                no_data = -3.40282346639e+038
                
            self.array = band.ReadAsArray(0, 
                                   0, 
                                   self.n_cols, 
                                   self.n_rows)  

            self.array = np.where(self.array == no_data,
                              np.nan,
                             self.array)
      
            self.array = np.round(self.array, 4)

            del band

            flat_array = self.array.flatten('F')
            
            del self.array
            
            return flat_array
        
    
        array_tupple = ()
        
        array_tupple = [array_tupple + (make_flat_array(i + 1), ) 
                                        for i in range(self.n_band)]
        
        self.array_concat = np.concatenate((array_tupple)).T 
        
        
        np.save(os.path.join(self.misc_output_path,
                             f'{file_name}_array_table.pny'),
            self.array_concat)
            
        return self.array_concat
    
    def make_single_array(self, 
                          band_number, 
                          file_name,
                          round_values = True):
        
        band = self.data_source.GetRasterBand(band_number)
        band.ComputeStatistics(0)
        no_data = band.GetNoDataValue()
        
        if no_data == None:
            no_data = -3.40282346639e+038
            
        self.array = band.ReadAsArray(0, 
                               0, 
                               self.n_cols, 
                               self.n_rows)  

        self.array = np.where(self.array == no_data,
                          0,
                         self.array)
        
        if round_values is True:
            self.array = np.round(self.array, 4)
        elif round_values is False:
            pass
                
        np.save(os.path.join(self.misc_output_path,
                     f'{file_name}_array_single.pny'),
                self.array)
        
        return self.array
    
    def single_array_to_raster(self, 
                               array,
                               file_name,
                               title):
        
        self.raster_output_path_name = os.path.join(self.raster_output_path,
                                          f'{file_name}.tif')        
        
        new_source = self.driver.Create(self.raster_output_path_name, 
                        self.n_cols, 
                        self.n_rows,
                        1, 
                        gdal.GDT_Float32)

        new_source.SetGeoTransform(self.gt) 
        new_source.SetProjection(self.projection)
        
        output_raster = new_source.GetRasterBand(1)
        output_raster.ComputeStatistics(0)
            
        output_raster.WriteArray(array, 
                                 0, 
                                 0)    
        
        plt.imshow(array)
        plt.title(title)
        plt.savefig(os.path.join(self.raster_output_path,
                        f'{file_name}.png'),
            dpi = 300,
            edgecolor = 'none')
        plt.show()
        
        
        
    
        
#raster_path = './'
#raster_name ='quezon_city'
#raster_extension = 'tif'
#
#ras_to_df = Raster_to_array(raster_path,
#                                raster_name,
#                               raster_extension)
#array_table = ras_to_df.make_array_table('test')
##
#












