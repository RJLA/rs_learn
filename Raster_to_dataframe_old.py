import pandas as pd
import numpy as np
from osgeo import gdal
from osgeo.gdalconst import *
import os
import matplotlib.pyplot as plt
from skimage import exposure
from datetime import datetime

class Raster_to_dataframe():
             
    def __init__ (self, 
                  raster_path,
                  raster_name,
                  raster_extension,
                  no_data_value = -3.40282346639e+038,
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

        self.no_data_value = no_data_value
        
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
        
    def make_df(self, 
                round_values = True):
                        
        def make_flat_array(band_number):

            band = self.data_source.GetRasterBand(band_number)
            band.ComputeStatistics(0)
            no_data = band.GetNoDataValue()
#            print(no_data)
#            no_data = self.no_data_value
            array = band.ReadAsArray(0, 
                                   0, 
                                   self.n_cols, 
                                   self.n_rows)  

            array = np.where(array == no_data,
                              np.nan,
                             array)
      
            if round_values is True:
                array = np.round(array, 4)
            elif round_values is False:
                pass

            del band

            flat_array = array.flatten('F')
            del array
            
            return flat_array
                
        array_tupple = ()

        for i in range(self.n_band):
            array_tupple = array_tupple + (make_flat_array(i + 1), )

        array_final = np.vstack(array_tupple).T
        
        del array_tupple
        
        self.main_df = pd.DataFrame(array_final, 
                            columns = ['band_%s'%(i + 1) for i in range(self.n_band)])
        
        
        del self.data_source
        del array_final
        
        self.main_df['prediction'] = np.nan
        df_clean = self.main_df.iloc[:,:-1].dropna(axis = 0, 
                                                 how = 'all')

        df_clean = df_clean.loc[~(df_clean == 0.0).all(axis = 1)]
        
        self.df_to_use = df_clean.fillna(0)
        
        del df_clean
        
        self.df_to_use_idx = self.df_to_use.index.values
        
        return self.df_to_use
         

    def df_to_raster(self, 
                  prediction, 
                  output_filename,
                  analysis_type):
        
        self.df_to_use = self.df_to_use.set_index(self.df_to_use_idx)
        self.df_to_use['prediction'] = prediction * 1
        self.main_df.update(self.df_to_use)
        df_as_array = np.array(self.main_df['prediction'])
        df_as_array = df_as_array.reshape(self.n_cols, 
                                      self.n_rows).T
        
        date_time_1 = str(datetime.now())
        date_time_2 = date_time_1.replace(":",
                                      "-").replace(' ',
                                                   '-').split('.')[0]
        
        self.raster_output_path_name = os.path.join(self.raster_output_path,
                                          '%s_%s_%s.tif'%\
                                          (output_filename,
                                           self.file_name,
                                           date_time_2))

        self.min_value = self.df_to_use['prediction'].min()
        self.max_value = self.df_to_use['prediction'].max()
        
        def create_new_source(data_type):
            
            new_source = self.driver.Create(self.raster_output_path_name, 
                            self.n_cols, 
                            self.n_rows,
                            1, 
                            data_type)

            new_source.SetGeoTransform(self.gt) 
            new_source.SetProjection(self.projection)
            
            return new_source
                
        if analysis_type == 'clf':
            
            new_source = create_new_source(gdal.GDT_Byte)
            output_raster = new_source.GetRasterBand(1)
            output_raster.ComputeStatistics(0)
            
            output_raster.WriteArray(df_as_array + 1, 
                                 0, 
                                 0)
            plt.imshow(df_as_array + 1, 
                       cmap = 'Set1')
            plt.colorbar()
            

            plt.title('%s'%output_filename, 
                      fontsize = 20)
            
            plt.savefig(os.path.join(self.raster_output_path,
                                    '%s.png'%output_filename),
                        dpi = 300,
                        edgecolor = 'none')
            
            plt.axis('off') 
            
        elif analysis_type == 'reg':
            try:
                img_eq = exposure.equalize_hist(prediction.to_numpy())
                
                self.main_df['prediction_eq'] = self.main_df['prediction'] = np.nan
                self.df_to_use['prediction_eq'] = img_eq

                df_as_array_eq = np.array(self.main_df['prediction_eq'])
                
                df_as_array_eq = df_as_array_eq.reshape(self.n_cols, 
                              self.n_rows).T
                
                
                fig, ax = plt.subplots()
    
                cax = ax.imshow(df_as_array_eq, 
                                cmap = 'Spectral_r')
    
                cbar = fig.colorbar(cax)
    
                cbar.set_ticks([img_eq.min(),
                                (img_eq.max() - img_eq.min()) / 2,
                                img_eq.max()])
                
                cbar.set_ticklabels(['Low',
                                     'Medium',
                                    'High'])        


                plt.title('%s'%output_filename, 
                          fontsize = 20)
                
                plt.savefig(os.path.join(self.raster_output_path,
                                        '%s.png'%output_filename),
                            dpi = 300,
                            edgecolor = 'none')
                
                plt.axis('off')          
                del self.df_to_use['prediction_eq']
                del img_eq 
                del df_as_array_eq
                                           
            except:
                ValueError     
                print('Error visualizing results')

                    
            self.main_df.update(self.df_to_use)                                        
            new_source = create_new_source(GDT_Float32)
            output_raster = new_source.GetRasterBand(1)
            output_raster.ComputeStatistics(0)
            
            output_raster.WriteArray(df_as_array, 
                                 0, 
                                 0)                                

          
        print()
        print(f'Output raster save at {self.raster_output_path_name}')
          
        new_source.FlushCache()
        output_raster.FlushCache()      
        

        
        del output_raster
        del new_source
        del df_as_array
        
        
#        
#rd = Raster_to_dataframe(raster_path = r'C:\Users\Dlaniger\Anaconda3\Lib\site-packages\rs_learn',
#                                    raster_name = 'quezon_city',
#                                    raster_extension = 'tif')
#
#
#df_main = rd.make_df()
#print(df_main)
#rd.df_to_raster(df_main['band_1'],
#         'test',
#         'reg')
#





