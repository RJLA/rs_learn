from osgeo import gdal_merge_customize as gmc
import sys
import os

def merge_rasters(ws, output_name,
                  *args):
    
    raster_output_path = os.path.join(ws, 
                      'output_rs_learn',
                      'raster_output')

    if not os.path.exists(raster_output_path):
            os.makedirs(raster_output_path)   
    
    sys.path.append(r'C:\Users\Dlaniger\Anaconda3\Lib\site-packages\osgeo')
    
    oname = os.path.join(raster_output_path,
                         f'merge_{output_name}.tif')
     
    gmc.main(['', 
              '-separate', 
              '-o', oname, 
              *args])    
