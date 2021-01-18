from osgeo import gdal_merge_customize as gmc
import os

def make_mosaic(file_name,*args):

    raster_output_path = os.path.join(os.getcwd(), 
                      'output_rs_learn',
                      'raster_output')
                    
    if not os.path.exists(raster_output_path):
        os.makedirs(raster_output_path)
    
    merge_file_name = os.path.join(raster_output_path, 
    '%s.tif'%file_name)
    
    gmc.main(['','-o', merge_file_name, 
              *args, '-separate'])

    print('mosaicking successful')
