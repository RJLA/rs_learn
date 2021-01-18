```python
import rs_learn as rsl
```

```python

```

```python
# raster_path = os.getcwd()
raster_path = './'
raster_name = 'quezon_city'
raster_extension = 'tif'
```

```python
# read raster file 
ras_to_df = rsl.Raster_to_dataframe(raster_path,
                                 raster_name,
                                  raster_extension)
```

```python
#create dataframe
df = ras_to_df.make_df()
```

```python
#preview dataframe
df.head(3)
```

```python
#change columns names
new_columns = ['Ultra_blue', 
              'Blue',
              'Green',
              'Red',
              'Nir',
              'Swir_1',
              'Swir_2',
              'Brightness_1',
              'Brightness_2',
              'Aerosol',
              'Pixel_qa',
              'Radsat_qa']
df.columns = new_columns
df.head(3)
```

```python
# compute for ndvi the pandas way
df['ndvi'] = (df['Nir'] - df['Red']) / (df['Nir'] + df['Red'])
```

```python
# rasterize and visualize
ras_to_df.df_to_raster(df['ndvi'],
                      'NDVI',
                      'reg')
```

```python
# apply machine learning algorithms, kmeans for example
from sklearn.cluster import KMeans
km = KMeans(n_clusters = 3, 
            random_state=0).fit(df)
clusters = km.predict(df)
```

```python
# rasterize and visualize
ras_to_df.df_to_raster(clusters,
                      'Clusters',
                      'clf')
```

```python

```
