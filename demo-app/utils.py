from arcgis.gis import GIS
from arcgis.geocoding import geocode
from arcgis.geometry import Point
import pandas as pd 


gis = GIS()

df = pd.read_csv('./data/2022/processed_weball.csv')

addresses = df[['Street one','Street two','City or town','State','ZIP code']].astype(str).agg(' '.join, axis=1)

for n, i in enumerate(addresses):
    geocode_result = geocode(address=i)
    if i and geocode_result:
        print(geocode_result)
        df.iloc[n, list(df.columns).index('lat')] = geocode_result[0]['location']['y']
        df.iloc[n, list(df.columns).index('lon')] = geocode_result[0]['location']['x']
    else:
        pass

df.to_csv('./data/2022/processed_weball_updated_coordinates.csv')
