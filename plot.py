import pandas as pd

# statistic_df = pd.read_csv(
#     'https://data.un.org/_Docs/SYB/CSV/SYB65_1_202209_Population,%20Surface%20Area%20and%20Density.csv',
#     encoding='latin-1',
#     header=1,
#     usecols=range(1,9)
# )

statistic_df = pd.read_csv(
    'SYB65_1_202209_Population, Surface Area and Density.csv',
    encoding='latin-1',
    header=1
)

statistic_df.rename(
    columns = {
        statistic_df.columns[0]: 'Region or Country', 
        'Value': 'Urban Percentage'
    }, 
    inplace=True
)

dataframe = statistic_df

import geopandas

country_geopandas = geopandas.read_file(
    geopandas.datasets.get_path('naturalearth_lowres')
)
country_geopandas = country_geopandas.merge(
    dataframe, # this should be the pandas with statistics at country level
    how='inner', 
    left_on=['name'], 
    right_on=['Region or Country']
)

from datetime import datetime
import folium


urban_area_map = folium.Map()
folium.Choropleth(
    geo_data=dataframe,
    name='choropleth',
    data=dataframe,
    columns=['Region or Country', 'Urban Percentage'],
    key_on='feature.properties.name',
    fill_color='Greens',
    nan_fill_color='Grey',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Percentage of population living in Urban areas'
).add_to(urban_area_map)
urban_area_map.save(f'./exports/urban_population_{datetime.now()}.html')