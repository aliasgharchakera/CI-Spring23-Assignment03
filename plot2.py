import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

# worldMap = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# fig, ax = plt.subplots()
# worldMap.plot(ax=ax, facecolor='lightgray', edgecolor='black')
# for i in self.colourMatch:
#     color = self.colourMatch[i]
#     if i in worldMap["iso_a3"].tolist():
#         worldMap[worldMap.iso_a3 == i].plot(color=color, ax=ax)
# plt.show()


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ax = world.plot()

for i in range(len(self.colourMatch)):
    color = self.colourMatch[i]
    country = self.winNeuronList[i].countryName
    if country in world["name"].tolist():
        world[world.name == country].plot(color=color, ax=ax)
plt.show()