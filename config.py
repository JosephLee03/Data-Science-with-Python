# -*- coding: utf-8 -*-
# @Author: Li Jiaxing
# @Email:li_jax@outlook.com
# @Created: 2024-03-31

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature

# the variable "places" is a dictionary that contains the longitude and latitude of 15 destinations
places = {"A":(10, 10), "B":(15, 20), "C":(20, 70), "D":(40, 10), "E":(50, 150), 
          "F":(125, 35), "G":(60, 120), "H":(100, 160), "I":(90, 100), "J":(120, 120),
          "K":(130, 40), "L":(20, 120), "M":(35, 40), "N":(60, 60), "O":(90, 90),
          }

# the start place
start = "A"

""" In order to encode the places, I use the following string "encode" to represent the sequence of 15 places,
    and I will create a function in function.py which I can input this string and places matrix, then return 
    the sum of the distances.
 
   the regulation of string:
    1. the length of the string should be 16, and the first and last character should be the same
    2. the string should contain all the 15 places, and each place should appear only once
 
"""
encode = "ABCDEFGHIJKLMNOA"  # it's just a sample for test, the sequence may be different after several epochs


# population size
pop_size = 100

# crossover probability
crossover_probability = 0.5

# mutation probability
mutation_probability = 0.3

# crossover start point
crossover_start = 7


if __name__ == "__main__":
    
    # plot the map
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    # set the extent of the map
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())  # 你可以根据需要调整范围

    # add features
    land = NaturalEarthFeature('physical', 'land', scale='50m', facecolor='0.8')
    ocean = NaturalEarthFeature('physical', 'ocean', scale= '50m', facecolor='0.5')
    countries = NaturalEarthFeature('cultural', 'admin_0_countries', scale = '50m', facecolor='none', edgecolor='black')

    ax.add_feature(land)
    ax.add_feature(ocean)
    ax.add_feature(countries)

    for place, (longitude, latitude) in places.items():
        ax.scatter(longitude, latitude, s=200, facecolors='none', edgecolors='red', zorder=5)
    ax.set_title('Locations on the Map')

    plt.show()