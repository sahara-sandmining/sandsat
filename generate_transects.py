#-------------------------------------------------------------------------------
# Name:        perp_lines.py
# Purpose:     Generates multiple profile lines perpendicular to an input line
#
# Author:      JamesS
# Found on: https://gis.stackexchange.com/questions/50108/elevation-profile-10-km-each-side-of-a-line/50841#50841
# 
# Created:     13/02/2013
# 
# Contributor: Sven-Arne Quist
# Updated on:  28-Apr-2020
#-------------------------------------------------------------------------------
""" Takes a shapefile containing a single line as input. Generates lines
    perpendicular to the original with the specified length and spacing and
    writes them to a new shapefile.

    The data should be in a projected co-ordinate system.
"""

import numpy as np
from fiona import collection
from shapely.geometry import LineString, MultiLineString

# ##############################################################################
# User input

# Input shapefile. Must be a single, simple line, in projected co-ordinates
in_shp = r'C:\Users\Administrator\Downloads\CoastSat\poyrefline1.shp'

# The shapefile to which the perpendicular lines will be written
out_shp = r'C:\Users\Administrator\Downloads\CoastSat\poyrefline1_transects.shp'

# Profile spacing. The distance at which to space the perpendicular profiles
# In the same units as the original shapefile (e.g. metres)
spc = 50

# Length of cross-sections to calculate either side of central line
# i.e. the total length will be twice the value entered here.
# In the same co-ordinates as the original shapefile
sect_len = 600
# ##############################################################################

'''
Here we need a different method: 
    
    - input: reference line (either geojson or pickle, probably geojson is the best)
    - output: geojson & pickle (.pkl)
    - transformation: write input file to Shapely LineString called 'line'
'''

'''
original script is here

# Open the shapefile and get the data
#source = collection(in_shp, "r")
data = source.next()['geometry']
line = LineString(data['coordinates'])

# Define a schema for the output features. Add a new field called 'Dist'
# to uniquely identify each profile
schema = source.schema.copy()
schema['properties']['Dist'] = 'float'

# Open a new sink for the output features, using the same format driver
# and coordinate reference system as the source.
sink = collection(out_shp, "w", driver=source.driver, schema=schema,
                  crs=source.crs)
'''

''' This part we keep, including the forloop'''

# Calculate the number of profiles to generate
n_prof = int(line.length/spc)

# Start iterating along the line
for prof in range(1, n_prof+1):
    # Get the start, mid and end points for this segment
    seg_st = line.interpolate((prof-1)*spc)
    seg_mid = line.interpolate((prof-0.5)*spc)
    seg_end = line.interpolate(prof*spc)

    # Get a displacement vector for this segment
    vec = np.array([[seg_end.x - seg_st.x,], [seg_end.y - seg_st.y,]])

    # Rotate the vector 90 deg clockwise and 90 deg counter clockwise
    rot_anti = np.array([[0, -1], [1, 0]])
    rot_clock = np.array([[0, 1], [-1, 0]])
    vec_anti = np.dot(rot_anti, vec)
    vec_clock = np.dot(rot_clock, vec)

    # Normalise the perpendicular vectors
    len_anti = ((vec_anti**2).sum())**0.5
    vec_anti = vec_anti/len_anti
    len_clock = ((vec_clock**2).sum())**0.5
    vec_clock = vec_clock/len_clock

    # Scale them up to the profile length
    vec_anti = vec_anti*sect_len
    vec_clock = vec_clock*sect_len

    # Calculate displacements from midpoint
    prof_st = (seg_mid.x + float(vec_anti[0]), seg_mid.y + float(vec_anti[1]))
    prof_end = (seg_mid.x + float(vec_clock[0]), seg_mid.y + float(vec_clock[1]))


'''
This part we need to change to write the output to geojson & pickle (.pkl)

    # Write to output
    rec = {'geometry':{'type':'LineString', 'coordinates':(prof_st, prof_end)},
           'properties':{'Id':0, 'Dist':(prof-0.5)*spc}}
    sink.write(rec)

# Tidy up
source.close()
sink.close()

'''
