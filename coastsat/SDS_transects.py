"""
This module contains functions to analyze the 2D shorelines along shore-normal
transects
    
Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""

# load modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
from shapely.geometry import LineString, MultiLineString, Point


# other modules
import skimage.transform as transform
from pylab import ginput

# CoastSat modules
from coastsat import SDS_tools

def auto_comp_transects(transect_settings):
    """
    Automatically generates transects along a line given as geojson. 
    Output is written to geojson as well. 
    
    Credits: Sven-Arne Quist & JamesS
    Date : 5-May-2020
    
    Arguments:
    -----------
    transect_settings: dict with the following keys
        'reference_filename' : str
            filename of the reference shoreline
        'spacing' : int 
            Profile spacing. The distance at which to space the perpendicular profiles
            In the same units as the original shapefile (e.g. metres)
        'section_length': int
             Length of cross-sections to calculate either side of central line
             i.e. the total length will be twice the value entered here.
        'addtional inputs': dict
            contents of dictionary settings in which general settings for the model are recorded
            contains: 'output_epsg': int with coordinate system defined as epsg code. 
        
    Returns:    
    -----------
    all_transects: GeoDataFrame 
        GeoDataFrame containing LineStrings with all transects perpendicular to the reference line.
    
    """
    spc = transect_settings['spacing']
    sect_len = transect_settings['section_length']
    output_epsg_no = settings['output_epsg']
    sitename = settings['inputs']['sitename']
    filepath = os.path.join(settings['inputs']['filepath'], sitename)
    
    # read reference shoreline in 'refline'
    refline = gpd.read_file(transect_settings['reference_filename'])
    
    # check for projected coordinates
    wgs84 = {'init': 'epsg:4326'}
    if refline.crs != wgs84:
        # if not in wgs84, reproject to get projected coordinates
        refline.crs = wgs84
    
    # convert refline into LineString to used in the forloop 
    line = LineString(refline.geometry[0])
    
    # prepare empty geometry list
    geometry = []

    # prepare emtpy transect ID dictionary     
    transect_id = {"name" : []}
    
    # Calculate the number of profiles to generate
    n_prof = int(line.length/spc)
    
    """ 
    The main structure of this forloop is credited to JamesS 
    and was found on https://gis.stackexchange.com/questions/50108/elevation-profile-10-km-each-side-of-a-line/50841#50841
    """
    
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
        
        # Connect starting point and end point of transect into a LineString
        prof_line = LineString([prof_st, prof_end])
        
        # append line to geometry list 
        geometry.append(prof_line)
        
        # append id to 
        transect_id["name"].append(str(prof))
    
    # group all transects into a single GeoDataFrame 
    all_transects = gpd.GeoDataFrame(transect_id, geometry=geometry)
    
    # reproject into output coordinate system
    outcrs = {'init': 'epsg:'+str(output_epsg_no)}
    all_transects.crs = outcrs
    
    # write to file
    all_transects.to_file(os.path.join(filepath, sitename + '_auto_transects.geojson'), driver='GeoJSON', encoding='utf-8')
    print("Transects are written to file")
    
    return(all_transects)


def create_transect(origin, orientation, length):
    """
    Create a transect given an origin, orientation and length.
    Points are spaced at 1m intervals.
    
    KV WRL 2018
    
    Arguments:
    -----------
    origin: np.array
        contains the X and Y coordinates of the origin of the transect
    orientation: int
        angle of the transect (anti-clockwise from North) in degrees
    length: int
        length of the transect in metres
        
    Returns:    
    -----------
    transect: np.array
        contains the X and Y coordinates of the transect
        
    """   
    
    # origin of the transect
    x0 = origin[0]
    y0 = origin[1]
    # orientation of the transect
    phi = (90 - orientation)*np.pi/180 
    # create a vector with points at 1 m intervals
    x = np.linspace(0,length,length+1)
    y = np.zeros(len(x))
    coords = np.zeros((len(x),2))
    coords[:,0] = x
    coords[:,1] = y 
    # translate and rotate the vector using the origin and orientation
    tf = transform.EuclideanTransform(rotation=phi, translation=(x0,y0))
    transect = tf(coords)
                
    return transect

def draw_transects(output, settings):
    """
    Draw shore-normal transects interactively on top of the mapped shorelines

    KV WRL 2018       

    Arguments:
    -----------
    output: dict
        contains the extracted shorelines and corresponding metadata
    settings: dict with the following keys
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)
            
    Returns:    
    -----------
    transects: dict
        contains the X and Y coordinates of all the transects drawn.
        Also saves the coordinates as a .geojson as well as a .jpg figure 
        showing the location of the transects.       
    """   
    
    sitename = settings['inputs']['sitename']
    filepath = os.path.join(settings['inputs']['filepath'], sitename)

    # plot the mapped shorelines
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.axis('equal')
    ax1.set_xlabel('Eastings [m]')
    ax1.set_ylabel('Northings [m]')
    ax1.grid(linestyle=':', color='0.5')
    for i in range(len(output['shorelines'])):
        sl = output['shorelines'][i]
        date = output['dates'][i]
        ax1.plot(sl[:, 0], sl[:, 1], '.', markersize=3, label=date.strftime('%d-%m-%Y'))
#    ax1.legend()
    fig1.set_tight_layout(True)
    mng = plt.get_current_fig_manager()                                         
    mng.window.showMaximized()
    ax1.set_title('Click two points to define each transect (first point is the ' +
                  'origin of the transect and is landwards, second point seawards).\n'+
                  'When all transects have been defined, click on <ENTER>', fontsize=16)
    
    # initialise transects dict
    transects = dict([])
    counter = 0
    # loop until user breaks it by click <enter>
    while 1:
        # let user click two points
        pts = ginput(n=2, timeout=1e9)
        if len(pts) > 0:
            origin = pts[0]
        # if user presses <enter>, no points are selected
        else:
            # save figure as .jpg
            fig1.gca().set_title('Transect locations', fontsize=16)
            fig1.savefig(os.path.join(filepath, 'jpg_files', sitename + '_transect_locations.jpg'), dpi=200)
            plt.title('Transect coordinates saved as ' + sitename + '_transects.geojson')
            plt.draw()
            # wait 2 seconds for user to visualise the transects that are saved
            ginput(n=1, timeout=2, show_clicks=True)
            plt.close(fig1)
            # break the loop
            break
        
        # add selectect points to the transect dict
        counter = counter + 1
        transect = np.array([pts[0], pts[1]])
        
        # alternative of making the transect the origin, orientation and length
#        temp = np.array(pts[1]) - np.array(origin)
#        phi = np.arctan2(temp[1], temp[0])
#        orientation = -(phi*180/np.pi - 90)
#        length = np.linalg.norm(temp)
#        transect = create_transect(origin, orientation, length)
        
        transects[str(counter)] = transect
        
        # plot the transects on the figure
        ax1.plot(transect[:,0], transect[:,1], 'b-', lw=2.5)
        ax1.plot(transect[0,0], transect[0,1], 'rx', markersize=10)
        ax1.text(transect[-1,0], transect[-1,1], str(counter), size=16,
                 bbox=dict(boxstyle="square", ec='k',fc='w'))
        plt.draw()
        
    # save transects.geojson
    gdf = SDS_tools.transects_to_gdf(transects)
    # set projection
    gdf.crs = {'init':'epsg:'+str(settings['output_epsg'])}
    # save as geojson    
    gdf.to_file(os.path.join(filepath, sitename + '_transects.geojson'), driver='GeoJSON', encoding='utf-8')
    # print the location of the files
    print('Transect locations saved in ' + filepath)
        
    return transects

def compute_intersection(output, transects, settings):
    """
    Computes the intersection between the 2D shorelines and the shore-normal.
    transects. It returns time-series of cross-shore distance along each transect.
    
    KV WRL 2018       

    Arguments:
    -----------
    output: dict
        contains the extracted shorelines and corresponding metadata
    transects: dict
        contains the X and Y coordinates of each transect
    settings: dict with the following keys
        'along_dist': int
            alongshore distance considered caluclate the intersection
              
    Returns:    
    -----------
    cross_dist: dict
        time-series of cross-shore distance along each of the transects. 
        Not tidally corrected.        
    """    
    
    # loop through shorelines and compute the median intersection    
    intersections = np.zeros((len(output['shorelines']),len(transects)))
    for i in range(len(output['shorelines'])):

        sl = output['shorelines'][i]
        
        for j,key in enumerate(list(transects.keys())): 
            
            # compute rotation matrix
            X0 = transects[key][0,0]
            Y0 = transects[key][0,1]
            temp = np.array(transects[key][-1,:]) - np.array(transects[key][0,:])
            phi = np.arctan2(temp[1], temp[0])
            Mrot = np.array([[np.cos(phi), np.sin(phi)],[-np.sin(phi), np.cos(phi)]])
    
            # calculate point to line distance between shoreline points and the transect
            p1 = np.array([X0,Y0])
            p2 = transects[key][-1,:]
            d_line = np.abs(np.cross(p2-p1,sl-p1)/np.linalg.norm(p2-p1))
            # calculate the distance between shoreline points and the origin of the transect
            d_origin = np.array([np.linalg.norm(sl[k,:] - p1) for k in range(len(sl))])
            # find the shoreline points that are close to the transects and to the origin
            # the distance to the origin is hard-coded here to 1 km 
            idx_dist = np.logical_and(d_line <= settings['along_dist'], d_origin <= 1000)
            # find the shoreline points that are in the direction of the transect (within 90 degrees)
            temp_sl = sl - np.array(transects[key][0,:])
            phi_sl = np.array([np.arctan2(temp_sl[k,1], temp_sl[k,0]) for k in range(len(temp_sl))])
            diff_angle = (phi - phi_sl)
            idx_angle = np.abs(diff_angle) < np.pi/2
            # combine the transects that are close in distance and close in orientation
            idx_close = np.where(np.logical_and(idx_dist,idx_angle))[0]     
            
            # in case there are no shoreline points close to the transect 
            if len(idx_close) == 0:
                intersections[i,j] = np.nan
            else:
                # change of base to shore-normal coordinate system
                xy_close = np.array([sl[idx_close,0],sl[idx_close,1]]) - np.tile(np.array([[X0],
                                   [Y0]]), (1,len(sl[idx_close])))
                xy_rot = np.matmul(Mrot, xy_close)
                # compute the median of the intersections along the transect
                intersections[i,j] = np.nanmedian(xy_rot[0,:])
    
    # fill the a dictionnary
    cross_dist = dict([])
    for j,key in enumerate(list(transects.keys())): 
        cross_dist[key] = intersections[:,j]   
    
    # save a .csv file for Excel users
    out_dict = dict([])
    out_dict['dates'] = output['dates']
    for key in transects.keys():
        out_dict['Transect '+ key] = cross_dist[key]
    df = pd.DataFrame(out_dict)
    fn = os.path.join(settings['inputs']['filepath'],settings['inputs']['sitename'],
                      'transect_time_series.csv')
    df.to_csv(fn, sep=',')
    print('Time-series of the shoreline change along the transects saved as:\n%s'%fn)
    
    return cross_dist