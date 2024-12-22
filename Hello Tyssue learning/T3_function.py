#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script defines all the sub-fuctions needed for my T3 transition.
Then I assemble them into a complete T3 transition main function.
"""

import numpy as np


def swap_detection(sheet, edge, epsilon):
    """
    Given an edge ID and epsilon, this function returns a list of vertex ID 
    that is within the "box" region of this edge that should perform T3 element
    intersection operation.

    Parameters
    ----------
    sheet : An Eptm instance
    
    edge : Int
        ID of the edge
    epsilon : float
        epsilon used for calculating 'box' region.

    Returns
    -------
    A list of vertex IDs that needs a T3 transition.

    """
    # Initialise the list for return use.
    verts = []
    # Grab the vertex ID of the endpoints of the edge.
    edge_end1 = sheet.edge_df.loc[edge,'srce']
    edge_end2 = sheet.edge_df.loc[edge,'trgt']
    # Set the x1, y2 and x2, y2 values based on the edge_end1 and 2.
    x1 = sheet.vert_df.loc[edge_end1, 'x']
    x2 = sheet.vert_df.loc[edge_end2, 'x']
    y1 = sheet.vert_df.loc[edge_end1, 'y']
    y2 = sheet.vert_df.loc[edge_end2, 'y']
    # Find the larger and smaller x,y values to compute the box region.
    x_larger = max(x1, x2)
    x_smaller = min(x1, x2)
    y_larger = max(y1, y2)
    y_smaller = min(y1, y2)
    x_larger += epsilon
    x_smaller -= epsilon
    y_larger += epsilon
    y_smaller -= epsilon
    # Now define the box region.
    # That is: {(x,y): x_smaller < x < x_larger and y_smaller < y < y_larger}
    for i in sheet.vert_df.index:
        if i in [edge_end1, edge_end2]:
            continue
        else:
            x = sheet.vert_df.loc[i,'x']
            y = sheet.vert_df.loc[i,'y']
            if x_smaller < x < x_larger and y_smaller < y < y_larger:
                verts.append(i)
            else:
                continue
    return verts

    

def case_classifier(sheet, edge, vert):
    """
    This function takes a pair of edge and vertex, returns their relative
    position as case number.

    Parameters
    ----------
    sheet : Eptm instance
    
    edge : 
        ID of the edge.
    vert : 
        ID of the vertex.

    Returns
    -------
    1 or 2
    Case 1 means the closest point is at one of the endpoints of the edge.
    Case 2 means the closest point is between the endpoints of the edge.

    """
    # Extract the coordinate of the srce and trgt point.
    edge_end1 = sheet.edge_df.loc[edge,['sx','sy']].to_numpy(dtype = float)
    edge_end2 = sheet.edge_df.loc[edge,['tx','ty']].to_numpy(dtype = float)
    # Unit vector of the line.
    line_unit = sheet.edge_df.loc[edge,['ux','uy']].to_numpy(dtype= float)
    # Now extract the coordinate of the point.
    point = sheet.vert_df.loc[vert,['x','y']].to_numpy(dtype=float)
    # adjust the coordinate of the point with regards to the srce.
    # then take the unit vector of it.
    srce_p = point-edge_end1
    srce_p_unit = srce_p/np.linalg.norm(srce_p)
    dot = np.dot(srce_p_unit, line_unit)
    if dot <0:
        nearest = edge_end1
        distance = np.linalg.norm(nearest-point)
        return 1, nearest, distance
    elif dot >1:
        nearest = edge_end2
        distance = np.linalg.norm(nearest-point)
        return 1, nearest, distance
    else:
        nearest = edge_end1 + dot*line_unit
        distance = np.linalg.norm(nearest-point)
        return 2, nearest, distance



def perturbate_T3(sheet, vert1, vert2, d_sep):
    """
    This function should be used when case_classifier() returns case 1.
    
    This function takes two vertices. One of them is the incoming vertex, 
    the other one is the endpoint of the edge. Then perturbate their location 
    slightly.
    

    """
    # Extract the coordinates of two vertices, then draw a virtual line between.
    v1_coord = sheet.vert_df.loc[vert1,['x','y']].to_numpy(dtype=float)
    v2_coord = sheet.vert_df.loc[vert2,['x','y']].to_numpy(dtype=float)
    virtual_line = v2_coord-v1_coord
    # use unit vector of the line to find coordinate of the middle point.
    length_vline = np.linalg.norm(virtual_line)
    unit_vline = virtual_line/length_vline
    mid_coord = v1_coord + length_vline/2 * unit_vline
    # Use the vector from midpoint to v2 to find the perpendicular vector.
    mid_v2 = v2_coord-mid_coord
    mid_perpendicular = np.array([-mid_v2[1],mid_v2[0]])
    mid_perpendicular = mid_perpendicular/np.linalg.norm(mid_perpendicular)
    mid_perpendicular = d_sep * mid_perpendicular
    
    # Now, we need to update the postion of vert1 and vert2.
    # Need the vector from v1 to mid for updating vert1.
    v1_mid = mid_coord-v1_coord
    sheet.vert_df.loc[vert1,['x','y']] += (v1_mid + mid_perpendicular )
    # Then update vert2.
    sheet.vert_df.loc[vert2,['x','y']] += (-mid_v2 - mid_perpendicular)
    return True
    




def intersection_point(edge, vert):
    """
    This function should be used when case_classifier() returns case 2.
    
    This function takes the edge and vertex. It returns the coordinate of the 
    inteserction (which is the closest point), and the distance between the 
    vertex and the insection point.

    Returns
    -------
    Distance: Float
        This should be the distance between the incoming vertex and the inter-
        section point.
    Coordinate: List
        This should be an list object that stores the coordinates of the inter-
        section point.

    """

    pass




def adjacency_check(sheet, vert1, vert2):
    """
    Returns True if vert1 and vert2 are connected by an edge. Otherwise False
    """
    
    exists = sheet.edge_df[
        ((sheet.edge_df['srce'] == vert1) & (sheet.edge_df['trgt'] == vert2)) |
        ((sheet.edge_df['srce'] == vert2) & (sheet.edge_df['trgt'] == vert1))
    ].any().any()  # Checks if any rows satisfy the condition

    return exists  # Return True if such a row exists, False otherwise



def merge_unconnected_vert(sheet,vert1, vert2):
    """
    This function marges two vertices that are not originally connected by an
    edge.
    First, the two vertices are connected by a new edge. 
    Then calling the collapse_edge() function to merge the two vertices.


    Returns
    -------
    ID of the merged vertex

    """
    
    pass
    
    
    
    
def insert_into_edge(sheet, edge, vert, position):
    """
    This function inserts the vertex (vert) into the edge (edge), the insertion
    takes place at the coordinate specified by position
    """

    pass



def resolve_local(sheet, end1, end2, midvert):
    """
    Vertices specified by end1, end2 and midvert should be colinear. 
    
    1) Use the edge formed by midvert and one of end1, end2 vertices as the
    base vector.
    
    2) Store the coordinates of all vertices that are connected to midvert, not
    including end1 and end2.
    
    3) Compute the dot product between each pair, form a list of vertex id based
    on the dot product as an 'order'.
    
    4) Based on the ordered vertex list, put a new vertex on the corresponding 
    edge with the correct coordinates, reconnect the vertices. 
    Step (4) is processed in a vertex by vertex fashion.


    """
    
    pass


def T3_swap(sheet, edge, vert):
    """
    This is the final T3 transition function that is assembled from subfunctions
    defined above.


    """
    pass





"""
This is the end of the script.
"""
