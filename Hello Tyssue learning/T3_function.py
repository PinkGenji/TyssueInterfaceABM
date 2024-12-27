#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script defines all the sub-fuctions needed for my T3 transition.
Then I assemble them into a complete T3 transition main function.
"""

import numpy as np
import pandas as pd

from tyssue.topology.base_topology import collapse_edge

from my_headers import put_vert

    

def dist_computer(sheet, edge, vert, d_sep):
    """
    This function takes a pair of edge and vertex, returns the distance and 
    the intersection point.

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
    Case 1 means the closest point is the srce vertex of the edge.
    Case 2 means the closest point is the trgt vertex of the edge.
    Case 3 means the closest point is between the endpoints of the edge.

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
        nearest = edge_end1 + d_sep*line_unit
        distance = np.linalg.norm(nearest-point)
        return nearest, distance
    elif dot >1:
        nearest = edge_end2 - d_sep*line_unit
        distance = np.linalg.norm(nearest-point)
        return nearest, distance
    else:
        nearest = edge_end1 + dot*line_unit
        distance = np.linalg.norm(nearest-point)
        return nearest, distance






def adjacency_check(sheet, edge, vert):
    """
    Returns True if there is an edge between the vert and any one of
    endpoints of the edge.
    """
    end1_id = sheet.edge_df.loc[edge,'srce']
    end2_id = sheet.edge_df.loc[edge,'trgt']
    
    # Check adjacency
    is_adjacent_to_end1 = ((sheet.edge_df['srce'] == end1_id) & (sheet.edge_df['trgt'] == vert)).any() or \
                          ((sheet.edge_df['srce'] == vert) & (sheet.edge_df['trgt'] == end1_id)).any()
    is_adjacent_to_end2 = ((sheet.edge_df['srce'] == end2_id) & (sheet.edge_df['trgt'] == vert)).any() or \
                          ((sheet.edge_df['srce'] == vert) & (sheet.edge_df['trgt'] == end2_id)).any()

    # Return True if adjacency to either endpoint is found
    return is_adjacent_to_end1 or is_adjacent_to_end2



def merge_unconnected_verts(sheet,vert1, vert2):
    """
    This function marges two vertices that are not originally connected by an
    edge.
    First, the two vertices are connected by a new edge. 
    Then calling the collapse_edge() function to merge the two vertices.


    Returns
    -------
    ID of the merged vertex

    """

    # Get the last row for concat.
    last_row = sheet.edge_df.tail(1)
    # Concatenate the original dataframe with the last row
    sheet.edge_df = pd.concat([sheet.edge_df, last_row], ignore_index=True)
    
    # Now connect the relevant verts by updating the entries.
    sheet.edge_df.loc[sheet.edge_df.index[-1],'srce'] = vert1
    sheet.edge_df.loc[sheet.edge_df.index[-1],'trgt'] = vert2
    
    # Collapse the new edge.
    return collapse_edge(sheet, sheet.edge_df.index[-1])
    
    # Note: Then need to sheet.reset_index(), then geom.update_all(sheet).


    
    
    
def insert_into_edge(sheet, edge, vert, position):
    """
    This function inserts the vertex (vert) into the edge (edge), the insertion
    takes place at the coordinate specified by position
    
    First, we put a new vertex on the edge with cut_place coordinate.
    Then, we update all the entries of vert_id to the new vertex id.
    
    Parameters
    ----------
    sheet : Eptm instance
    
    edge : 
        ID of the edge.
    vert : 
        ID of the vertex.
    position:
        coordinate used to generate the new vertex.
    
    """
    
    # First, put a new vertex on the edge, the new vertex has ID, cut_id
    cut_vert, cut_edge, cut_op_edge = put_vert(sheet, edge, position)
    
    # Update the relevant entry
    for i in sheet.edge_df.index:
        if sheet.edge_df.loc[i,'srce'] == vert:
            sheet.edge_df.loc[i,'srce'] = cut_vert
        elif sheet.edge_df.loc[i, 'trgt'] == vert:
            sheet.edge_df.loc[i,'trgt'] = cut_vert
        else:
            continue
    return cut_vert
    





def resolve_local(sheet, end1, end2, midvert, d_sep):
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
    
    # Collect all the vertices that are connected to the vertex.
    associated_vert = set()
    for i in sheet.edge_df.index:
        srce = sheet.edge_df.loc[i, 'srce']
        trgt = sheet.edge_df.loc[i, 'trgt']
        if srce == midvert and trgt not in {end1, end2}:
            associated_vert.add(trgt)
        elif trgt == midvert and srce not in {end1, end2}:
            associated_vert.add(srce)

    # Use midvert -> end1 to get a principle unit vector.
    end1_coord = sheet.vert_df.loc[end1,['x','y']].to_numpy(dtype=float)
    mid_coord = sheet.vert_df.loc[midvert,['x','y']].to_numpy(dtype=float)
    principle_unit = end1_coord-mid_coord

    principle_unit = principle_unit/np.linalg.norm(principle_unit)
    
    # For each vertex in associated_vert, we compute the dot product and get
    # a dictionary, keys are the vertex ID and the values are the dot product.
    dot_dict = {}
    # Compute the unit vector of midvert -> associated.
    for i in associated_vert:
        temp_coord = sheet.vert_df.loc[i,['x','y']].to_numpy(dtype=float)
        vect_unit = temp_coord-mid_coord
        vect_unit = vect_unit/np.linalg.norm(vect_unit)
        dot_product = np.dot( principle_unit , vect_unit )
        dot_dict.update({i:dot_product})
    # Sort the dictionary by values, from the largest to lowest.
    dot_dict_sorted = dict(sorted(dot_dict.items(), key=lambda item: item[1], reverse=True))
    sorted_keys = list(dot_dict_sorted.keys()) 
    
    # Now, we use len(sorted_keys)//2 to determine which index should be consider
    # to stay at midvert.
    # Then, for all element which has index less than middle is resolved to the 
    # edge formed by midvert and end1. all element with index larger than middle
    # is resolved to the edge formed by midvert and end2.
    #
    # Logic for resolve in an edge.
    # mid_index: computed index that should stay, element_index: current element's index.
    # IF element_index == mid_index, THEN stay
    # IF element_index < mid_index, THEN consider the edge formed by end1 and midvert.
    # the distance between midvert and current element (ID of vertex) is d_sep*abs(mid_index - element_index)
    # IF element_index > mid_index, THEN consider the edge formed by end2 and midvert.
    # the distance between midvert and current element is then d_sep*abs(element_index-mid_index)
    
    # First, get the ID of the edge formed by midvert and end1.
    for i in sheet.edge_df.index:
        # Check for the edge formed by end1 and midvert
        if (sheet.edge_df.loc[i, 'srce'] == end1 and sheet.edge_df.loc[i, 'trgt'] == midvert) or \
           (sheet.edge_df.loc[i, 'srce'] == midvert and sheet.edge_df.loc[i, 'trgt'] == end1):
            edge1 = i
    
        # Check for the edge formed by end2 and midvert
        elif (sheet.edge_df.loc[i, 'srce'] == end2 and sheet.edge_df.loc[i, 'trgt'] == midvert) or \
             (sheet.edge_df.loc[i, 'srce'] == midvert and sheet.edge_df.loc[i, 'trgt'] == end2):
            edge2 = i
    
    middle_index = len(sorted_keys)//2
    for i in sorted_keys:
        element_index = sorted_keys.index(i)
        if element_index == middle_index:
            continue
        elif element_index < middle_index:
            edge_consider = edge1
            position = mid_coord + d_sep*(middle_index - element_index)*principle_unit
            new_vert = put_vert(sheet, edge_consider, position)[0]
            # Then update the relevant entries in the edge_df.
            for i in sheet.edge_df.index:
                if sheet.edge_df.loc[i,'srce'] == i:
                    sheet.edge_df.loc[i,'srce'] = new_vert
                elif sheet.edge_df.loc[i, 'trgt'] == i:
                    sheet.edge_df.loc[i,'trgt'] = new_vert
                else:
                    continue
        else:
            edge_consider = edge2
            position = mid_coord - d_sep*(middle_index - element_index)*principle_unit
            new_vert = put_vert(sheet, edge_consider, position)[0]
            # Then update the relevant entries in the edge_df.
            for i in sheet.edge_df.index:
                if sheet.edge_df.loc[i,'srce'] == i:
                    sheet.edge_df.loc[i,'srce'] = new_vert
                elif sheet.edge_df.loc[i, 'trgt'] == i:
                    sheet.edge_df.loc[i,'trgt'] = new_vert
                else:
                    continue
    
    # Then need to:
        # sheet.reset_index()
        # geom.update_all(sheet)




def T3_transition(sheet, edge, vert):
    """
    This is the final T3 transition function that is assembled from subfunctions
    defined above.
    
    Logic:
        For each boundary edge, use dist_computer to compute the distance and 
        
        If distance < d_min:
            determine adjacency:
                if adjacent:
                    new_vert = put_vert @ nearest [0]
                    merge_unconnected_verts(incoming_vert, new_vert)
                if not adjacent:
                    insert_into_edge(sheet, edge, incoming vert, position = nearest)
                    resolve_local
                    
                    
    """
    





"""
This is the end of the script.
"""
