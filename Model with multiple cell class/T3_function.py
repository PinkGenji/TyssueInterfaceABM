#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script defines all the sub-fuctions needed for my T3 transition.
Then I assemble them into a complete T3 transition main function.
"""

import numpy as np
import pandas as pd

from tyssue.topology.base_topology import collapse_edge, merge_vertices
from tyssue import PlanarGeometry as geom

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
    distance: If the closest point is one of the endpoints of the edge,
    then the distance is defined to be the Euclidean distance between the closer
    endpoint and the incoming vertex. If the closest point is between the two
    endpoints, then the distance is the Euclidean distance between the incoming
    vertex and the point between the endpoints.
    
    nearest: This is the coordinate of the nearest point from the incoming
    vertex to the edge.

    """
    # Extract the coordinate of the srce and trgt point.
    edge_end1 = sheet.edge_df.loc[edge,['srce']]
    edge_end2 = sheet.edge_df.loc[edge,['trgt']]
    end1_position = sheet.vert_df.loc[edge_end1, ['x','y']].to_numpy(dtype = float).flatten()
    end2_position = sheet.vert_df.loc[edge_end2, ['x','y']].to_numpy(dtype = float).flatten()
    # The line is from the end1 to end2.
    line = np.round(end2_position - end1_position,7)
    line_length = np.round(np.linalg.norm(line),7)
    line_unit = line /line_length
    
    # Now extract the coordinate of the point.
    point = sheet.vert_df.loc[vert,['x','y']].to_numpy(dtype=float).flatten()
    # adjust the coordinate of the point with regards to the srce.
    # then take the unit vector of it.
    srce_p = np.round(point-end1_position, 7)
    srce_p_scaled = srce_p/line_length
    dot = np.dot(srce_p_scaled, line_unit)
    if int(vert) not in sheet.vert_df.index:
        # vertex was removed by a topology event
        return np.inf, None
    elif dot <0:
        distance = np.round(np.linalg.norm(point-end1_position),7)
        nearest = end1_position + d_sep*line_unit
        return distance, nearest
    elif dot >1:
        distance = np.round(np.linalg.norm(point-end2_position),7)
        nearest = end2_position - d_sep*line_unit
        return distance, nearest
    else:
        nearest = end1_position + dot * line_unit
        distance = np.round(np.linalg.norm(nearest-point),7)
        return distance, nearest



def get_edge_id(sheet, vert1, vert2):
    """
    Returns the ID of the edge that connects the given two vertices.
    
    Parameters:
    - sheet: The sheet object containing edge_df (DataFrame with 'srce' and 'trgt' columns).
    - vert1: ID of the first vertex.
    - vert2: ID of the second vertex.
    
    Returns:
    - The edge ID if an edge exists between the two vertices.
    - None if no edge exists between vert1 and vert2.
    """
    # Find the edge where (vert1, vert2) or (vert2, vert1) are the endpoints
    edge = sheet.edge_df[
        ((sheet.edge_df['srce'] == vert1) & (sheet.edge_df['trgt'] == vert2)) |
        ((sheet.edge_df['srce'] == vert2) & (sheet.edge_df['trgt'] == vert1))
    ]
    
    if not edge.empty:
        return edge.index[0]  # Return the edge ID (index)
    
    return None  # Return None if no edge is found



def adjacency_check(sheet, edge, vert):
    """
    Returns the ID of the edge that connects the given vert to either of the endpoints of the specified edge.
    """
    end1_id = sheet.edge_df.loc[edge, 'srce']
    end2_id = sheet.edge_df.loc[edge, 'trgt']
    
    # Find adjacent edge to end1
    adjacent_edge_to_end1 = sheet.edge_df[
        ((sheet.edge_df['srce'] == end1_id) & (sheet.edge_df['trgt'] == vert)) |
        ((sheet.edge_df['srce'] == vert) & (sheet.edge_df['trgt'] == end1_id))
    ]
    
    if not adjacent_edge_to_end1.empty:
        return 1, adjacent_edge_to_end1.iloc[0].name
    
    # Find adjacent edge to end2
    adjacent_edge_to_end2 = sheet.edge_df[
        ((sheet.edge_df['srce'] == end2_id) & (sheet.edge_df['trgt'] == vert)) |
        ((sheet.edge_df['srce'] == vert) & (sheet.edge_df['trgt'] == end2_id))
    ]
    
    if not adjacent_edge_to_end2.empty:
        return 2, adjacent_edge_to_end2.iloc[0].name

    # Return None if no adjacent edge is found
    return None




def merge_unconnected_verts(sheet,vert1, vert2):
    """
    This function merges two vertices that are not originally connected by an edge.
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
    return collapse_edge(sheet, sheet.edge_df.index[-1], reindex=False, allow_two_sided=True)
    
    # Note: Then need to sheet.reset_index(), then geom.update_all(sheet).


    
    
    
def insert_into_edge(sheet, edge, vert, position):
    """
    This function inserts the vertex (vert) into the edge (edge), the insertion
    takes place at the coordinate specified by position, then update the relevant
    rows in edge df.
    
    First, we put a new vertex on the edge with cut_place coordinate.
    Then, we update all the entries of vert_id to the new vertex id.
    
    Notice: To remove the old vertex, we can use sheet.reset_index() afterwards.
    
    Parameters
    ----------
    sheet : Eptm instance
    
    edge : 
        ID of the edge.
    vert : 
        ID of the vertex.
    position:
        coordinate used to generate the new vertex.
        
    Returns
    ----------
    cut_vert: the ID of the newly created vertex.
    
    """
    
    # First, put a new vertex on the edge, the new vertex has ID, cut_id
    cut_vert, cut_edge, cut_op_edge = put_vert(sheet, edge, position)

    # Update the edge df entries, replace 'vert' by 'cut_vert'
    for i in sheet.edge_df.index:
        if sheet.edge_df.loc[i,'srce'] == vert:
            sheet.edge_df.loc[i,'srce'] = cut_vert
        if sheet.edge_df.loc[i, 'trgt'] == vert:
            sheet.edge_df.loc[i,'trgt'] = cut_vert
        else:
            continue
    return cut_vert
    # Need to follow a sheet.reset_index() to remove the old vertex.


def del_iso_vert(sheet):
    """
    This function removes isolated vertex without reindex.

    """
    
    # Identify the connected vertices by checking which vertices are in the edge source or target
    connected_vertices = set(sheet.edge_df.srce).union(sheet.edge_df.trgt)

    # Filter out the disconnected vertices by retaining only those in the connected_vertices set
    sheet.vert_df = sheet.vert_df.loc[sheet.vert_df.index.isin(connected_vertices)]

    # Filter the faces to remove those that refer to disconnected vertices
    sheet.face_df = sheet.face_df[
        sheet.face_df.apply(lambda row: all(vertex in connected_vertices for vertex in row), axis=1)
    ]



def resolve_local(sheet, end1, end2, midvert, d_sep):
    """
    end1, end2, midvert are IDs of vertices. Midvert is the middle vertex that
    connects both end1 and end2.
    
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
    print(f'associated vertices are: {sorted_keys} ')
    '''
    Now, we use len(sorted_keys)//2 to determine which index should be consider
    to stay at midvert.
    Then, for all element which has index less than middle is resolved to the 
    edge formed by midvert and end1. all element with index larger than middle
    is resolved to the edge formed by midvert and end2.
    
    Logic for resolve in an edge:
    mid_index: computed index that should stay, element_index: current element's index.
    IF element_index == mid_index, THEN stay
    IF element_index < mid_index, THEN consider the edge formed by end1 and midvert.
    the distance between midvert and current element (ID of vertex) is d_sep*abs(mid_index - element_index)
    IF element_index > mid_index, THEN consider the edge formed by end2 and midvert.
    the distance between midvert and current element is then d_sep*abs(element_index-mid_index)
    '''
    # First, get the ID of the edge formed by midvert and end1.
    # Initialize edge1 and edge2 to None, in case they are not found
    edge1 = None
    edge2 = None
    
    for i in sheet.edge_df.index:
        # Check for the edge formed by end1 and midvert
        if (sheet.edge_df.loc[i, 'srce'] == end1 and sheet.edge_df.loc[i, 'trgt'] == midvert) or \
           (sheet.edge_df.loc[i, 'srce'] == midvert and sheet.edge_df.loc[i, 'trgt'] == end1):
            edge1 = i
    
        # Check for the edge formed by end2 and midvert
        if (sheet.edge_df.loc[i, 'srce'] == end2 and sheet.edge_df.loc[i, 'trgt'] == midvert) or \
             (sheet.edge_df.loc[i, 'srce'] == midvert and sheet.edge_df.loc[i, 'trgt'] == end2):
            edge2 = i
    
    # Ensure both edges are found before proceeding
    if edge1 is None or edge2 is None:
        raise ValueError(f"Edges between {midvert} and {end1} or {midvert} and {end2} not found.")

    middle_index = len(sorted_keys) // 2
    print(f'middle index is: {middle_index}')
    
    
    for element_index, vertex_id in enumerate(sorted_keys):
        print(f'Now resolving vert {vertex_id}')
        if element_index == middle_index:
            continue

        if element_index < middle_index:
            print(f'element index: {element_index} < middle index')
            edge_consider = edge1
            position = mid_coord + d_sep * abs(middle_index - element_index) * principle_unit
            # Put the new vertex on the edge and update edge_df
            new_vert = put_vert(sheet, edge_consider, position)[0]
            print(f'put a new vertex: {new_vert}')
            for e_id in sheet.edge_df.index:
                if sheet.edge_df.loc[e_id, 'srce'] == vertex_id and sheet.edge_df.loc[e_id,'trgt'] == midvert:
                    sheet.edge_df.loc[e_id, 'trgt'] = new_vert
                if sheet.edge_df.loc[e_id, 'trgt'] == vertex_id and sheet.edge_df.loc[e_id, 'srce'] == midvert:
                    sheet.edge_df.loc[e_id, 'srce'] = new_vert
        if element_index > middle_index:
            print(f'element index: {element_index} > middle index')
            edge_consider = edge2
            position = mid_coord - d_sep * abs(element_index - middle_index) * principle_unit
            # Put the new vertex on the edge and update edge_df
            new_vert = put_vert(sheet, edge_consider, position)[0]
            print(f'put a new vertex: {new_vert}')
            for e_id in sheet.edge_df.index:
                if sheet.edge_df.loc[e_id, 'srce'] == vertex_id and sheet.edge_df.loc[e_id,'trgt'] == midvert:
                    sheet.edge_df.loc[e_id, 'trgt'] = new_vert
                if sheet.edge_df.loc[e_id, 'trgt'] == vertex_id and sheet.edge_df.loc[e_id, 'srce'] == midvert:
                    sheet.edge_df.loc[e_id, 'srce'] = new_vert
    
    # Then need to:
        # sheet.reset_index()
        # geom.update_all(sheet)




def resolve_local_adj(sheet, merged_vert, old_vert, d_sep):
    """
    Given the ID of the vertices that changed its position or stayed, I compute
    the unit vector from changed_vert to old_vert.
    Then put new vertices on the edge at least d_sep away.
    Then use the dot product trick, and update the relevant dataframes.
    ---------
    merged_vert: ID of the vertex as a consequence of merged vertices.
    old_vert: ID of the vertex that is not changed.
    
    
    """
    # Collect all the vertices that are connected to the vertex.
    associated_vert = set()
    for i in sheet.edge_df.index:
        srce = sheet.edge_df.loc[i, 'srce']
        trgt = sheet.edge_df.loc[i, 'trgt']
        if srce == merged_vert and trgt != old_vert:
            associated_vert.add(trgt)
        elif trgt == merged_vert and srce != old_vert:
            associated_vert.add(srce)

    # Use to get a principle unit vector, arrow from merged_vert to old_vert.
    old_coord = sheet.vert_df.loc[old_vert,['x','y']].to_numpy(dtype=float)
    merged_coord = sheet.vert_df.loc[merged_vert,['x','y']].to_numpy(dtype=float)
    principle_unit = old_coord - merged_coord
    principle_unit = principle_unit/np.linalg.norm(principle_unit)

    # For each vertex in associated_vert, we compute the dot product and get
    # a dictionary, keys are the vertex ID and the values are the dot product.
    dot_dict = {}
    # Compute the unit vector of merged_vert -> associated.
    for i in associated_vert:
        temp_coord = sheet.vert_df.loc[i,['x','y']].to_numpy(dtype=float)
        vect_unit = temp_coord-merged_coord
        vect_unit = vect_unit/np.linalg.norm(vect_unit)
        dot_product = np.dot( principle_unit , vect_unit )
        dot_dict.update({i:dot_product})
    # Sort the dictionary by values, from the largest to lowest.
    dot_dict_sorted = dict(sorted(dot_dict.items(), key=lambda item: item[1], reverse=True))
    sorted_keys = list(dot_dict_sorted.keys()) 
    print(f'sorted_keys: {sorted_keys}')
    
    
    # Now, I have a sorted list that contains the associated vertices.
    # put_vert at the positions that based on the element_index.
    # the distance from the changed_vert can be calculated by the rule: 
    # [len(sorted_keys)-1-element_index] * d_sep.
    
    # First, we need to know which edge is connecting id_kept and the old_vert.
    edge_between = get_edge_id(sheet, merged_vert , old_vert )
    for element_index, vertex_id in enumerate(sorted_keys):
        # Put the new vertex on the edge and update edge_df
        position = merged_coord + principle_unit*d_sep*(len(sorted_keys) -1 -element_index )
        new_vert = put_vert(sheet, edge_between, position)[0]
        for e_id in sheet.edge_df.index:
            if sheet.edge_df.loc[e_id, 'srce'] == vertex_id and sheet.edge_df.loc[e_id,'trgt'] == merged_vert:
                sheet.edge_df.loc[e_id, 'trgt'] = new_vert
            if sheet.edge_df.loc[e_id, 'trgt'] == vertex_id and sheet.edge_df.loc[e_id,'srce'] == merged_vert :
                sheet.edge_df.loc[e_id, 'srce'] = new_vert





def T3_swap(sheet, edge_collide, vert_incoming, nearest_coord, d_sep):
    """
    This is the final T3 transition function that is assembled using functions defined earlier in this script.
    
    Presumption: I assume that I have used dist_computer() function to compute the distance between a pair of vertex and
    edge already.
    If distance < d_min, then I will call is T3_transition function to perform a T3 transition.
    
    Logic:
        
        If distance < d_min:
            determine adjacency:
                if adjacent:
                    new_vert = put_vert @ nearest [0]
                    merge_unconnected_verts(incoming_vert, new_vert)
                if not adjacent:
                    insert_into_edge(sheet, edge, incoming vert, position = nearest)
                    resolve_local

    """
    # First, determine the adjacency.
    result = adjacency_check(sheet, edge_collide, vert_incoming)
    if result is None:
        print("No adjacent edge found, proceeding with non-adjacent logic.")
        endpoint1 = sheet.edge_df.loc[edge_collide,'srce'] 
        endpoint2 = sheet.edge_df.loc[edge_collide,'trgt']
        print(f'end1: {endpoint1}, end2: {endpoint2}')
        middle_vertex = insert_into_edge(sheet, edge_collide, vert_incoming, nearest_coord)
        resolve_local(sheet, endpoint1, endpoint2, middle_vertex, d_sep)
    
    else:
        adj_check, edge_connection = result
        print('adjacent')
        # If it's adjacent, then move the connected endpoint to nearest.
        # Then merge the incoming vert with the connected endpoint.
        # Merge means: the two vertices are merged into one, at the middle
        # point of the edge that connected them.
        ep1 = sheet.edge_df.loc[edge_collide,'srce'] 
        ep2 = sheet.edge_df.loc[edge_collide,'trgt']
        # If adj_check is 1, then vert_incoming is connected to srce.
        # If adj_check is 2, then vert_incoming is connected to trgt.
    
        if adj_check == 2: # Then it is connected to endpoint2 
            sheet.vert_df.loc[ep2,'x'] = nearest_coord[0]
            sheet.vert_df.loc[ep2, 'y'] = nearest_coord[1]
            print(f'changed position of vert {ep2}')
            # The id of the vertex after collapse is the smaller id of the vertices.
            id_kept = min(sheet.edge_df.loc[edge_connection,['srce','trgt']] )
            collapse_edge(sheet, edge_connection, reindex=False, allow_two_sided=False)
            # The new edge is formed by id_kept and ep1.
            edge_new = get_edge_id(sheet, id_kept, ep1)
            #resolve_local_adj(sheet, id_kept, ep1, d_sep)
            
        if adj_check == 1: # Then it is connected to endpoint1.
            sheet.vert_df.loc[ep1,'x'] = nearest_coord[0]
            sheet.vert_df.loc[ep1, 'y'] = nearest_coord[1]
            print(f'changed position of vert {ep1}')
            id_kept = min(sheet.edge_df.loc[edge_connection,['srce','trgt']] )
            collapse_edge(sheet, edge_connection, reindex=False, allow_two_sided=False)
            # The new edge is formed by id_kept and ep2.
            edge_new = get_edge_id(sheet, id_kept, ep2)
            # Then resolve local.
            #resolve_local_adj(sheet, id_kept, ep2,  d_sep)

    # sheet.reset_index()
    # geom.update_all(sheet)





"""
This is the end of the script.
"""
