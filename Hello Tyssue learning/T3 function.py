#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script defines all the sub-fuctions needed for my T3 transition.
Then I assemble them into a complete T3 transition main function.
"""

import decimal as dc




def case_classifier(edge, vert):
    """
    This function takes a pair of edge and vertex, returns their relative
    position as case number.

    Parameters
    ----------
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
    pass


def perturbate_T3(vert1, vert2):
    """
    This function should be used when case_classifier() returns case 1.
    
    This function takes two vertices. One of them is the incoming vertex, 
    the other one is the endpoint of the edge. Then perturbate their location 
    slightly.
    

    """
    
    pass




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

def form_edge(sheet, vert1, vert2):
    """
    This function constructs an edge between vert1 and vert2.

    Parameters
    ----------
    sheet: eptm object
    
    vert1 : Int
        ID of the vertex 1
    vert2 : Int
        ID of the vertex 2

    Returns
    -------
    Int, the ID of the newly formed edge.

    """
    
    edge_cols = sheet.edge_df()





"""
This is the end of the script.
"""
