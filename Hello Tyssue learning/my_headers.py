# -*- coding: utf-8 -*-
"""
This script contains all my personal defined functions to be used.
"""
import numpy as np

from tyssue.topology.base_topology import add_vert
from tyssue.topology.sheet_topology import face_division
from tyssue import PlanarGeometry as geom


def delete_face(sheet_obj, face_deleting):
    """

    
    Parameters
    ----------
    sheet_obj : Epithelium
        An Epithelium 'Sheet' object from Tyssue.
    face_deleting : Int
        The index of the face to be deleted.

    Returns
    -------
    A Pandas Data Frame that deletes the face, with border edges are single
    arrowed, without index resetting.

    """
    # Compute all edges associated with the face, then drop these edges in df.
    associated_edges = sheet_obj.edge_df[sheet_obj.edge_df['face'] == face_deleting]
    sheet_obj.edge_df.drop(associated_edges.index, inplace = True)
    
    # All associated edges are removed, now remove the 'empty' face and reindex.
    sheet_obj.face_df.drop(face_deleting , inplace =True)


def xprod_2d(vec1, vec2):
    """
    

    Parameters
    ----------
    vec1 : Iterable
        First vector
    vec2 : Iterable
        Second vector

    Returns
    -------
    A scalar value that is the 2D cross product,
    equivalent to setting z=0 in 3D.

    """
    scalar = vec1[0]*vec2[1] - vec1[1]*vec2[0]
    return scalar


def put_vert(eptm, edge, coord_put):
    """Adds a vertex somewhere in the an edge,

    which is split as is its opposite(s)

    Parameters
    ----------
    eptm : a :class:`Epithelium` instance
    edge : int
    the index of one of the half-edges to split
    coord_put: list
    the coordinate of the new vertex to be put

    Returns
    -------
    new_vert : int
    the index to the new vertex
    new_edges : int or list of ints
    index to the new edge(s). For a sheet, returns
    a single index, for a 3D epithelium, returns
    the list of all the new parallel edges
    new_opp_edges : int or list of ints
    index to the new opposite edge(s). For a sheet, returns
    a single index, for a 3D epithelium, returns
    the list of all the new parallel edges


    In the simple case whith two half-edge, returns
    indices to the new edges, with the following convention:

    s    e    t
      ------>
    * <------ *
    oe

    s    e       ne   t
      ------   ----->
    * <----- * ------ *
        oe   nv   noe

    where "e" is the passed edge as argument, "s" its source "t" its
    target and "oe" its opposite. The returned edges are the ones
    between the new vertex and the input edge's original target.
    """

    srce, trgt = eptm.edge_df.loc[edge, ["srce", "trgt"]]
    opposites = eptm.edge_df[
        (eptm.edge_df["srce"] == trgt) & (eptm.edge_df["trgt"] == srce)
    ]
    parallels = eptm.edge_df[
        (eptm.edge_df["srce"] == srce) & (eptm.edge_df["trgt"] == trgt)
    ]

    new_vert = eptm.vert_df.loc[srce:srce]
    eptm.vert_df = eptm.vert_df.append(new_vert, ignore_index=True)
    new_vert = eptm.vert_df.index[-1]
    eptm.vert_df.loc[new_vert, eptm.coords] = coord_put
    new_edges = []

    for p, p_data in parallels.iterrows():
        eptm.edge_df.loc[p, "trgt"] = new_vert
        eptm.edge_df = eptm.edge_df.append(p_data, ignore_index=True)
        new_edge = eptm.edge_df.index[-1]
        eptm.edge_df.loc[new_edge, "srce"] = new_vert
        eptm.edge_df.loc[new_edge, "trgt"] = trgt
        new_edges.append(new_edge)

    new_opp_edges = []
    for o, o_data in opposites.iterrows():
        eptm.edge_df.loc[o, "srce"] = new_vert
        eptm.edge_df = eptm.edge_df.append(o_data, ignore_index=True)
        new_opp_edge = eptm.edge_df.index[-1]
        eptm.edge_df.loc[new_opp_edge, "trgt"] = new_vert
        eptm.edge_df.loc[new_opp_edge, "srce"] = trgt
        new_opp_edges.append(new_opp_edge)

    # ## Sheet special case
    if len(new_edges) == 1:
        new_edges = new_edges[0]
    if len(new_opp_edges) == 1:
        new_opp_edges = new_opp_edges[0]
    elif len(new_opp_edges) == 0:
        new_opp_edges = None
    return new_vert, new_edges, new_opp_edges


def divisibility_check(eptm, cell_id):
    """
    

    Parameters
    ----------
    eptm : epithelium object
        
    cell_id : int
        The index of the cell being checked

    Returns
    -------
    Boolean, True for 'can divide', False for 'cannot divide'

    """
    eptm.get_opposite()
    if any(eptm.edge_df[eptm.edge_df.loc[:,'face'] == cell_id].loc[:,'opposite']==-1) == True:
        return True
    else:
        return False



def lateral_split(eptm, mother):
    """
    Split the cell by choosing one of the edges to be a basal edge.

    Parameters
    ----------
    eptm : a: Class: 'Epithelium' instance

    mother : int
        the index of the mother cell.

    Returns
    -------
    daughter: face index of new cell.

    """
    edge_in_cell = eptm.edge_df[eptm.edge_df.loc[:,'face'] == mother]
    # Obtain the index for one of the basal edges.
    basal_edges = edge_in_cell[ edge_in_cell.loc[:,'opposite']==-1 ]
    basal_edge_index = basal_edges.index[np.random.randint(0,len(basal_edges))]
    #get the vertex index of the newly added mid point.
    basal_mid = add_vert(eptm, edge = basal_edge_index)[0]
    geom.update_all(eptm)

    # re-rewite the edge_in_cell to include the new vertex.
    edge_in_cell = eptm.edge_df[eptm.edge_df.loc[:,'face'] == mother]
    condition = edge_in_cell.loc[:,'srce'] == basal_mid
    # extract the x-coordiante from array, then convert to a float type.
    
    # Extract the centre vertex.
    c0x = float(edge_in_cell[condition].loc[:,'fx'].values[0])
    c0y = float(edge_in_cell[condition].loc[:,'fy'].values[0])
    c0 = [c0x, c0y]
    cent_dict = {'y': c0y, 'is_active': 1, 'x': c0x}
    eptm.vert_df = eptm.vert_df.append(cent_dict, ignore_index = True)
    # The append function adds the new row in the last row, we the use iloc to 
    # get the index of the last row, hence the index of the centre point.
    cent_index = eptm.vert_df.index[-1]
    
    # Extract for source vertex coordinates
    p0x = float(edge_in_cell[condition].loc[:,'sx'].values[0])
    p0y = float(edge_in_cell[condition].loc[:,'sy'].values[0])
    p0 = [p0x, p0y]

    # Extract the directional vector.
    rx = float(edge_in_cell[condition].loc[:,'rx'].values[0])
    ry = float(edge_in_cell[condition].loc[:,'ry'].values[0])
    r  = [-rx, -ry]   # use the line in opposite direction.
    
    # We need to use iterrows to iterate over rows in pandas df
    # The iteration has the form of (index, series)
    # The series can be sliced.
    for index, row in edge_in_cell.iterrows():
        s0x = row['sx']
        s0y = row['sy']
        t0x = row['tx']
        t0y = row['ty']
        v1 = [s0x-p0x,s0y-p0y]
        v2 = [t0x-p0x,t0y-p0y]
        # if the xprod_2d returns negative, then line intersects the line segment.
        if xprod_2d(r, v1)*xprod_2d(r, v2) < 0:
            #print(f'The edge that is intersecting is: {index}')
            dx = row['dx']
            dy = row['dy']
            c1 = (dx*ry/rx)-dy
            c2 = s0y-p0y - (s0x*ry/rx) + (p0x*ry/rx)
            k=c2/c1
            intersection = [s0x+k*dx, s0y+k*dy]
            oppo_index = int(put_vert(eptm, index, intersection)[0])
    # Do face division
    new_face_index = face_division(eptm, mother = mother, vert_a = basal_mid, vert_b = oppo_index )
    # Put a vertex at the centroid, on the newly formed edge (last row in df).
    put_vert(eptm, edge = eptm.edge_df.index[-1], coord_put = c0)
    eptm.update_num_sides()
    return new_face_index
    #second_half = face_division(eptm, mother = mother, vert_a = oppo_index, vert_b = cent_index)
    #print(f'The new edge has first half as: {first_half} and second half as: {second_half} ')


def lateral_division(sheet, manager, cell_id, division_rate):
    """Defines a lateral division behavior.
    The function is composed of:
        1. check if the cell is CT cell and ready to split.
        2. generate a random number from (0,1), and compare with a threshold.
        3. two daughter cells starts growing until reach a threshold.
        
    
    Parameters
    ----------
    sheet: a :class:`Sheet` object
    cell_id: int
        the index of the dividing cell
    crit_area: float
        the area at which 
    growth_speed: float
        increase in the area per unit time
        A_0(t + dt) = A0(t) + growth_speed
    """

    # if the cell area is larger than the crit_area, we let the cell divide.
    if sheet.face_df.loc[cell_id, "cell_type"] == 'CT' and sheet.face_df.loc[cell_id, 'division_status'] == 'ready':
        # A random float number is generated between (0,1)
        prob = np.random.uniform(0,1)
        if prob < division_rate:
            daughter = lateral_split(sheet, mother = cell_id)
            sheet.face_df.loc[cell_id,'growth_speed'] = (sheet.face_df.loc[cell_id,'prefered_area'] - sheet.face_df.loc[cell_id, 'area'])/5
            sheet.face_df.loc[cell_id, 'division_status'] = 'growing'
            sheet.face_df.loc[daughter,'growth_speed'] = (sheet.face_df.loc[daughter,'prefered_area'] - sheet.face_df.loc[daughter, 'area'])/5
            sheet.face_df.loc[daughter, 'division_status'] = 'growing'
            # update geometry
            geom.update_all(sheet)
        else:
            pass
            
    elif sheet.face_df.loc[cell_id, "cell_type"] == 'CT' and sheet.face_df.loc[cell_id, 'division_status'] == 'growing':
        sheet.face_df.loc[cell_id,'prefered_area'] = sheet.face_df.loc[cell_id,'area'] + sheet.face_df.loc[cell_id,'growth_speed']
        if sheet.face_df.loc[cell_id,'area'] <= sheet.face_df.loc[cell_id,'prefered_area']:
            # restore division_status and prefered_area
            sheet.face_df.loc[cell_id, 'division_status'] = 'ready'
            sheet.face_df.loc[cell_id, "prefered_area"] = 1.0
    else:
        pass
        







""" This is the end of the script. """
