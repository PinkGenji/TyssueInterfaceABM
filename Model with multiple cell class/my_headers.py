# -*- coding: utf-8 -*-
"""
This script contains all my personal defined functions to be used.
"""
import numpy as np
import pandas as pd
import math
from collections import Counter
from decimal import Decimal

from tyssue.topology.sheet_topology import type1_transition
from tyssue.topology.base_topology import add_vert, drop_two_sided_faces
from tyssue.topology.sheet_topology import face_division
from tyssue import PlanarGeometry as geom
from tyssue.dynamics.planar_vertex_model import PlanarModel as model


def dot(v, w):
    x, y = v
    X, Y = w
    return x * X + y * Y


def length(v):
    x, y = v
    return math.sqrt(x * x + y * y)


def vector(b, e):
    """
    Creates a vector: b-e (arrow from e to b), stored as a tuple.

    """
    x, y = b
    X, Y = e
    return (round(X - x, 5), round(Y - y, 5))


def unit(v):
    x, y = v
    mag = length(v)
    return (x / mag, y / mag)


def distance(p0, p1):
    return length(vector(p0, p1))


def scale(v, sc):
    x, y = v
    return (x * sc, y * sc)


def add(v, w):
    x, y = v
    X, Y = w
    return (x + X, y + Y)


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
    sheet_obj.edge_df.drop(associated_edges.index, inplace=True)

    # All associated edges are removed, now remove the 'empty' face and reindex.
    sheet_obj.face_df.drop(face_deleting, inplace=True)


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
    scalar = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    return scalar


def closest_pair_dist(a, end1, end2):
    towards_end1 = distance(end1, a)
    towards_end2 = distance(end2, a)
    if towards_end1 < towards_end2:
        a = vector(a, end1)
        a_hat = a / round(np.linalg.norm(a), 4)
        return a_hat
    elif towards_end1 > towards_end2:
        a = vector(a, end2)
        a_hat = a / round(np.linalg.norm(a), 4)
        return a_hat


def put_vert(eptm, edge, coord_put):
    """Adds a vertex somewhere in the edge,

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
    eptm.vert_df = pd.concat([eptm.vert_df, new_vert], ignore_index=True)
    new_vert = eptm.vert_df.index[-1]
    eptm.vert_df.loc[new_vert, eptm.coords] = coord_put

    eptm.edge_df.loc[parallels.index, 'trgt'] = new_vert
    eptm.edge_df = pd.concat([eptm.edge_df, parallels], ignore_index=True)
    new_edges = eptm.edge_df.index[-parallels.index.size:]
    eptm.edge_df.loc[new_edges, 'srce'] = new_vert
    eptm.edge_df.loc[new_edges, 'trgt'] = trgt

    new_oppo_edges = []
    if len(opposites.index):
        eptm.edge_df.loc[opposites.index, 'srce'] = new_vert
        eptm.edge_df = pd.concat([eptm.edge_df, opposites], ignore_index=True)
        new_oppo_edges = eptm.edge_df.index[-opposites.index.size:]
        eptm.edge_df.loc[new_oppo_edges, 'trgt'] = new_vert
        eptm.edge_df.loc[new_oppo_edges, 'srce'] = trgt

    # ## Sheet special case
    if len(new_edges) == 1:
        new_edges = new_edges[0]
    if len(new_oppo_edges) == 1:
        new_opp_edges = new_oppo_edges[0]
    else:
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
    if any(eptm.edge_df[eptm.edge_df.loc[:, 'face'] == cell_id].loc[:, 'opposite'] == -1) == True:
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
    edge_in_cell = eptm.edge_df[eptm.edge_df.loc[:, 'face'] == mother]
    # Obtain the index for one of the basal edges.
    basal_edges = edge_in_cell[edge_in_cell.loc[:, 'opposite'] == -1]
    basal_edge_index = basal_edges.index[np.random.randint(0, len(basal_edges))]
    # get the vertex index of the newly added mid point.
    basal_mid = add_vert(eptm, edge=basal_edge_index)[0]
    geom.update_all(eptm)

    # re-rewite the edge_in_cell to include the new vertex.
    edge_in_cell = eptm.edge_df[eptm.edge_df.loc[:, 'face'] == mother]
    condition = edge_in_cell.loc[:, 'srce'] == basal_mid
    # extract the x-coordiante from array, then convert to a float type.

    # Extract the centre vertex.
    c0x = float(edge_in_cell[condition].loc[:, 'fx'].values[0])
    c0y = float(edge_in_cell[condition].loc[:, 'fy'].values[0])
    c0 = [c0x, c0y]
    cent_dict = {'y': c0y, 'is_active': 1, 'x': c0x}
    # Convert cent_dict into a DataFrame and concatenate it
    cent_df = pd.DataFrame([cent_dict])
    eptm.vert_df = pd.concat([eptm.vert_df, cent_df], ignore_index=True)
    cent_index = eptm.vert_df.index[-1]

    # Extract for source vertex coordinates
    p0x = float(edge_in_cell[condition].loc[:, 'sx'].values[0])
    p0y = float(edge_in_cell[condition].loc[:, 'sy'].values[0])
    p0 = [p0x, p0y]

    # Extract the directional vector.
    rx = float(edge_in_cell[condition].loc[:, 'rx'].values[0])
    ry = float(edge_in_cell[condition].loc[:, 'ry'].values[0])
    r = [-rx, -ry]  # use the line in opposite direction.

    # We need to use iterrows to iterate over rows in pandas df
    # The iteration has the form of (index, series)
    # The series can be sliced.
    for index, row in edge_in_cell.iterrows():
        s0x = row['sx']
        s0y = row['sy']
        t0x = row['tx']
        t0y = row['ty']
        v1 = [s0x - p0x, s0y - p0y]
        v2 = [t0x - p0x, t0y - p0y]
        # if the xprod_2d returns negative, then line intersects the line segment.
        if xprod_2d(r, v1) * xprod_2d(r, v2) < 0:
            # print(f'The edge that is intersecting is: {index}')
            dx = row['dx']
            dy = row['dy']
            c1 = (dx * ry / rx) - dy
            c2 = s0y - p0y - (s0x * ry / rx) + (p0x * ry / rx)
            k = c2 / c1
            intersection = [s0x + k * dx, s0y + k * dy]
            oppo_index = int(put_vert(eptm, index, intersection)[0])
    # Do face division
    new_face_index = face_division(eptm, mother=mother, vert_a=basal_mid, vert_b=oppo_index)
    # Put a vertex at the centroid, on the newly formed edge (last row in df).
    put_vert(eptm, edge=eptm.edge_df.index[-1], coord_put=c0)
    eptm.update_num_sides()
    return new_face_index
    # second_half = face_division(eptm, mother = mother, vert_a = oppo_index, vert_b = cent_index)
    # print(f'The new edge has first half as: {first_half} and second half as: {second_half} ')


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
        prob = np.random.uniform(0, 1)
        if prob < division_rate:
            daughter = lateral_split(sheet, mother=cell_id)
            sheet.face_df.loc[cell_id, 'growth_speed'] = (sheet.face_df.loc[cell_id, 'prefered_area'] -
                                                          sheet.face_df.loc[cell_id, 'area']) / 5
            sheet.face_df.loc[cell_id, 'division_status'] = 'growing'
            sheet.face_df.loc[daughter, 'growth_speed'] = (sheet.face_df.loc[daughter, 'prefered_area'] -
                                                           sheet.face_df.loc[daughter, 'area']) / 5
            sheet.face_df.loc[daughter, 'division_status'] = 'growing'
            # update geometry
            geom.update_all(sheet)
        else:
            pass

    elif sheet.face_df.loc[cell_id, "cell_type"] == 'CT' and sheet.face_df.loc[cell_id, 'division_status'] == 'growing':
        sheet.face_df.loc[cell_id, 'prefered_area'] = sheet.face_df.loc[cell_id, 'area'] + sheet.face_df.loc[
            cell_id, 'growth_speed']
        if sheet.face_df.loc[cell_id, 'area'] <= sheet.face_df.loc[cell_id, 'prefered_area']:
            # restore division_status and prefered_area
            sheet.face_df.loc[cell_id, 'division_status'] = 'ready'
            sheet.face_df.loc[cell_id, "prefered_area"] = 1.0
    else:
        pass


def T1_check(eptm, threshold, scale):
    for i in eptm.sgle_edges:
        if eptm.edge_df.loc[i, 'length'] < threshold:
            type1_transition(eptm, edge01=i, multiplier=scale)
            print(f'Type 1 transition applied to edge {i} \n')
        else:
            continue


def my_ode(eptm):
    valid_verts = eptm.active_verts[eptm.active_verts.isin(eptm.vert_df.index)]
    grad_U = model.compute_gradient(eptm).loc[valid_verts]
    dr_dt = -grad_U.values / eptm.vert_df.loc[valid_verts, 'viscosity'].values[:, None]
    return dr_dt


def collapse_edge(sheet, edge, reindex=True, allow_two_sided=False):
    """Collapses edge and merges it's vertices, creating (or increasing the rank of)
    a rosette structure.

    If `reindex` is `True` (the default), resets indexes and topology data.
    The edge is collapsed on the smaller of the srce, trgt indexes
    (to minimize reindexing impact)

    Returns the index of the collapsed edge's remaining vertex (its srce)

    """

    srce, trgt = np.sort(sheet.edge_df.loc[edge, ["srce", "trgt"]]).astype(int)

    # edges = sheet.edge_df[
    #     ((sheet.edge_df["srce"] == srce) & (sheet.edge_df["trgt"] == trgt))
    #     | ((sheet.edge_df["srce"] == trgt) & (sheet.edge_df["trgt"] == srce))
    # ]

    # has_3_sides = np.any(
    #     sheet.face_df.loc[edges["face"].astype(int), "num_sides"] < 4
    # )
    # if has_3_sides and not allow_two_sided:
    #     warnings.warn(
    #         f"Collapsing edge {edge} would result in a two sided face, aborting"
    #     )
    #     return -1

    sheet.vert_df.loc[srce, sheet.coords] = sheet.vert_df.loc[
        [srce, trgt], sheet.coords
    ].mean(axis=0)
    sheet.vert_df.drop(trgt, axis=0, inplace=True)
    # rewire
    sheet.edge_df.replace({"srce": trgt, "trgt": trgt}, srce, inplace=True)
    # all the edges parallel to the original
    collapsed = sheet.edge_df.query("srce == trgt")
    sheet.edge_df.drop(collapsed.index, axis=0, inplace=True)
    return srce

def type1_transition_custom(sheet, edge01, multiplier=1.5):
    """A safe custom T1 transition that avoids invalid topology and preserves Tyssue structure."""
    import numpy as np

    # Get source, target and associated face
    srce, trgt, face = sheet.edge_df.loc[edge01, ["srce", "trgt", "face"]].astype(int)

    # Get opposite edge
    opp_edge = sheet.edge_df.loc[edge01, "opposite"]
    if opp_edge == -1:
        # Do not perform T1 on boundary edge
        return

    # Get opposite face and vertices
    opp_face = sheet.edge_df.loc[opp_edge, "face"]
    srce_opp, trgt_opp = sheet.edge_df.loc[opp_edge, ["trgt", "srce"]].astype(int)

    # Coordinates before collapse
    coord_1 = sheet.vert_df.loc[srce, sheet.coords].values
    coord_2 = sheet.vert_df.loc[trgt, sheet.coords].values
    midpoint = (coord_1 + coord_2) / 2

    # Create two new vertices offset from the midpoint
    dir_vec = coord_2 - coord_1
    if np.linalg.norm(dir_vec) < 1e-10:
        return  # Avoid division by zero if edge already collapsed
    normal_vec = np.array([-dir_vec[1], dir_vec[0]])  # 90-degree rotation for 2D normal
    normal_vec = normal_vec / np.linalg.norm(normal_vec)

    offset = multiplier * sheet.settings.get("threshold_length", 1.0) / 2
    v1_coords = midpoint + offset * normal_vec
    v2_coords = midpoint - offset * normal_vec

    # Create new vertices
    new_v1 = sheet.vert_df.index.max() + 1
    new_v2 = new_v1 + 1
    sheet.vert_df.loc[new_v1] = [*v1_coords, 1]
    sheet.vert_df.loc[new_v2] = [*v2_coords, 1]

    # Redirect old edges that pointed to srce/trgt to new_v1/new_v2
    def rewire_face_edges(face, old_v, new_v):
        face_edges = sheet.edge_df[sheet.edge_df['face'] == face]
        for idx in face_edges.index:
            if sheet.edge_df.loc[idx, 'srce'] == old_v:
                sheet.edge_df.loc[idx, 'srce'] = new_v
            if sheet.edge_df.loc[idx, 'trgt'] == old_v:
                sheet.edge_df.loc[idx, 'trgt'] = new_v

    rewire_face_edges(face, srce, new_v1)
    rewire_face_edges(face, trgt, new_v1)
    rewire_face_edges(opp_face, srce, new_v2)
    rewire_face_edges(opp_face, trgt, new_v2)

    # Replace the collapsed edge with a new edge between v1 and v2
    sheet.edge_df.loc[edge01, ['srce', 'trgt']] = [new_v1, new_v2]
    sheet.edge_df.loc[edge01, 'face'] = face
    sheet.edge_df.loc[edge01, 'is_active'] = 1

    # Update the opposite edge
    sheet.edge_df.loc[opp_edge, ['srce', 'trgt']] = [new_v2, new_v1]
    sheet.edge_df.loc[opp_edge, 'face'] = opp_face
    sheet.edge_df.loc[opp_edge, 'is_active'] = 1

    # Update opposite relationships
    sheet.edge_df.loc[edge01, 'opposite'] = opp_edge
    sheet.edge_df.loc[opp_edge, 'opposite'] = edge01

    return edge01



def division_mt(sheet, rng, cent_data, cell_id):
    """
    This division function invovles mitosis index.
    The cells keep growing, when the area exceeds a critical area, then
    the cell divides.

    Parameters
    ----------
    sheet: a :class:`Sheet` object
    cell_id: int
        the index of the dividing cell
    """
    condition = sheet.edge_df.loc[:, 'face'] == cell_id
    edge_in_cell = sheet.edge_df[condition]
    # We need to randomly choose one of the edges in cell 2.
    chosen_index = rng.choice(list(edge_in_cell.index))
    # Extract and store the centroid coordinate.
    c0x = float(cent_data.loc[cent_data['face'] == cell_id, ['fx']].values[0])
    c0y = float(cent_data.loc[cent_data['face'] == cell_id, ['fy']].values[0])
    c0 = [c0x, c0y]

    # Add a vertex in the middle of the chosen edge.
    new_mid_index = add_vert(sheet, edge=chosen_index)[0]
    # Extract for source vertex coordinates of the newly added vertex.
    p0x = sheet.vert_df.loc[new_mid_index, 'x']
    p0y = sheet.vert_df.loc[new_mid_index, 'y']
    p0 = [p0x, p0y]

    # Compute the directional vector from new_mid_point to centroid.
    rx = c0x - p0x
    ry = c0y - p0y
    r = [rx, ry]  # use the line in opposite direction.
    # We need to use iterrows to iterate over rows in pandas df
    # The iteration has the form of (index, series)
    # The series can be sliced.
    for index, row in edge_in_cell.iterrows():
        s0x = row['sx']
        s0y = row['sy']
        t0x = row['tx']
        t0y = row['ty']
        v1 = [s0x - p0x, s0y - p0y]
        v2 = [t0x - p0x, t0y - p0y]
        # if the xprod_2d returns negative, then line intersects the line segment.
        if xprod_2d(r, v1) * xprod_2d(r, v2) < 0 and index != chosen_index:
            dx = row['dx']
            dy = row['dy']
            c1 = dx * ry - dy * rx
            c2 = s0y * rx - p0y * rx - s0x * ry + p0x * ry
            k = c2 / c1
            intersection = [s0x + k * dx, s0y + k * dy]
            oppo_index = put_vert(sheet, index, intersection)[0]
            # Split the cell with a line.
            new_face_index = face_division(sheet, mother=cell_id, vert_a=new_mid_index, vert_b=oppo_index)
            # Put a vertex at the centroid, on the newly formed edge (last row in df).
            cent_index = put_vert(sheet, edge=sheet.edge_df.index[-1], coord_put=c0)[0]
            # Draw two random numbers from uniform distribution [10,15] for mitosis cycle duration.
            # random_int_1 = rng.integers(10000, 15000) / 1000
            # random_int_2 = rng.integers(10000, 15000) / 1000
            # # Assign mitosis cycle duration to the two daughter cells.
            # sheet.face_df.loc[cell_id,'T_cycle'] = Decimal(random_int_1)
            # sheet.face_df.loc[new_face_index,'T_cycle'] = Decimal(random_int_2)

            # Following lines are commented out: Instead of using a new variable, I will minus T_cycle after each dt step by dt.
            # sheet.face_df.loc[cell_id, 'T_age'] = dt
            # sheet.face_df.loc[new_face_index,'T_age'] = dt

            print(f'cell {cell_id} is divided, dauther cell {new_face_index} is created.')
            return new_face_index



def division_mt_ver2(sheet, rng, cent_data, cell_id):
    """
    This division function invovles mitosis index.
    The cells keep growing, when the area exceeds a critical area, then
    the cell divides.

    Parameters
    ----------
    sheet: a :class:`Sheet` object
    cell_id: int
        the index of the dividing cell
    """
    condition = sheet.edge_df.loc[:, 'face'] == cell_id
    edge_in_cell = sheet.edge_df[condition]
    # We need to randomly choose one of the edges in cell 2.
    chosen_index = rng.choice(list(edge_in_cell.index))
    # Extract and store the centroid coordinate.
    c0x = float(cent_data.loc[cent_data['face'] == cell_id, ['fx']].values[0])
    c0y = float(cent_data.loc[cent_data['face'] == cell_id, ['fy']].values[0])
    c0 = [c0x, c0y]

    # Add a vertex in the middle of the chosen edge.
    new_mid_index = add_vert(sheet, edge=chosen_index)[0]
    # Extract for source vertex coordinates of the newly added vertex.
    p0x = sheet.vert_df.loc[new_mid_index, 'x']
    p0y = sheet.vert_df.loc[new_mid_index, 'y']
    p0 = [p0x, p0y]

    # Compute the directional vector from new_mid_point to centroid.
    rx = c0x - p0x
    ry = c0y - p0y
    r = [rx, ry]  # use the line in opposite direction.
    # We need to use iterrows to iterate over rows in pandas df
    # The iteration has the form of (index, series)
    # The series can be sliced.
    for index, row in edge_in_cell.iterrows():
        s0x = row['sx']
        s0y = row['sy']
        t0x = row['tx']
        t0y = row['ty']
        v1 = [s0x - p0x, s0y - p0y]
        v2 = [t0x - p0x, t0y - p0y]
        # if the xprod_2d returns negative, then line intersects the line segment.
        if xprod_2d(r, v1) * xprod_2d(r, v2) < 0 and index != chosen_index:
            dx = row['dx']
            dy = row['dy']
            c1 = dx * ry - dy * rx
            c2 = s0y * rx - p0y * rx - s0x * ry + p0x * ry
            k = c2 / c1
            intersection = [s0x + k * dx, s0y + k * dy]
            oppo_index = put_vert(sheet, index, intersection)[0]
            # Split the cell with a line.
            new_face_index = face_division(sheet, mother=cell_id, vert_a=new_mid_index, vert_b=oppo_index)
            # Put a vertex at the centroid, on the newly formed edge (last row in df).
            cent_index = put_vert(sheet, edge=sheet.edge_df.index[-1], coord_put=c0)[0]
            # Draw two random numbers from uniform distribution [10,15] for mitosis cycle duration.
            # random_int_1 = rng.integers(10000, 15000) / 1000
            # random_int_2 = rng.integers(10000, 15000) / 1000
            # # Assign mitosis cycle duration to the two daughter cells.
            # sheet.face_df.loc[cell_id,'T_cycle'] = Decimal(random_int_1)
            # sheet.face_df.loc[new_face_index,'T_cycle'] = Decimal(random_int_2)

            # Following lines are commented out: Instead of using a new variable, I will minus T_cycle after each dt step by dt.
            # sheet.face_df.loc[cell_id, 'T_age'] = dt
            # sheet.face_df.loc[new_face_index,'T_age'] = dt

            print(f'cell {cell_id} is divided, dauther cell {new_face_index} is created.')
            return new_face_index
        # === Fallback strategy if no intersecting edge is found ===
        # Sort edges by angle w.r.t. vector to centroid, pick the most orthogonal one
    print(f"⚠️ No intersection found for cell {cell_id}, applying fallback division.")
    vectors = []
    for idx, row in edge_in_cell.iterrows():
        sx, sy, tx, ty = row['sx'], row['sy'], row['tx'], row['ty']
        mx, my = (sx + tx) / 2, (sy + ty) / 2
        edge_vec = [tx - sx, ty - sy]
        to_centroid = [c0x - mx, c0y - my]
        norm = np.linalg.norm(edge_vec) * np.linalg.norm(to_centroid)
        if norm == 0:
            angle = 0
        else:
            dot = edge_vec[0] * to_centroid[0] + edge_vec[1] * to_centroid[1]
            angle = np.abs(dot / norm)  # closer to 0 = more perpendicular
        vectors.append((idx, angle))

    # Pick the most orthogonal edge (minimum cosine)
    fallback_edge_idx = sorted(vectors, key=lambda x: x[1])[0][0]
    fallback_vert = add_vert(sheet, edge=fallback_edge_idx)[0]

    fallback_coords = sheet.vert_df.loc[fallback_vert, ['x', 'y']].values.tolist()
    midpoint = [(p0x + fallback_coords[0]) / 2, (p0y + fallback_coords[1]) / 2]
    mid_vert = put_vert(sheet, edge=fallback_edge_idx, coord_put=midpoint)[0]

    new_face_index = face_division(sheet, mother=cell_id, vert_a=new_mid_index, vert_b=mid_vert)
    cent_index = put_vert(sheet, edge=sheet.edge_df.index[-1], coord_put=c0)[0]
    print(f'Fallback division succeeded: cell {cell_id} split, daughter {new_face_index} created.')
    return new_face_index

def time_step_bot(sheet, dt, max_dist_allowed):
    # Force computing and updating positions.
    valid_active_verts = sheet.active_verts[sheet.active_verts.isin(sheet.vert_df.index)]
    pos = sheet.vert_df.loc[valid_active_verts, sheet.coords].values
    # Compute the force with opposite of gradient direction.
    dot_r = my_ode(sheet)

    movement = dot_r * dt
    current_movement = np.linalg.norm(movement, axis=1)
    while max(current_movement, default=0) > max_dist_allowed:
        print('dt adjusted')
        dt /= 2
        movement = dot_r * dt
        current_movement = np.linalg.norm(movement, axis=1)
    return dt, movement


def boundary_nodes(sheet):
    """
    This returns a list of the vertex index that are boundary nodes.

    """
    boundary = set()
    for i in sheet.edge_df.index:
        if sheet.edge_df.loc[i, 'opposite'] == -1:
            boundary.add(sheet.edge_df.loc[i, 'srce'])
            boundary.add(sheet.edge_df.loc[i, 'trgt'])
    boudnary_vert = sheet.vert_df.loc[list(boundary), ['x', 'y']]
    return boudnary_vert


def T3_detection(sheet, edge_index, d_min):
    """
    This detects if an edge will collide with another vertex within a d_min
    distance for both node1 and node2 zones. It returns a Boolean.
    """
    # Get the nodes of the edge
    node1 = sheet.edge_df.loc[edge_index, 'srce']
    node2 = sheet.edge_df.loc[edge_index, 'trgt']

    # Compute the radius of the detection zones
    radius = (1 / 4 * sheet.edge_df.loc[edge_index, 'length'] ** 2 + d_min ** 2) ** 0.5

    # Get the coordinates of node1 and node2
    node1_coords = np.array(sheet.vert_df.loc[node1, ['x', 'y']])
    node2_coords = np.array(sheet.vert_df.loc[node2, ['x', 'y']])

    # Define the zones as dictionaries
    node1_zone = {
        'center': node1_coords,
        'radius': radius
    }
    node2_zone = {
        'center': node2_coords,
        'radius': radius
    }

    # Logic to check collision for both zones
    for idx, vertex in boundary_nodes(sheet).iterrows():
        if idx == node1 or idx == node2:
            continue  # Skip the nodes of the edge itself

        vertex_coords = np.array(vertex[['x', 'y']])

        # Check if the vertex is within node1's zone
        distance_to_node1 = np.linalg.norm(vertex_coords - node1_zone['center'])
        if distance_to_node1 < node1_zone['radius']:
            return True  # Collision detected in node1's zone

        # Check if the vertex is within node2's zone
        distance_to_node2 = np.linalg.norm(vertex_coords - node2_zone['center'])
        if distance_to_node2 < node2_zone['radius']:
            return True  # Collision detected in node2's zone

    return False  # No collision detected in either zone


def pnt2line(pnt, start, end):
    """
    Given a line with coordinates 'start' and 'end' and the
    coordinates of a point 'pnt' the proc returns the shortest
    distance from pnt to the line and the coordinates of the
    nearest point on the line.

    1  Convert the line segment to a vector ('line_vec').
    2  Create a vector connecting start to pnt ('pnt_vec').
    3  Find the length of the line vector ('line_len').
    4  Convert line_vec to a unit vector ('line_unitvec').
    5  Scale pnt_vec by line_len ('pnt_vec_scaled').
    6  Get the dot product of line_unitvec and pnt_vec_scaled ('t').
    7  Ensure t is in the range 0 to 1.
    8  Use t to get the nearest location on the line to the end
        of vector pnt_vec_scaled ('nearest').
    9  Calculate the distance from nearest to pnt_vec_scaled.
    10 Translate nearest back to the start/end line.
    Malcolm Kesson 16 Dec 2012

    Parameters
    ----------
    pnt : coordinate of the point

    start : coordinate of the starting point of the line segment
    end : coordinate of the end point of the line segment

    Returns
    -------
    tuple type. (dist, nearest)
    dist : float, distance between the point and line segment
    nearest : tuple of coordinates in floats of the nearest point on the line

    """
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0 / line_len)
    t = dot(line_unitvec, pnt_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return dist, nearest


def edge_extension(sheet, edge_id, total_extension):
    """
    Extends the edge equally on both ends of the line segment.

    Parameters
    ----------
    sheet : eptm instance
        Instance of the simulation object containing the cell sheet.
    edge_id : int
        Index of the edge to extend.
    total_extension : float
        Total length to add to the edge (split equally between source and target).

    Returns
    -------
    None.
    """
    import numpy as np  # Ensure NumPy is imported

    # Extract source and target vertex IDs
    srce_id, trgt_id = sheet.edge_df.loc[edge_id, ['srce', 'trgt']]

    # Extract source and target positions as numpy arrays
    srce = sheet.vert_df.loc[srce_id, ['x', 'y']].values
    trgt = sheet.vert_df.loc[trgt_id, ['x', 'y']].values
    # Compute the unit vector in the direction of the edge
    a = trgt - srce  # Vector from source to target
    a_hat = a / np.linalg.norm(a)  # Convert to unit vector (NumPy array)
    # Compute the extension vector
    extension = a_hat * total_extension / 2  # NumPy array supports elementwise operations

    # Update the source and target positions
    sheet.vert_df.loc[srce_id, ['x', 'y']] -= extension
    sheet.vert_df.loc[trgt_id, ['x', 'y']] += extension

    # Update geometry
    geom.update_all(sheet)


def adjacency_check(sheet, vert1, vert2):
    """
    Returns True if vert1 and vert2 are connected by an edge. Otherwise False
    """

    exists = sheet.edge_df[
        ((sheet.edge_df['srce'] == vert1) & (sheet.edge_df['trgt'] == vert2)) |
        ((sheet.edge_df['srce'] == vert2) & (sheet.edge_df['trgt'] == vert1))
        ].any().any()  # Checks if any rows satisfy the condition

    return exists  # Return True if such a row exists, False otherwise


def adjacent_vert(sheet, v, srce_id, trgt_id):
    adjacent = None
    if adjacency_check(sheet, v, srce_id) == True:
        adjacent = srce_id
    elif adjacency_check(sheet, v, trgt_id) == True:
        adjacent = trgt_id
    return adjacent


def are_vertices_in_same_face(sheet, vert1, vert2):
    # Find the faces where each vertex appears as either 'srce' or 'trgt'
    faces_vert1 = set(sheet.edge_df[sheet.edge_df['srce'] == vert1]['face']).union(
        sheet.edge_df[sheet.edge_df['trgt'] == vert1]['face']
    )
    faces_vert2 = set(sheet.edge_df[sheet.edge_df['srce'] == vert2]['face']).union(
        sheet.edge_df[sheet.edge_df['trgt'] == vert2]['face']
    )

    # Check if there is any intersection between the faces of the two vertices
    return bool(faces_vert1.intersection(faces_vert2))


def find_boundary(sheet):
    """Find boundary vertices and edges."""
    boundary_vert = set()
    boundary_edge = set()
    for i in sheet.edge_df.index:
        if sheet.edge_df.loc[i, 'opposite'] == -1:
            boundary_vert.add(sheet.edge_df.loc[i, 'srce'])
            boundary_vert.add(sheet.edge_df.loc[i, 'trgt'])
            boundary_edge.add(i)
    return boundary_vert, boundary_edge


def T3_transition(sheet, edge_id, vert_id, d_min, d_sep, nearest):
    # Extract source and target vertex IDs
    srce_id, trgt_id = sheet.edge_df.loc[edge_id, ['srce', 'trgt']]
    # Extract source and target positions as numpy arrays
    endpoint1 = sheet.vert_df.loc[srce_id, ['x', 'y']].values
    endpoint2 = sheet.vert_df.loc[trgt_id, ['x', 'y']].values
    endpoints = [endpoint1, endpoint2]
    # store the associated edges, aka, rank
    edge_associated = sheet.edge_df[(sheet.edge_df['srce'] == vert_id) | (sheet.edge_df['trgt'] == vert_id)]
    rank = (len(edge_associated) - len(edge_associated[edge_associated['opposite'] == -1])) / 2 + len(
        edge_associated[edge_associated['opposite'] == -1])
    vert_associated = list(set(edge_associated['srce'].tolist() + edge_associated['trgt'].tolist()) - {vert_id})
    filtered_rows = sheet.vert_df[sheet.vert_df.index.isin(vert_associated)]
    # extend the edge if needed.
    if sheet.edge_df.loc[edge_id, 'length'] < d_sep * rank:
        extension_needed = d_sep * rank - sheet.edge_df.loc[edge_id, 'length']
        edge_extension(sheet, edge_id, extension_needed)
    sorted_rows_id = []
    # Check adjacency.
    v_adj = adjacent_vert(sheet, vert_id, srce_id, trgt_id)
    if v_adj is not None:
        # First we move the common point.
        sheet.vert_df.loc[v_adj, ['x', 'y']] = list(nearest)

        # Then, we need to update via put-vert and update
        # sequentially by d_sep.
        # The sequence is determined by the sign of the difference
        # between x-value of (nearest - end)
        if nearest[0] - sheet.vert_df.loc[v_adj, 'x'] < 0:
            # Then shall sort x-value from largest to lowest.
            sorted_rows = filtered_rows.sort_values(by='x', ascending=False)
            sorted_rows_id = list(sorted_rows.index)
            # Then pop twice since an extra put vert is only needed for rank 2 adjacent.
            sorted_rows_id.pop(0)
            sorted_rows_id.pop(0)

        else:  # Then shall sort from lowest to largest.
            sorted_rows = filtered_rows.sort_values(by='x', ascending=True)
            sorted_rows_id = list(sorted_rows.index)
            sorted_rows_id.pop(0)
            sorted_rows_id.pop(0)

        # If rank is > 2, then we need to compute more.
        if sorted_rows_id:
            # Store the starting point as the nearest, then compute the unit vector.
            last_coord = nearest
            a = vector(nearest, sheet.vert_df.loc[v_adj, ['x', 'y']].values)
            a_hat = a / round(np.linalg.norm(a), 4)

            for i in sorted_rows_id:
                last_coord += a_hat * d_sep
                new_vert_id = put_vert(sheet, edge_id, last_coord)[0]
                sheet.edge_df.loc[sheet.edge_df['srce'] == i, 'srce'] = new_vert_id
                sheet.edge_df.loc[sheet.edge_df['trgt'] == i, 'trgt'] = new_vert_id

    # Now, for the case of non adjacent.
    elif v_adj is None:
        # The number of points we need to put on the edge is same as the rank.
        a = vector(nearest, sheet.vert_df.loc[srce_id, ['x', 'y']].values)
        a_hat = a / round(np.linalg.norm(a), 4)

        if rank == 2:
            coord1 = nearest - 0.5 * d_sep * a_hat
            coord2 = nearest + 0.5 * d_sep * a_hat
            new_id_1 = put_vert(sheet, edge_id, coord1)[0]
            new_id_2 = put_vert(sheet, edge_id, coord2)[0]
            new_vert_id = [new_id_1, new_id_2]

            # Now, the x-value sorting is based on the distance
            # between the point to the srce_id.
            if nearest[0] - sheet.vert_df.loc[srce_id, 'x'] < 0:
                sorted_rows = filtered_rows.sort_values(by='x', ascending=True)
                sorted_rows_id = list(sorted_rows.index)
            if nearest[0] - sheet.vert_df.loc[srce_id, 'x'] < 0:
                sorted_rows = filtered_rows.sort_values(by='x', ascending=False)
                sorted_rows_id = list(sorted_rows.index)
            for i in sorted_rows_id:
                for j in list(range(len(new_vert_id))):
                    sheet.edge_df.loc[sheet.edge_df['srce'] == i, 'srce'] = new_vert_id[j]
                    sheet.edge_df.loc[sheet.edge_df['trgt'] == i, 'trgt'] = new_vert_id[j]
        elif rank == 3:
            coord1 = nearest - 0.5 * d_sep * a_hat
            coord2 = nearest
            coord3 = nearest + 0.5 * d_sep * a_hat
            new_id_1 = put_vert(sheet, edge_id, coord1)[0]
            new_id_2 = put_vert(sheet, edge_id, coord2)[0]
            new_id_3 = put_vert(sheet, edge_id, coord3)[0]
            new_vert_id = [new_id_1, new_id_2, new_id_3]

            # Now, the x-value sorting is based on the distance
            # between the point to the srce_id.
            if nearest[0] - sheet.vert_df.loc[srce_id, 'x'] < 0:
                sorted_rows = filtered_rows.sort_values(by='x', ascending=True)
                sorted_rows_id = list(sorted_rows.index)
            if nearest[0] - sheet.vert_df.loc[srce_id, 'x'] < 0:
                sorted_rows = filtered_rows.sort_values(by='x', ascending=False)
                sorted_rows_id = list(sorted_rows.index)
            for i in sorted_rows_id:
                for j in list(range(len(new_vert_id))):
                    sheet.edge_df.loc[sheet.edge_df['srce'] == i, 'srce'] = new_vert_id[j]
                    sheet.edge_df.loc[sheet.edge_df['trgt'] == i, 'trgt'] = new_vert_id[j]


def division_2(sheet, rng, cent_data, cell_id):
    """The cells keep growing, when the area exceeds a critical area, then
    the cell divides.

    Parameters
    ----------
    sheet: a :class:`Sheet` object
    cell_id: int
        the index of the dividing cell
    crit_area: float
        the area at which
    growth_rate: float
        increase in the area per unit time
        A_0(t + dt) = A0(t) * (1 + growth_rate * dt)
    """
    condition = sheet.edge_df.loc[:, 'face'] == cell_id
    edge_in_cell = sheet.edge_df[condition]
    # We need to randomly choose one of the edges in cell 2.
    chosen_index = rng.choice(list(edge_in_cell.index))
    # Extract and store the centroid coordinate.
    c0x = float(cent_data.loc[cent_data['face'] == cell_id, ['fx']].values[0])
    c0y = float(cent_data.loc[cent_data['face'] == cell_id, ['fy']].values[0])
    c0 = [c0x, c0y]

    # Add a vertex in the middle of the chosen edge.
    new_mid_index = add_vert(sheet, edge=chosen_index)[0]
    # Extract for source vertex coordinates of the newly added vertex.
    p0x = sheet.vert_df.loc[new_mid_index, 'x']
    p0y = sheet.vert_df.loc[new_mid_index, 'y']
    p0 = [p0x, p0y]

    # Compute the directional vector from new_mid_point to centroid.
    rx = c0x - p0x
    ry = c0y - p0y
    r = [rx, ry]  # use the line in opposite direction.
    # We need to use iterrows to iterate over rows in pandas df
    # The iteration has the form of (index, series)
    # The series can be sliced.
    for index, row in edge_in_cell.iterrows():
        s0x = row['sx']
        s0y = row['sy']
        t0x = row['tx']
        t0y = row['ty']
        v1 = [s0x - p0x, s0y - p0y]
        v2 = [t0x - p0x, t0y - p0y]
        # if the xprod_2d returns negative, then line intersects the line segment.
        if xprod_2d(r, v1) * xprod_2d(r, v2) < 0 and index != chosen_index:
            dx = row['dx']
            dy = row['dy']
            c1 = dx * ry - dy * rx
            c2 = s0y * rx - p0y * rx - s0x * ry + p0x * ry
            k = c2 / c1
            intersection = [s0x + k * dx, s0y + k * dy]
            oppo_index = put_vert(sheet, index, intersection)[0]
            # Split the cell with a line.
            new_face_index = face_division(sheet, mother=cell_id, vert_a=new_mid_index, vert_b=oppo_index)
            # Put a vertex at the centroid, on the newly formed edge (last row in df).
            cent_index = put_vert(sheet, edge=sheet.edge_df.index[-1], coord_put=c0)[0]
            random_int_1 = rng.integers(10000, 15000) / 1000
            random_int_2 = rng.integers(10000, 15000) / 1000
            sheet.face_df.loc[cell_id, 'T_cycle'] = np.array(random_int_1, dtype=np.float64)
            sheet.face_df.loc[new_face_index, 'T_cycle'] = np.array(random_int_2, dtype=np.float64)
            sheet.face_df.loc[cell_id, 'prefered_area'] = 1
            sheet.face_df.loc[new_face_index, 'prefered_area'] = 1
            print(f'cell {cell_id} is divided, dauther cell {new_face_index} is created.')
            return new_face_index


def neighbour_edge(sheet, face1, face2):
    """
    Finds the index of the neighboring edge shared by face1 and face2,
    considering the 'opposite' relationship between edges.

    Args:
        sheet: The cell sheet containing edge and vertex information.
        face1: Index of the first face.
        face2: Index of the second face.

    Returns:
        The index of the shared neighboring edge if it exists, otherwise None.
    """
    # Get the edges associated with face1 and face2
    face1_edges = sheet.edge_df[sheet.edge_df["face"] == face1]
    face2_edges = sheet.edge_df[sheet.edge_df["face"] == face2]

    # Check for common edges by matching the 'opposite' relationship
    for edge_id in face1_edges.index:
        opposite_edge = sheet.edge_df.loc[edge_id, "opposite"]
        if opposite_edge in face2_edges.index:
            return edge_id  # Return the edge from face1

    print(f'Cannot find mutual edge between cell {face1} and {face2} \n')
    return None

def find_stb_neighbors(sheet):
    merge_pair = None
    for i in sheet.edge_df.index:
        oppo = sheet.edge_df.loc[i, 'opposite']
        face1 = sheet.edge_df.loc[i, 'face']

        # Skip invalid entries
        if oppo == -1 or pd.isna(oppo) or pd.isna(face1):
            continue
        if oppo not in sheet.edge_df.index or face1 not in sheet.face_df.index:
            continue

        face2 = sheet.edge_df.loc[oppo, 'face']
        if pd.isna(face2) or face2 not in sheet.face_df.index:
            continue

        # Check STB–STB and ensure mutual edge
        if (sheet.face_df.loc[face1, 'cell_class'] == 'STB' and
            sheet.face_df.loc[face2, 'cell_class'] == 'STB'):

            mutual_edge = neighbour_edge(sheet, face1, face2)
            if mutual_edge is not None:
                return tuple(sorted((face1, face2)))  # Just return one pair
    return None


def cell_merge(sheet, face1, face2, new_cell_class):
    """
    Removes an edge and its opposite edge, giving a cell merging behaviour.

    Args:
        sheet: The cell sheet containing edge and vertex information.
        face1 and face2: ID of the two cells that are going to be merged.
        new_cell_class: Cell class of the resultant cell after merge.
    """
    # Identify all mutual edges between face1 and face2
    mutual_edges = []
    face1_edges = sheet.edge_df[sheet.edge_df["face"] == face1]
    face2_edges = sheet.edge_df[sheet.edge_df["face"] == face2]

    # Check for common edges by matching the 'opposite' relationship
    for edge_id in face1_edges.index:
        opposite_edge = sheet.edge_df.loc[edge_id, "opposite"]
        if opposite_edge in face2_edges.index:
            mutual_edges.append(edge_id)
            mutual_edges.append(opposite_edge)

    if not mutual_edges:
        raise ValueError(f"No mutual edges found between face {face1} and face {face2}")

    new_face_id = min(face1, face2)  # Default to the smaller ID if neither is 'ST'
    obsolete_face_id = max(face1, face2)
    new_prefered_area = sheet.face_df.loc[face1,'prefered_area'] + sheet.face_df.loc[face2,'prefered_area']

    # Reassign all edges belonging to face1 or face2 to the new face ID
    sheet.edge_df.loc[sheet.edge_df['face'].isin([face1, face2]), 'face'] = new_face_id

    # Drop all mutual edges
    sheet.edge_df.drop(index=mutual_edges, inplace=True)
    # Drop the obsolete face
    if obsolete_face_id in sheet.face_df.index:
        sheet.face_df.drop(index=obsolete_face_id, inplace=True)

    # Assign 'ST' to the new face and update associated edges
    sheet.face_df.loc[new_face_id, "cell_class"] = new_cell_class
    sheet.face_df.loc[new_face_id,'prefered_area'] = new_prefered_area
    return new_face_id


def update_cell_type(sheet, face_id, new_type='ST'):
    """
    Updates the 'cell type' of all edges associated with a given face ID to a new type.

    Args:
        sheet: The cell sheet containing edge and vertex information.
        face_id: The ID of the face whose edges' 'cell type' should be updated.
        new_type: The new cell type to assign to the edges (default is 'ST').
    """
    # Find the edges associated with the given face ID
    edges_to_update = sheet.edge_df[sheet.edge_df['face'] == face_id].index

    # Update the 'cell type' property for these edges
    sheet.edge_df.loc[edges_to_update, 'cell type'] = new_type


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
    edge_end1 = sheet.edge_df.loc[edge, 'srce']
    edge_end2 = sheet.edge_df.loc[edge, 'trgt']
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
            x = sheet.vert_df.loc[i, 'x']
            y = sheet.vert_df.loc[i, 'y']
            if x_smaller < x < x_larger and y_smaller < y < y_larger:
                verts.append(i)
            else:
                continue
    return verts


def perturbate_T3(sheet, vert1, vert2, d_sep):
    """
    This function can be used when two vertices are going to collide.

    Logic:
        create a virtual line between the 2 vertices.


    """
    # Extract the coordinates of two vertices, then draw a virtual line between.
    v1_coord = sheet.vert_df.loc[vert1, ['x', 'y']].to_numpy(dtype=float)
    v2_coord = sheet.vert_df.loc[vert2, ['x', 'y']].to_numpy(dtype=float)
    virtual_line = v2_coord - v1_coord
    # use unit vector of the line to find coordinate of the middle point.
    length_vline = np.linalg.norm(virtual_line)
    unit_vline = virtual_line / length_vline
    mid_coord = v1_coord + length_vline / 2 * unit_vline
    # Use the vector from midpoint to v2 to find the perpendicular vector.
    mid_v2 = v2_coord - mid_coord
    mid_perpendicular = np.array([-mid_v2[1], mid_v2[0]])
    mid_perpendicular = mid_perpendicular / np.linalg.norm(mid_perpendicular)
    mid_perpendicular = d_sep * mid_perpendicular

    # Now, we need to update the postion of vert1 and vert2.
    # Need the vector from v1 to mid for updating vert1.
    v1_mid = mid_coord - v1_coord
    sheet.vert_df.loc[vert1, ['x', 'y']] += (v1_mid + mid_perpendicular)
    # Then update vert2.
    sheet.vert_df.loc[vert2, ['x', 'y']] += (-mid_v2 - mid_perpendicular)
    return True

def extrude_face(sheet, face_index):
    """
    This is a function that removes a face with face_index, but only deletes vertices that are unique to that face,
    i.e., not shared with any other faces.

    This function returns the indices of the vertices that are removed.
    """
    # 1. Get the face's half-edges
    face_edges = sheet.edge_df[sheet.edge_df["face"] == face_index]
    face_edge_indices = face_edges.index

    # 2. Get the face's vertices
    face_verts = np.union1d(face_edges["srce"].values, face_edges["trgt"].values)

    # 3. Drop the face's edges (they're only half-edges owned by this face)
    sheet.edge_df.drop(index=face_edge_indices, inplace=True)

    # 4. Drop the face
    sheet.face_df.drop(index=face_index, inplace=True)

    # 5. Determine which face vertices are now unused (not in any edge)
    remaining_edges = sheet.edge_df
    remaining_verts = np.union1d(
        remaining_edges["srce"].values,
        remaining_edges["trgt"].values
    )

    removable_verts = [v for v in face_verts if v not in remaining_verts]

    # 6. Drop unused vertices
    if removable_verts:
        sheet.vert_df.drop(index=removable_verts, inplace=True)

    # 7. Topology and indexing cleanup
    drop_two_sided_faces(sheet)  # only if relevant
    sheet.reset_index()
    sheet.reset_topo()

    return removable_verts


def record_cell_energy_dynamic(sheet, model, face_ids):
    """
    Return energy contributions for a list of face_ids,
    dynamically using model.labels for energy terms.
    """
    geom.update_all(sheet)

    # Get per-element energy terms (as Series) from each effector
    energy_terms = model.compute_energy(sheet, full_output=True)
    label_to_term = dict(zip(model.labels, energy_terms))

    records = {}
    for fid in face_ids:
        entry = {}
        for label, term in label_to_term.items():
            if isinstance(term, pd.Series):
                if term.index.equals(sheet.edge_df.index):  # edge-based term
                    edge_idxs = sheet.edge_df[sheet.edge_df['face'] == fid].index
                    entry[label] = term.loc[edge_idxs].sum()
                elif term.index.equals(sheet.face_df.index):  # face-based term
                    entry[label] = term.loc[fid]
                else:
                    raise ValueError(f"Unrecognized term index for label {label}")
            else:
                raise TypeError(f"Unexpected type for energy term: {type(term)}")
        records[fid] = entry

    return records

def remove_collapsed_edges(sheet):
    """Removes degenerate edges where srce == trgt, which causes issues in Tyssue when call type1_transition()."""
    bad_edges = sheet.edge_df.query("srce == trgt").index
    if len(bad_edges) > 0:
        print(f"Removing {len(bad_edges)} collapsed edge(s): {list(bad_edges)}")
        sheet.edge_df.drop(index=bad_edges, inplace=True)
        sheet.reset_index(order=False)  # Re-index edges to keep things consistent

def remove_self_loop_edges(sheet):
    bad_edges = sheet.edge_df[sheet.edge_df['srce'] == sheet.edge_df['trgt']].index
    if len(bad_edges) > 0:
        print(f"Removed {len(bad_edges)} edges with srce == trgt.")
        sheet.remove(bad_edges)
        sheet.reset_index(order=True)
        geom.update_all(sheet)



""" This is the end of the script. """
