"""
This script contains the functions that describes the transition between cell classes.
"""


# Now we write a function that for an input of a cell dataframe, it changes
# 10% of its "S" cells to "G1" class, and doubles the target area of new G1 cells.
def select_S_to_G1(face_df, rng, percentage=0.1):
    """Converts 10% of 'S' cells to 'G1' and doubles their target area.

    Parameters:
        face_df (DataFrame): DataFrame containing cell properties.
        rng (np.random.Generator): Random number generator for reproducibility.
        percentage (float): Probability of conversion (default 10%).
    """

    s_cells = face_df['cell_class'] == "S"
    s_indices = face_df.index[s_cells]  # Get indices of the S cells.
    number_to_change = int(len(s_indices) * percentage)

    if number_to_change > 0:
        # Using RNG, we pick the indices of the cells that will change class.
        selected_indices = rng.choice(s_indices, size=number_to_change, replace=False)
        face_df.loc[selected_indices, 'cell_class'] = 'G1'  # Update selected cells.
        face_df.loc[selected_indices, 'prefered_area'] *= 2  # Double the target area.


# Now, we need to have a function that converts all "G1" cells that reach 97%
# of their target area (as an approximation of full size) to be "M" class.
def select_G1_to_M(face_df):
    """Converts all 'G1' cells to 'M' when they reach 97% of their target area.

    Parameters:
        face_df (DataFrame): DataFrame containing cell properties.
    """

    for i in face_df.index:
        if face_df.loc[i, 'class'] == "G1" and face_df.loc[i, 'area'] >= 0.97 * face_df.loc[i, 'prefered_area']:
            face_df.loc[i, 'class'] = "M"

        # Now, we write a function that performs cell division on all "M" cells.


def divide_cells_in_M(face_df):
    """Divides all 'M' cells and assigns both parent and daughter to 'G2'.

    Parameters:
        face_df (DataFrame): DataFrame containing cell properties.
    """

    M_cells = face_df[face_df["class"] == "M"].index.tolist()

    for cell_id in M_cells:
        # Perform cell division and get the new daughter cell ID
        daughter_id = cell_division(face_df, cell_id, geom)

        # Reset properties for both parent and daughter
        face_df.loc[[cell_id, daughter_id], "prefered_area"] = 1.0
        face_df.loc[[cell_id, daughter_id], "class"] = "G2"  # Set both to "G2"

        print(f"Cell {cell_id} divided into {daughter_id} and both are now G2")


# Now, we write a function that converts all "G2" cells to "S" class when their
# area is 97% of the target area.
def select_G2_to_S(face_df):
    """Converts all 'G2' cells to 'S' when they reach 97% of their target area.

    Parameters:
        face_df (DataFrame): DataFrame containing cell properties.
    """

    for i in face_df.index:
        if face_df.loc[i, 'class'] == "G2" and face_df.loc[i, 'area'] >= 0.97 * face_df.loc[i, 'prefered_area']:
            face_df.loc[i, 'class'] = "S"

        # Now, we write a function that converts an "S" cell to an "F" cell with 10%


# probability, if it has at least one neighboring "STB" cell.
def select_S_to_F(face_df, sheet, rng, percentage=0.1):
    """Converts 10% of 'S' cells to 'F' if they have an 'STB' neighbor.

    Parameters:
        face_df (DataFrame): DataFrame containing cell properties.
        sheet: The tissue sheet object (to get neighbors).
        rng (np.random.Generator): Random number generator for reproducibility.
        percentage (float): Probability of conversion (default 10%).
    """

    s_cells = face_df["class"] == "S"
    s_indices = face_df.index[s_cells]  # Get indices of the S cells

    # Filter only those with at least one 'STB' neighbor
    valid_indices = [i for i in s_indices if any(face_df.loc[n, "class"] == "STB" for n in sheet.get_neighbours(i))]

    number_to_change = int(len(valid_indices) * percentage)

    if number_to_change > 0:
        # Randomly select which 'S' cells will turn into 'F'
        selected_indices = rng.choice(valid_indices, size=number_to_change, replace=False)
        face_df.loc[selected_indices, "class"] = "F"  # Update selected cells


""" This is the end of the script."""
