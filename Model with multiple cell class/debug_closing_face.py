# Read-in the HDF 5 file of debug_closing_face then investigate the bug.
from matplotlib import pyplot as plt
import numpy as np
from tyssue import Sheet
from tyssue.draw import sheet_view
from tyssue import PlanarGeometry as geom
from tyssue.topology.base_topology import close_face
from tyssue.io import hdf5
from tyssue.config.draw import sheet_spec


# Read the dataset as HDF5 file.
dsets = hdf5.load_datasets('Debug_closing_face_error.hdf5')
sheet2 = Sheet('reloaded',dsets)
fig, ax = sheet_view(sheet2)
plt.show()      # Show the plot to double check.

# use get_valid() mask over self.edge_df for invalid faces.
validate_series = sheet2.get_valid()
false_indices = validate_series[validate_series == False].index
# Use the false_indices of the edges, find the faces that are not closed.
unclosed_face = []
for i in false_indices:
    face = sheet2.edge_df.loc[i,'face']
    if face not in unclosed_face:
        unclosed_face.append(face)
unclosed_face = list(set(sheet2.edge_df.loc[false_indices, 'face']))
edge_num = len(sheet2.edge_df)
print(f'Initially, unclosed faces are: {unclosed_face}, there are {edge_num} edges in total')

# Now, update the draw sepcs to highlight these faces.
draw_specs = sheet_spec()
# Enable face visibility.
draw_specs['face']['visible'] = True
for i in sheet2.face_df.index:
    if i in unclosed_face:
        sheet2.face_df.loc[i,'color'] = 0.7
    else:
        sheet2.face_df.loc[i,'color'] = 0.1

draw_specs['face']['color'] = sheet2.face_df['color']
draw_specs['face']['alpha'] = 0.2   # Set transparency.

# Enable the visibility of vertices.
sheet2.vert_df['rand'] = np.linspace(0.0, 1.0, num=sheet2.vert_df.shape[0])

cmap = plt.colormaps['viridis']
color_cmap = cmap(sheet2.vert_df.rand)
draw_specs['vert']['visible'] = True

draw_specs['vert']['color'] = color_cmap
draw_specs['vert']['alpha'] = 0.5
draw_specs['vert']['s'] = 20

fig, ax = sheet_view(sheet2, ['x', 'y'], **draw_specs)
plt.show()

# Then close the faces based on the series that stores the invalid faces.
for i in unclosed_face:
    close_face(sheet2, i)
    geom.update_all(sheet2)

# Recheck the invalid faces.
validate_series = sheet2.get_valid()
false_indices = validate_series[validate_series == False].index
unclosed_face = []
for i in false_indices:
    face = sheet2.edge_df.loc[i,'face']
    if face not in unclosed_face:
        unclosed_face.append(face)
unclosed_face = list(set(sheet2.edge_df.loc[false_indices, 'face']))
edge_num = len(sheet2.edge_df)
print(f'Unclosed faces are: {unclosed_face}, there are {edge_num} edges in total')

# See the effect of sanitize on the data frames,
# sanitize is done via merge border edges into a single edge, so there is no vertex formed by two edges on border.
sheet2.sanitize(trim_borders=True)
geom.update_all(sheet2)

# Recheck the invalid faces.
validate_series = sheet2.get_valid()
false_indices = validate_series[validate_series == False].index
unclosed_face = []
for i in false_indices:
    face = sheet2.edge_df.loc[i,'face']
    if face not in unclosed_face:
        unclosed_face.append(face)
unclosed_face = list(set(sheet2.edge_df.loc[false_indices, 'face']))
edge_num = len(sheet2.edge_df)
print(f'After sanitize without border trim, Unclosed faces are: {unclosed_face}, there are {edge_num} edges in total')
fig, ax = sheet_view(sheet2)
plt.show() 

print('\n This is the end of this script. (＾• ω •＾) ')
