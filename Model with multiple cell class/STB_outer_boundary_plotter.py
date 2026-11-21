import matplotlib.pyplot as plt
from tyssue import Sheet
from tyssue.io import hdf5

# One shared figure for all three surfaces
fig, ax = plt.subplots(figsize=(10, 4))

# ---- Uniform alpha ----
dsets = hdf5.load_datasets('uniform_alpha.hdf5')
uniform_sheet = Sheet('reloaded', dsets)
STB_ids = uniform_sheet.face_df[
    uniform_sheet.face_df['cell_class'].isin(['STB', 'boundary_fixed'])
].index
uniform_stb_surface_xys = uniform_sheet.edge_df[
    (uniform_sheet.edge_df['face'].isin(STB_ids)) & (uniform_sheet.edge_df['opposite'] == -1)][['sx', 'sy']]
sorted_uniform = uniform_stb_surface_xys.sort_values(by='sx').reset_index(drop=True)
ax.plot(sorted_uniform['sx'], sorted_uniform['sy'], '-o', color='tab:blue',
        markersize=3, linewidth=1.5, label='Uniform area elasticity')

# ---- 80% alpha ----
dsets2 = hdf5.load_datasets('80percent_alpha.hdf5')
reduced_sheet = Sheet('reloaded', dsets2)
STB_ids = reduced_sheet.face_df[
    reduced_sheet.face_df['cell_class'].isin(['STB', 'boundary_fixed'])
].index
reduced_stb_surface_xys = reduced_sheet.edge_df[
    (reduced_sheet.edge_df['face'].isin(STB_ids)) & (reduced_sheet.edge_df['opposite'] == -1)][['sx', 'sy']]
sorted_reduced = reduced_stb_surface_xys.sort_values(by='sx').reset_index(drop=True)
ax.plot(sorted_reduced['sx'], sorted_reduced['sy'], '-o', color='tab:orange',
        markersize=3, linewidth=1.5, label='80% area elasticity')

# ---- 50% alpha ----
dsets3 = hdf5.load_datasets('50percent_alpha.hdf5')
half_sheet = Sheet('reloaded', dsets3)
STB_ids = half_sheet.face_df[
    half_sheet.face_df['cell_class'].isin(['STB', 'boundary_fixed'])
].index
half_stb_surface_xys = half_sheet.edge_df[
    (half_sheet.edge_df['face'].isin(STB_ids)) & (half_sheet.edge_df['opposite'] == -1)][['sx', 'sy']]
sorted_half = half_stb_surface_xys.sort_values(by='sx').reset_index(drop=True)
ax.plot(sorted_half['sx'], sorted_half['sy'], '-o', color='tab:green',
        markersize=3, linewidth=1.5, label='50% area elasticity')

# ---- Finish the plot ----
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('STB outer surface comparison')
ax.set_aspect('equal', adjustable='datalim')
ax.legend()
plt.tight_layout()
plt.show()