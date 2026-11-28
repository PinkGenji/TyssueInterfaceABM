import matplotlib.pyplot as plt
from tyssue import Sheet
from tyssue.draw import sheet_view
from tyssue.io import hdf5


def clean_path(raw):
    """Strip whitespace and the quotes Windows adds with 'Copy as path'."""
    return raw.strip().strip('"').strip("'")


def load_sheet(path):
    """Load a Tyssue Sheet from an HDF5 file."""
    dsets = hdf5.load_datasets(path)
    sheet = Sheet('reloaded', dsets)

    # sheet_view expects a 'visible' column in some Tyssue versions.
    for df in (sheet.vert_df, sheet.edge_df, sheet.face_df):
        if 'visible' not in df.columns:
            df['visible'] = True
    return sheet


def show_sheet(path):
    sheet = load_sheet(path)
    print(f'Loaded {sheet.Nf} faces, {sheet.Ne} edges, {sheet.Nv} vertices')

    fig, ax = sheet_view(sheet, coords=['x', 'y'])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(path.replace('\\', '/').split('/')[-1])
    plt.tight_layout()
    plt.show()  # blocks until you close the window


if __name__ == '__main__':
    print("Paste an HDF5 file path to view it. Type 'q' or press Enter to quit.")

    while True:
        raw = input('\nHDF5 path: ')
        path = clean_path(raw)

        if path == '' or path.lower() == 'q':
            break

        try:
            show_sheet(path)
        except FileNotFoundError:
            print(f'File not found: {path}')
        except Exception as e:
            print(f'Could not load or plot this file: {type(e).__name__}: {e}')

    print('Done.')