# -*- coding: utf-8 -*-
"""
Two ways to define the arrangement of the hexagon genrated in hexa_grid2d.
"""

# Draw voronoi diagram
import numpy as np
import matplotlib.pylab as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from tyssue.generation import  from_2d_voronoi

nx = 3
ny=2
distx=1
disty = 1
noise = 0

def hexa_grid(nx, ny, distx, disty, noise=None):
    """Creates an hexagonal grid of points
    """
    cy, cx = np.mgrid[0:ny+2, 0:nx+2]
    cx = cx.astype(float)
    cy = cy.astype(float)
    cy[::2, :] += 0.5

    centers = np.vstack([cx.flatten(), cy.flatten()]).astype(float).T
    centers[:, 0] *= distx
    centers[:, 1] *= disty
    if noise is not None:
        pos_noise = np.random.normal(scale=noise, size=centers.shape)
        centers += pos_noise
# uncomment the following line to get a voronoi diagram with flat top/bot.
    centers = np.flip(centers,1)
    return centers


grid = hexa_grid(nx, ny, distx, disty, noise)
#grid = np.flip(grid,1)
datasets = from_2d_voronoi(Voronoi(grid))

vor = Voronoi(grid)

fig = voronoi_plot_2d(vor)
plt.show()





