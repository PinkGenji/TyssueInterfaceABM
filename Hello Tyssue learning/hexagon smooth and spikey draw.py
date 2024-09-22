# -*- coding: utf-8 -*-
"""
Two ways to define the arrangement of the hexagon genrated in hexa_grid2d.
"""

# Draw voronoi diagram
import numpy as np
import matplotlib.pylab as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from tyssue.generation import  from_2d_voronoi


""" Code for spikey top """
nx = 5
ny=5
distx=1
disty = 1
noise = 0

def hexa_grid(nx, ny, distx, disty, noise=None):
    """Creates an hexagonal grid of points
    """
    cy, cx = np.mgrid[0:ny, 0:nx]
    cx = cx.astype(float)
    cy = cy.astype(float)
    cx[::2, :] += 0.5

    centers = np.vstack([cx.flatten(), cy.flatten()]).astype(float).T
    centers[:, 0] *= distx
    centers[:, 1] *= disty
    if noise is not None:
        pos_noise = np.random.normal(scale=noise, size=centers.shape)
        centers += pos_noise
    return centers


grid = hexa_grid(nx, ny, distx, disty, noise)
#grid = np.flip(grid,1)
datasets = from_2d_voronoi(Voronoi(grid))

vor = Voronoi(grid)

fig = voronoi_plot_2d(vor)
plt.show()

""" Code for smooth top. """

def hexa_grid(nx, ny, distx, disty, noise=None):
    """Creates an hexagonal grid of points
    """
    cy, cx = np.mgrid[0:ny, 0:nx]
    cx = cx.astype(float)
    cy = cy.astype(float)
    cx[::2, :] += 0.5

    centers = np.vstack([cx.flatten(), cy.flatten()]).astype(float).T
    centers[:, 0] *= distx
    centers[:, 1] *= disty
    if noise is not None:
        pos_noise = np.random.normal(scale=noise, size=centers.shape)
        centers += pos_noise

    # Flip the x and y to get smooth top
    centers = np.flip(centers,1)
    return centers


grid = hexa_grid(nx, ny, distx, disty, noise)
#grid = np.flip(grid,1)
datasets = from_2d_voronoi(Voronoi(grid))

vor = Voronoi(grid)

fig = voronoi_plot_2d(vor)
plt.show()

