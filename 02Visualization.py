# -*- coding: utf-8 -*-
"""
This file is for learning the 02Visualization.py of tyssue package.
"""

'''
As seen before, parameters are passed around in tyssue through specifications,
nested dictionaries of parameters. 
We use a similar mechanism to specify visulization functions.

'''
# Most visualization will be done with sheet_view function.

from pprint import pprint
import numpy as np
import pandas as pd

import matplotlib.pylab as plt
import ipyvolume as ipv

import tyssue

from tyssue import Sheet, SheetGeometry as geom
from tyssue.generation import three_faces_sheet
from tyssue.draw import sheet_view
from tyssue import config
from tyssue import Monolayer, config, MonolayerGeometry
from tyssue.generation import extrude

datasets, _ = three_faces_sheet()
sheet = Sheet('3cells_2D',datasets)

geom.update_all(sheet)





























'''
This is the end of the file.
'''
