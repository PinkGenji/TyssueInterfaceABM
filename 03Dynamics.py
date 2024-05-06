# -*- coding: utf-8 -*-
"""
This file is for learning 03Dynamics of the tyssue package.
"""

'''
One of the objectives of tyssue is to make it easy to define and change the expression
of the epithelium energy. For this, we define two classes of obejcts: Effector and Model.

Effector: those define a single energy term, evaluated on the mesh, and depending
on the values in the data frame.
For example, line tension, for which the energy is proportional to the length of the hal-edge,
is defined as an Effector object. 
For each Effector, we define a way to compute the energy and its spatial derivative, the gradient.

Model: a model is basically a collection of effectors, with the mechanisms to combine
them to define the total dynamics of the system.

In general, the parameters of the models are addressable at the single element level.
For example, the line tension can be set for each individual edge.

'''




