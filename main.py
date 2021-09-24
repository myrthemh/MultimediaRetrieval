import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh
import pyrender
import preprocess, analyze, utils
trimesh.util.attach_to_log()

def render(meshes, showWireframe=True):
  scene = pyrender.Scene()
  for mesh in meshes:
    mesh1 = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh1)
  if showWireframe:
    for mesh in meshes:
      #Add copy of mesh in wireframe mode
      material = pyrender.Material()
      #mesh = scale_mesh(mesh, 1.001)
      wireframe = pyrender.Mesh.from_trimesh(mesh, wireframe=True, smooth=False, material=material)
      scene.add(wireframe)
  pyrender.Viewer(scene, use_raymond_lighting=True)

# Step 1
def step_1():
  mesh = trimesh.load('testModels/db/0/m0/m0.off', force='mesh')
  render([mesh])
  
# Step 2

# The key constraints here are that (a) the reduced database should contain at least 200 shapes; (b) you should have
# shapes of most (ideally all) of the existing class types in the database; (c) try to balance the classes, i.e.,
# do not use tens of shapes of one class and only a handful of shapes of another class. That is: Start small. When
# this works, add more shapes to your (already) functioning code, and repeat the tests.
#
# Start building a simple filter that checks all shapes in the database. The filter should output, for each shape
#
# - the class of the shape
# - the number of faces and vertices of the shape
# - the type of faces (e.g. only triangles, only quads, mixes of triangles and quads)
# - the axis-aligned 3D bounding box of the shapes

analyze.filter_database(utils.originalDB, utils.excelPath)
preprocess.process_all()
analyze.filter_database(utils.refinedDB, utils.refinedexcelPath)
analyze.save_all_histograms(utils.read_excel(original=True), utils.imagePath)
analyze.save_all_histograms(utils.read_excel(original=False), utils.refinedImagePath)
