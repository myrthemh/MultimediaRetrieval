import logging
import os
import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh
import utils

def scale_mesh(mesh, scale):
  #Make a vector to scale x, y and z in the mesh to this value
  scaleVector = [scale, scale, scale]

  #Create transformation matrix
  matrix = np.eye(4)
  matrix[:3, :3] *= scaleVector
  mesh.apply_transform(matrix)
  return mesh

def ensure_dir(file_path):
  directory = os.path.dirname(file_path)
  if not os.path.exists(directory):
    os.makedirs(directory)

def refine_outliers(show=False):
  df = pd.read_excel(utils.excelPath)
  undersampled = df[df["subsampled_outlier"] == True]
  for path in undersampled["path"]:
    refined_path = path[:11] + 'refined_' + path[11:]
    refined_mesh = subdivide(trimesh.load(path, force='mesh'))
    ensure_dir(refined_path)
    trimesh.exchange.export.export_mesh(refined_mesh, refined_path, file_type="off")
    if show:
      mesh1 = trimesh.load(refined_path, force='mesh')
      mesh2 = trimesh.load(path, force='mesh')
      meshes = [mesh1, mesh2]
      for i, m in enumerate(meshes):
        m.apply_translation([0, 0, i * 1])

      trimesh.Scene(meshes).show()

def normalize_mesh(path):
  mesh = trimesh.load(path, force='mesh')

  #Center the mass of the mesh on (0,0,0)
  center_mass = mesh.center_mass
  mesh.apply_translation(-center_mass)
  
  #Get the highest value we can scale with so it still fits within the unit cube
  scale_value = 1 / max(mesh.bounds.flatten())
  mesh = scale_mesh(mesh, scale_value)
  
  # print(mesh.bounds)
  # print(mesh.center_mass)
  return mesh

def subdivide(mesh):
  x = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
  newmesh = trimesh.Trimesh(vertices=x[0], faces=x[1])
  return newmesh
  
