import logging
import os
import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh
import utils
import subdivision

def scale_mesh(mesh, scale):
  #Make a vector to scale x, y and z in the mesh to this value
  scaleVector = [scale, scale, scale]

  #Create transformation matrix
  matrix = np.eye(4)
  matrix[:3, :3] *= scaleVector
  mesh.apply_transform(matrix)
  return mesh

def save_mesh(mesh, path):
  utils.ensure_dir(path)
  trimesh.exchange.export.export_mesh(mesh, path, file_type="off")

# def refine_outlier(show=False):
#   df = pd.read_excel(utils.excelPath)
#   undersampled = df[df["subsampled_outlier"] == True]
#   for path in undersampled["path"]:
#     refined_path = path[:11] + 'refined_' + path[11:]
#     mesh = trimesh.load(path, force='mesh')
#     refined_mesh = subdivision.subdivide(mesh, utils.target_vertices, show=show)
#     save_mesh(refined_mesh, refined_path)
#     trimesh.exchange.export.export_mesh(refined_mesh, refined_path, file_type="off")

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


def process_all():
  #Perform all preprocessing steps on all meshes:
  df = pd.read_excel(utils.excelPath)
  normalize_values_before = []
  normalize_values_after = []
  for index, row in df.iterrows():
    path = row['path']
    mesh = trimesh.load(row['path'])
    normalize_values_before.append(mesh.center_mass)
    mesh = normalize_mesh(mesh)
    normalize_values_after.append(mesh.center_mass)
    if row['subsampled_outlier']:
      mesh = subdivision.subdivide(mesh, utils.target_vertices)
    if row['supersampled_outlier']:
      mesh = subdivision.superdivide(mesh, utils.target_faces)
      save_mesh(mesh, refined_path)
    refined_path = path[:11] + 'refined_' + path[11:]
    save_mesh(mesh, refined_path)
  avgs_before = [sum(vals) / len(normalize_values_before) for vals in zip(*normalize_values_before)]
  avgs_after = [sum(vals) / len(normalize_values_after) for vals in zip(*normalize_values_after)]
  print(avgs_before, avgs_after)