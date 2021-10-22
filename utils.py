import os

import numpy as np
import pandas as pd

excelPath = "features/original.xlsx"
refinedexcelPath = "features/refined.xlsx"
picklePath = "features/original.pkl"
refinedpicklePath = "features/refined.pkl"
imagePath = "graphs/original/"
refinedImagePath = "graphs/refined/"
originalDB = "testModels/db"
refinedDB = "testModels/refined_db"
target_vertices = 1000
target_faces = 2000
hist_amount = 10000
hist_features  = ['A3', 'D1', 'D2', 'D3', 'D4']
scal_features  = ["area", "axis-aligned_bounding_box_distance", "diameter", "compactness", "eccentricity" ]
scal_features_norm = ["area_norm", "axis-aligned_bounding_box_distance_norm", "diameter_norm", "compactness_norm", "eccentricity_norm"]
hist_features_norm = ["A3_norm", "D1_norm", "D2_norm", "D3_norm", "D4_norm"]
norm_vector_path = "features/vector.npy"
emd_norm_vector_path = "features/dist_vector.npy"


def read_excel(original=True):
  # Load the excel into a pandas df
  if original:
    return pd.read_pickle(picklePath)
  else:
    return pd.read_pickle(refinedpicklePath)

def save_excel(df, original = True):
  if original:
    df.to_pickle(picklePath)
    df.to_excel(excelPath)
  else:
    df.to_pickle(refinedpicklePath)
    df.to_excel(refinedexcelPath)

def ensure_dir(file_path):
  directory = os.path.dirname(file_path)
  if not os.path.exists(directory):
    os.makedirs(directory)


def get_face_barycentre(face, vertices):
  return (vertices[face[0]] + vertices[face[1]] + vertices[face[2]]) / 3


def refined_path(path):
  return path[:11] + 'refined_' + path[11:]


def angle(vector1, vector2):
  return np.arccos(np.clip(np.dot(vector1, vector2), -1.0, 1.0))


def shape_paths(dbfolder):
  paths = []
  for path, subdirs, files in os.walk(dbfolder):
    for name in files:
      if name.endswith('.off'):
        paths.append((os.path.join(path, name)))
  return paths
