import os

import pandas as pd
import numpy as np

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


def read_excel(original=True):
  # Load the excel into a pandas df
  if original:
    return pd.read_pickle(picklePath)
  else:
    return pd.read_pickle(refinedpicklePath)


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
