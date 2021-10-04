import os

import pandas as pd

excelPath = "features/original.xlsx"
refinedexcelPath = "features/refined.xlsx"
imagePath = "graphs/original/"
refinedImagePath = "graphs/refined/"
originalDB = "testModels/db"
refinedDB = "testModels/refined_db"
target_vertices = 1000
target_faces = 2000


def read_excel(original=True):
  # Load the excel into a pandas df
  if original:
    return pd.read_excel(excelPath, index_col=0)
  else:
    return pd.read_excel(refinedexcelPath, index_col=0)


def ensure_dir(file_path):
  directory = os.path.dirname(file_path)
  if not os.path.exists(directory):
    os.makedirs(directory)


def get_face_barycentre(face, vertices):
  return (vertices[face[0]] + vertices[face[1]] + vertices[face[2]]) / 3

def refined_path(path):
  return path[:11] + 'refined_' + path[11:]

def shape_paths(dbfolder):
  paths = []
  for path, subdirs, files in os.walk(dbfolder):
    for name in files:
      if name.endswith('.off'):
        paths.append((os.path.join(path, name)))
  return paths
