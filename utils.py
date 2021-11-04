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
sim_images_path = "simimages/"
ann_images_path = "annimages/"
eval_images_path = "evalimages/"
sim_image_size = 256
# originalDB = "veelModels/db"
# refinedDB = "veelModels/refined_db"
target_vertices = 1000
target_faces = 2000
nr_bins_hist = 20
hist_amount = 10000
query_size = 5
hist_features = ['A3', 'D1', 'D2', 'D3', 'D4']
scal_features = ["area", "axis-aligned_bounding_box_distance", "diameter", "compactness", "eccentricity"]
scal_features_norm = ["area_norm", "axis-aligned_bounding_box_distance_norm", "diameter_norm", "compactness_norm",
                      "eccentricity_norm"]
hist_features_norm = ["A3_norm", "D1_norm", "D2_norm", "D3_norm", "D4_norm"]
norm_vector_path = "features/vector.npy"
emd_norm_vector_path = "features/dist_vector.npy"
classes = [
  "Insect",  # 0
  "Farm animal",  # 1
  "People",  # 2
  "Face",  # 3
  "Building",  # 4
  "Container",  # 5
  "LampOrWatch",  # 6
  "Stabweapon",  # 7
  "Chair",  # 8
  "Table",  # 9
  "Flowerpot",  # 10
  "Tool",  # 11
  "Airplane",  # 12
  "Aircraft",  # 13
  "Spacecraft",  # 14
  "Car",  # 15
  "Chess piece",  # 16
  "DoorOrChest",  # 17
  "Satellite"  # 18
]

def read_excel(original=True):
  # Load the excel into a pandas df
  if original:
    return pd.read_pickle(picklePath)
  else:
    return pd.read_pickle(refinedpicklePath)


def save_excel(df, original=True):
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

def image_paths(class_folder, ann=False):
  paths = []
  if ann:
    p = ann_images_path
  else:
    p = sim_images_path
  for path, subdirs, files in os.walk(p + '/' + str(class_folder)):
    for name in files:
      if name.endswith('.png'):
        paths.append((os.path.join(path, name)))
  return paths

def eigen_values_vectors(mesh):
  covm = np.cov(mesh.vertices.T)
  values, vectors = np.linalg.eig(covm)
  # values[i] corresponds to vector[:,i}
  return values, vectors


def eigen_angle(mesh):
  x, y, z = eigen_xyz(mesh)
  return min(angle(x, [1, 0, 0]), angle(x, [-1, 0, 0]))


def eigen_xyz(mesh):
  values, vectors = eigen_values_vectors(mesh)
  eig_vector_x = vectors[:, np.argmax(values)]  # largest
  eig_vector_y = vectors[:, np.argsort(values)[1]]  # second largest
  eig_vector_z = np.cross(eig_vector_x, eig_vector_y)
  return eig_vector_x, eig_vector_y, eig_vector_z
