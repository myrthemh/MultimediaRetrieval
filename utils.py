import os
from matplotlib.pyplot import cla

import numpy as np
import pandas as pd
from trimesh import util

excelPath = "features/original.xlsx"
refinedexcelPath = "features/refined.xlsx"
picklePath = "features/original.pkl"
refinedpicklePath = "features/refined.pkl"
imagePath = "graphs/original/"
refinedImagePath = "graphs/refined/"
# originalDB = "testModels/db"
# refinedDB = "testModels/refined_db"
sim_images_path = "simimages/"
ann_images_path = "annimages/"
eval_images_path = "evalimages/"
sim_image_size = 256
originalDB = "veelModels/db"
refinedDB = "veelModels/refined_db"
target_vertices = 5000
target_faces = 10000
nr_bins_hist = 20
hist_amount = 1000000
query_size = 5
hist_features = ['A3', 'D1', 'D2', 'D3', 'D4']
scal_features = ["area", "axis-aligned_bounding_box_volume", "diameter", "compactness", "eccentricity", "volume"]
scal_features_norm = ["area_norm", "axis-aligned_bounding_box_volume_norm", "diameter_norm", "compactness_norm",
                      "eccentricity_norm", "volume_norm"]
hist_features_norm = ["A3_norm", "D1_norm", "D2_norm", "D3_norm", "D4_norm"]
norm_vector_path = "features/vector.npy"
emd_norm_vector_path = "features/dist_vector.npy"

weight_vectors = np.array([[1,1,1,1,1,1,1,1,1,1,1],
                          # [1,0.3,0.3,0.1,0.3,0.5,1,0.3,1,0.3,2],
                          # [2,0.3,0.3,0.1,0.3,0.5,1,0.3,1,0.3,2],
                          # [1,0.3,0.3,0.1,0.3,0.5,1,0.3,1,0.3,1],
                          # [1,0.5,0.5,0.1,0.5,0.7,1,0.5,1,0.5,1],
                          # [2,0.2,0.2,0.1,0.2,0.7,1,0.2,1,0.2,2],
                          # [2,0.2,0.2,0.05,0.2,0.35,1,0.2,1,0.2,2],
                          # [2,0.2,0.2,0.1,0.2,0.35,1,0.2,1,0.2,2]
                       ])

# weight_vectors = np.array([[1,1,1,1,1,1,1,1,1,1,1],
#                             [1,0,0,0,0,0,0,0,0,0,0],
#                             [0,1,0,0,0,0,0,0,0,0,0],
#                             [0,0,1,0,0,0,0,0,0,0,0],
#                             [0,0,0,1,0,0,0,0,0,0,0],
#                             [0,0,0,0,1,0,0,0,0,0,0],
#                             [0,0,0,0,0,1,0,0,0,0,0],
#                             [0,0,0,0,0,0,1,0,0,0,0],
#                             [0,0,0,0,0,0,0,1,0,0,0],
#                             [0,0,0,0,0,0,0,0,1,0,0],
#                             [0,0,0,0,0,0,0,0,0,1,0],
#                             [0,0,0,0,0,0,0,0,0,0,1],
#                             [0,1,1,1,1,1,1,1,1,1,1],
#                             [1,0,1,1,1,1,1,1,1,1,1],
#                             [1,1,0,1,1,1,1,1,1,1,1],
#                             [1,1,1,0,1,1,1,1,1,1,1],
#                             [1,1,1,1,0,1,1,1,1,1,1],
#                             [1,1,1,1,1,0,1,1,1,1,1],
#                             [1,1,1,1,1,1,0,1,1,1,1],
#                             [1,1,1,1,1,1,1,0,1,1,1],
#                             [1,1,1,1,1,1,1,1,0,1,1],
#                             [1,1,1,1,1,1,1,1,1,0,1],
#                             [1,1,1,1,1,1,1,1,1,1,0],])


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


def unit_vector(vector, transpose=False):
  if transpose:
    # print(np.divide(vector.T, np.linalg.norm(vector, axis=1)).T)
    return np.divide(vector.T, np.linalg.norm(vector, axis=1)).T
  # print(np.divide(vector, np.linalg.norm(vector)))
  return np.divide(vector, np.linalg.norm(vector))


def angle(vector1, vector2):
  return np.arccos(np.clip(np.dot(unit_vector(vector1), unit_vector(vector2)), -1.0, 1.0))

def angle_points(points):
  return np.arccos(np.clip(np.dot(unit_vector(points[0] - points[1]), unit_vector(points[0] - points[2])), -1.0, 1.0))

def shape_paths(dbfolder):
  paths = []
  for path, subdirs, files in os.walk(dbfolder):
    for name in files:
      if name.endswith('.off'):
        paths.append((os.path.join(path, name)))
  return paths


def image_paths(class_folder, ann=False):
  df = read_excel(original=False)
  paths = []
  if ann:
    p = ann_images_path
  else:
    p = sim_images_path
  for path, subdirs, files in os.walk(p + '/' + classes[class_folder]):
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

def class_dictionaries():
  f = open("veelModels/classification/v1/coarse1/coarse1Train.cla", "r")
  f2 = open("veelModels/classification/v1/coarse1/coarse1Test.cla", "r")
  index_to_class = {}
  class_sizes = {}
  class_indices = {}
  for file in [f,f2]:
    for x in file:
      words = x.split()
      if len(words) == 3:
        currentClass = words[0]
        if currentClass in class_sizes:
          class_sizes[currentClass] += int(words[2])
        else:
          class_sizes[currentClass] = int(words[2])
      if len(words) == 1:
        index_to_class[int(words[0])] = currentClass
  sorted_list = dict(sorted(class_sizes.items(), key=lambda item: item[1], reverse=True)[:15])
  return index_to_class, sorted_list

classes = list(class_dictionaries()[1].keys())
class_dictionaries()