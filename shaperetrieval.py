from os import path
from numpy import histogram
from numpy.linalg import norm
from scipy.spatial import distance
import numpy as np
import trimesh
import utils
import analyze

def compute_eucledian_distance(vector1, vector2):
  return norm(vector1 - vector2)

def cosine_difference(vector1, vector2):
  return distance.cosine(vector1, vector2)

def sortmethod(x):
  return x[0]

def find_similar_meshes(mesh):
  #Analyze the mesh
  mesh_info = analyze.fill_mesh_info(mesh, -1, "path", features=True)
  #Get the feature vector 
  df = utils.read_excel(original=False)
  single_features = ["volume", "area", "eccentricity", "eigen_x_angle", "diameter", "compactness"]
  histogram_features = ["A3", "D1", "D2", "D3", "D4"]

  #Get feature vector:
  single_vector = np.asarray([mesh_info[column] for column in single_features])
  histogram_vector = np.concatenate(np.asarray([mesh_info[column] for column in histogram_features]))
  feature_vector = np.concatenate([single_vector, histogram_vector])

  distances = []
  #Compare with all meshes
  for index, row in df.iterrows():
    other_single_vector = np.asarray(row[single_features])
    other_histogram_vector = np.concatenate(np.asarray(row[histogram_features]))
    other_feature_vector = np.concatenate([other_single_vector, other_histogram_vector])

    distance = compute_eucledian_distance(feature_vector, other_feature_vector)
    distances.append((distance, row['path'])) 
  distances.sort(key=sortmethod)
  return distances

mesh = trimesh.load('testModels/refined_db/1/m112/m112.off', force='mesh')
distances = find_similar_meshes(mesh)
mesh.show()
for dist in distances:
  meshx = trimesh.load(dist[1], force='mesh')
  meshx.show()