from os import path
from numpy import histogram
from numpy.linalg import norm
from scipy.spatial import distance
import numpy as np
import trimesh
import utils
import analyze
import preprocess

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
  single_features = ["volume_norm", "area_norm", "eccentricity_norm", "eigen_x_angle_norm", "diameter_norm", "compactness_norm"]
  histogram_features = ["A3_norm", "D1_norm", "D2_norm", "D3_norm", "D4_norm"]

  #Get feature vector:
  single_vector = np.asarray([mesh_info[column] for column in utils.scal_features])
  histograms = np.asarray([mesh_info[column] for column in utils.hist_features])
  histogram_vector = np.concatenate([preprocess.sum_divide(x) for x in histograms])
  with open(utils.norm_vector_path, 'rb') as f:
    vectors = np.load(f)
    single_vector -= vectors[0]
    single_vector /= vectors[1]
    single_vector += vectors[2]
    single_vector /= vectors[3] 
  feature_vector = np.concatenate([single_vector, histogram_vector])
  
  distances = []
  #Compare with all meshes
  for index, row in df.iterrows():
    other_single_vector = np.asarray(row[utils.scal_features_norm])
    other_histogram_vector = np.concatenate(np.asarray(row[utils.hist_features_norm]))
    other_feature_vector = np.concatenate([other_single_vector, other_histogram_vector])
    distance = compute_eucledian_distance(feature_vector, other_feature_vector)
    distances.append((distance, row['path'])) 
  distances.sort(key=sortmethod)
  return distances

mesh = trimesh.load('testModels/refined_db/0/m0/m0.off', force='mesh')
distances = find_similar_meshes(mesh)
mesh.show()
for dist in distances:
  meshx = trimesh.load(dist[1], force='mesh')
  meshx.show()