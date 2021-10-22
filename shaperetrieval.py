from os import path
from numpy import histogram
from numpy.linalg import norm
from scipy.spatial import distance
import numpy as np
import trimesh
import utils
import analyze
import preprocess
from scipy.stats import wasserstein_distance

def compute_eucledian_distance(vector1, vector2):
  return norm(vector1 - vector2)

def cosine_difference(vector1, vector2):
  return distance.cosine(vector1, vector2)

def sortmethod(x):
  return x[0]

def find_similar_meshes(mesh_path):
  #Analyze the mesh
  mesh = trimesh.load(mesh_path, force='mesh')
  mesh_info = analyze.fill_mesh_info(mesh, -1, "path", features=True)
  df = utils.read_excel(original=False)

  #Get feature vectors:
  single_vector = np.asarray([mesh_info[column] for column in utils.scal_features])
  histograms = np.asarray([mesh_info[column] for column in utils.hist_features])
  histogram_vector = [preprocess.sum_divide(x) for x in histograms]

  #Standardize the scalar features:
  with open(utils.norm_vector_path, 'rb') as f:
    vectors = np.load(f)
    single_vector -= vectors[0]
    single_vector /= vectors[1]

  distances = []
  #Compare with all meshes
  with open(utils.emd_norm_vector_path, 'rb') as f:
    emd_vector = np.load(f)
    for index, row in df.iterrows():
      if(mesh_path == row['path']):
        continue
      other_single_vector = np.asarray(row[utils.scal_features_norm])
      other_histogram_vector = np.asarray(row[utils.hist_features_norm])
      scalar_distance = compute_eucledian_distance(single_vector, other_single_vector)
      hist_distances = [wasserstein_distance(histogram_vector[i], other_histogram_vector[i]) for i in range(len(histogram_vector))]
      
      #Standardize histogram distances:
      hist_distances /= emd_vector
      distance = 0.5 * scalar_distance + sum(0.25 * hist_distances)
      print(scalar_distance, hist_distances)
      distances.append((distance, row['path'])) 
  distances.sort(key=sortmethod)
  return distances


distances = find_similar_meshes('testModels/refined_db/0/m0/m0.off')
mesh = trimesh.load('testModels/refined_db/0/m0/m0.off', force='mesh')
mesh.show()
for dist in distances:
  meshx = trimesh.load(dist[1], force='mesh')
  meshx.show()