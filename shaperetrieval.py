from numpy.linalg import norm
from scipy.spatial import distance
from trimesh import util
import utils
import analyze

def compute_eucledian_distance(vector1, vector2):
  return norm(vector1, vector2)

def cosine_difference(vector1, vector2):
  return distance.cosine(vector1, vector2)

def find_similar_meshes(mesh):
  #Analyze the mesh
  mesh_info = analyze.fill_mesh_info(mesh, "x", "path", features=True)

  #Get the feature vector 
  df = utils.read_excel(original=False)
  columns = ["volume", "area", "eccentricity", "eigen_x_angle", "diameter", "compactness", "A3", "D1", "D2", "D3", "D4"]
  for index, row in df.iterrows():
    #feature_vector