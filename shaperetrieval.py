from numpy.linalg import norm
from scipy.spatial import distance

def compute_eucledian_distance(vector1, vector2):
  return norm(vector1, vector2)

def cosine_difference(vector1, vector2):
  return distance.cosine(vector1, vector2)