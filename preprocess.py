import numpy as np
import pandas as pd
import trimesh

import analyze
import subdivision
import utils
import main

def scale_mesh(mesh, scale):
  # Make a vector to scale x, y and z in the mesh to this value
  scaleVector = [scale, scale, scale]

  # Create transformation matrix
  matrix = np.eye(4)
  matrix[:3, :3] *= scaleVector
  mesh.apply_transform(matrix)
  return mesh


def barycenter(mesh):
  # Returns the face area weighted barycentr of the mesh.
  faces = np.asarray(mesh.faces)
  weighted_vertices = []
  for index, face in enumerate(faces):
    weighted_vertices.append(mesh.area_faces[index] * utils.get_face_barycentre(face, mesh.vertices))
  bary_centre = sum(weighted_vertices) / mesh.area
  return bary_centre


def save_mesh(mesh, path):
  utils.ensure_dir(path)
  trimesh.exchange.export.export_mesh(mesh, path, file_type="off")


def normalize_mesh(mesh):
  # Fix normals
  if mesh.body_count > 1:
    trimesh.Trimesh.fix_normals(mesh, multibody=True)
  else:
    trimesh.Trimesh.fix_normals(mesh)

  # Center the mass of the mesh on (0,0,0)
  center_mass = barycenter(mesh)
  mesh.apply_translation(-center_mass)

  # Get the highest value we can scale with so it still fits within the unit cube
  scale_value = 1 / max(abs(mesh.bounds.flatten()))
  mesh = scale_mesh(mesh, scale_value)

  return mesh

def eigen_values_vectors(mesh):
  covm = np.cov(mesh.vertices.T)
  values, vectors = np.linalg.eig(covm)
  #values[i] corresponds to vector[:,i}
  return values, vectors

def eigen_xyz(mesh):
  values, vectors = eigen_values_vectors(mesh)
  eig_vector_x = vectors[:,np.argmax(values)]  #largest
  eig_vector_y = vectors[:,np.argsort(values)[1]]  # second largest
  eig_vector_z = np.cross(eig_vector_x, eig_vector_y)
  return eig_vector_x, eig_vector_y, eig_vector_z

def translate_eigen(mesh):
  c = barycenter(mesh)
  eig_vector_x, eig_vector_y, eig_vector_z = eigen_xyz(mesh)
  for vertex in mesh.vertices:
    vc = vertex - c
    vertex[0] = np.dot(vc, eig_vector_x)
    vertex[1] = np.dot(vc, eig_vector_y)
    vertex[2] = np.dot(vc, eig_vector_z)
  return mesh


def process_all(show_subdivide=False, show_superdivide=False):
  # Perform all preprocessing steps on all meshes:
  df = pd.read_excel(utils.excelPath)
  for index, row in df.iterrows():
    path = row['path']
    mesh = trimesh.load(path)
    refined_path = utils.refined_path(path)

    if row['subsampled_outlier']:
      mesh2 = subdivision.subdivide(mesh, utils.target_vertices)
      if show_subdivide:
        main.compare([mesh, mesh2])
    if row['supersampled_outlier']:
      mesh2 = subdivision.superdivide(mesh, utils.target_faces)
      if show_superdivide:
        main.compare([mesh, mesh2])
    else:
      mesh2 = mesh

    mesh2 = normalize_mesh(mesh2)
    mesh2 = translate_eigen(mesh2)
    if analyze.barycentre_distance(mesh2) < 1:
      save_mesh(mesh2, refined_path)
