import numpy as np
import pandas as pd
import trimesh

import analyze
import subdivision
import utils


def scale_mesh(mesh, scale):
  # Make a vector to scale x, y and z in the mesh to this value
  scaleVector = [scale, scale, scale]

  # Create transformation matrix
  matrix = np.eye(4)
  matrix[:3, :3] *= scaleVector
  mesh.apply_transform(matrix)
  return mesh


def calc_center_mass(mesh):
  meshVolume = 0
  temp = (0, 0, 0)

  for face in mesh.faces:
    center = (mesh.vertices[face[0]] + mesh.vertices[face[1]] + mesh.vertices[face[2]]) / 4
    volume = np.dot(mesh.vertices[face[0]], np.cross(mesh.vertices[face[1]], mesh.vertices[face[2]])) / 6
    meshVolume += volume
    temp = center * volume
  meshCenter = temp / meshVolume

  return meshCenter


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


# def refine_outlier(show=False):
#   df = pd.read_excel(utils.excelPath)
#   undersampled = df[df["subsampled_outlier"] == True]
#   for path in undersampled["path"]:
#     refined_path = path[:11] + 'refined_' + path[11:]
#     mesh = trimesh.load(path, force='mesh')
#     refined_mesh = subdivision.subdivide(mesh, utils.target_vertices, show=show)
#     save_mesh(refined_mesh, refined_path)
#     trimesh.exchange.export.export_mesh(refined_mesh, refined_path, file_type="off")

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


def process_all():
  # Perform all preprocessing steps on all meshes:
  df = pd.read_excel(utils.excelPath)
  for index, row in df.iterrows():
    path = row['path']
    mesh = trimesh.load(row['path'])
    refined_path = path[:11] + 'refined_' + path[11:]

    if row['subsampled_outlier']:
      mesh = subdivision.subdivide(mesh, utils.target_vertices)
    if row['supersampled_outlier']:
      mesh = subdivision.superdivide(mesh, utils.target_faces)
    mesh = normalize_mesh(mesh)
    if analyze.barycentre_distance(mesh) < 1:
      save_mesh(mesh, refined_path)
