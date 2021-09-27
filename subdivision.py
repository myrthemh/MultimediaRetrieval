import trimesh
import numpy as np


def selection_sort(x):
  for i in range(len(x)):
    swap = i + np.argmin(x[i:])
    (x[i], x[swap]) = (x[swap], x[i])
  return x

def computeaveragecell(mesh):
  return mesh.area / len(mesh.area_faces)

def sortFunction(x):
  return x[1]

def get_area_indices_list(mesh):
  area_with_index = []
  for index, area in enumerate(mesh.area_faces):
    area_with_index.append((index, area))
  area_with_index.sort(key=sortFunction, reverse=True)
  return area_with_index

def subdivide(mesh, target_vertices=1000, show=False):
  n_subdivided = 0
  NC = len(mesh.area_faces)
  NP = len(mesh.vertices)
  NP2 = NP
  NC2 = 0
  updated_vertices = np.asarray(mesh.vertices)
  updated_faces = np.asarray(mesh.faces)
  indices_to_delete = []
  area_with_index = get_area_indices_list(mesh)
  counter = 0
  while len(updated_vertices) < target_vertices:
    if counter > len(area_with_index) - 1:
      updated_faces = np.delete(updated_faces, indices_to_delete, axis=0)
      indices_to_delete.clear()
      newmesh = trimesh.Trimesh(vertices=updated_vertices, faces=updated_faces)
      area_with_index = get_area_indices_list(newmesh)
      counter = 0

    index, area = area_with_index[counter]
    face = updated_faces[index]
    center = (updated_vertices[face[0]] + updated_vertices[face[1]] + updated_vertices[face[2]]) / 3

    #new vertices
    updated_vertices = np.append(updated_vertices, [center], axis=0)
    center_index = len(updated_vertices) - 1

    #new faces
    face1 = [face[0], face[1], center_index]
    face2 = [face[0], center_index, face[2]]
    face3 = [center_index, face[1], face[2]]
    newfaces = [face1, face2, face3]

    updated_faces = np.append(updated_faces, newfaces, axis=0)
    indices_to_delete.append(index)

    # update mesh statistics (delete? not currently using)
    n_subdivided = n_subdivided + 1
    NP2 = NP2 + 1
    NC2 = NC2 + 3
    counter += 1

    #batch delete all 'old' triangles that have been subdivided
  updated_faces = np.delete(updated_faces, indices_to_delete, axis=0)
  newmesh = trimesh.Trimesh(vertices=updated_vertices, faces=updated_faces, process=False)
  if show:
    meshes = [mesh, newmesh]
    for i, m in enumerate(meshes):
      m.apply_translation([0, 0, i * 1])

    trimesh.Scene(meshes).show()
  return newmesh

def superdivide(mesh, target_faces=2000, show=False):
  newmesh = trimesh.Trimesh.simplify_quadratic_decimation(mesh, target_faces)
  if show:
    meshes = [mesh, newmesh]
    for i, m in enumerate(meshes):
      m.apply_translation([0, 0, i * 1])
    trimesh.Scene(meshes).show()
  return newmesh

