import numpy as np
import trimesh

import utils


def sortFunction(x):
  return x[1]


def get_area_indices_list(mesh):
  area_with_index = []
  for index, area in enumerate(mesh.area_faces):
    area_with_index.append((index, area))
  area_with_index.sort(key=sortFunction, reverse=True)
  return area_with_index


def subdivide(mesh, target_vertices=1000):
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

    # new vertices
    updated_vertices = np.append(updated_vertices, [center], axis=0)
    center_index = len(updated_vertices) - 1

    # new faces
    face1 = [face[0], face[1], center_index]
    face2 = [face[0], center_index, face[2]]
    face3 = [center_index, face[1], face[2]]
    newfaces = [face1, face2, face3]

    updated_faces = np.append(updated_faces, newfaces, axis=0)
    indices_to_delete.append(index)

    counter += 1

  # batch delete all 'old' triangles that have been subdivided
  updated_faces = np.delete(updated_faces, indices_to_delete, axis=0)
  newmesh = trimesh.Trimesh(vertices=updated_vertices, faces=updated_faces, process=False)
  return newmesh


def test_subdivide(mesh, target_vertices=1000):
  updated_vertices = np.asarray(mesh.vertices)
  updated_faces = np.asarray(mesh.faces)
  updated_edges = np.asarray(mesh.edges)
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
    vertices = [updated_vertices[face[0]], updated_vertices[face[1]], updated_vertices[face[2]]]
    angles = [utils.angle(vertices[0] - vertices[1], vertices[0] - vertices[2]),
              utils.angle(vertices[1] - vertices[0], vertices[1] - vertices[2]),
              utils.angle(vertices[2] - vertices[0], vertices[2] - vertices[1])]
    sorted_vertices = np.argsort(angles)
    center = (vertices[sorted_vertices[0]] + vertices[sorted_vertices[1]]) / 2

    # new vertices
    updated_vertices = np.append(updated_vertices, [center], axis=0)
    center_index = len(updated_vertices) - 1

    # new faces
    face1 = [face[sorted_vertices[0]], center_index, face[sorted_vertices[2]]]
    face2 = [center_index, face[sorted_vertices[1]], face[sorted_vertices[2]]]

    newfaces = [face1, face2]

    updated_faces = np.append(updated_faces, newfaces, axis=0)
    indices_to_delete.append(index)

    # adjacent face
    for edge_index, edge in enumerate(mesh.edges):
      if edge.tolist() == [face[sorted_vertices[0]], face[sorted_vertices[1]]] or edge.tolist() == [face[sorted_vertices[1]], face[sorted_vertices[0]]]:
        face_index_edge = mesh.edges_face[edge_index]
        if face_index_edge != index:
          v4 = np.setdiff1d(updated_faces[face_index_edge], edge).item()
          face1 = [face[sorted_vertices[0]], center_index, v4]
          face2 = [center_index, v4, face[sorted_vertices[1]]]
          newfaces = [face1, face2]
          updated_faces = np.append(updated_faces, newfaces, axis=0)
          indices_to_delete.append(face_index_edge)
          # print(indices_to_delete[-1] == face_index_edge)
    counter += 1
  updated_faces = np.delete(updated_faces, indices_to_delete, axis=0)
  newmesh = trimesh.Trimesh(vertices=updated_vertices, faces=updated_faces, process=True)
  return newmesh


def superdivide(mesh, target_faces=2000):
  newmesh = trimesh.Trimesh.simplify_quadratic_decimation(mesh, target_faces)
  return newmesh


# mesh = trimesh.load('testModels/db/0/m3/m3.off', force='mesh')
# mesh2 = new_subdivide(mesh)
# print(" ")
# print(mesh.is_watertight)
# trimesh.Trimesh.fix_normals(mesh2, multibody=True)
# print(mesh2.is_watertight)
# meshes = [mesh, mesh2]
# for i, m in enumerate(meshes):
#   m.apply_translation([0, 0, i])
#   m.visual.vertex_colors = trimesh.visual.random_color()
# trimesh.Scene(meshes).show()
