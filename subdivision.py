import trimesh
import numpy as np


def selection_sort(x):
  for i in range(len(x)):
    swap = i + np.argmin(x[i:])
    (x[i], x[swap]) = (x[swap], x[i])
  return x

def computeaveragecell(mesh):
  return mesh.area / len(mesh.area_faces)

def subdivide(mesh, min_area, show=False):
  n_subdivided = 0
  NC = len(mesh.area_faces)
  NP = len(mesh.vertices)
  NP2 = NP
  NC2 = 0
  updated_vertices = np.asarray(mesh.vertices)
  updated_faces = np.asarray(mesh.faces)
  area_sorted_cells = np.sort(mesh.area_faces)[::-1]
  for index,face in enumerate(mesh.faces):
    if mesh.area_faces[index] >= min_area: # if face area is smaller than cutoff, split the face into 3 new faces
      center = (mesh.vertices[face[0]] + mesh.vertices[face[1]] + mesh.vertices[face[2]]) / 3
      #new vertices
      updated_vertices = np.append(updated_vertices, [center], axis=0)
      center_index = len(updated_vertices) - 1
      #new faces
      face1 = [face[0], face[1], center_index]
      face2 = [face[0], center_index, face[2]]
      face3 = [center_index, face[1], face[2]]
      newfaces = [face1, face2, face3]
      updated_faces = np.delete(updated_faces, index, axis=0)
      updated_faces = np.append(updated_faces, newfaces, axis=0)

      # update mesh statistics
      n_subdivided = n_subdivided + 1
      NP2 = NP2 + 1
      NC2 = NC2 + 3

  print(len(mesh.faces))
  print(len(updated_faces))
  if show:
    newmesh = trimesh.Trimesh(vertices=updated_vertices, faces=updated_faces)
    meshes = [mesh, newmesh]
    for i, m in enumerate(meshes):
      m.apply_translation([0, 0, i * 1])
    trimesh.Scene(meshes).show()


# mesh objects can be created from existing faces and vertex data
# mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], faces=[[0, 1, 2]])

mesh = trimesh.load('testModels/db/17/m1708/m1708.off', force='mesh')
subdivide(mesh, 0.00000000000001, show=True)