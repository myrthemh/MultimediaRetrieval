import numpy as np
import trimesh
import trimesh.grouping as grouping

import analyze
import main
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


def save_mesh(mesh, path):
  utils.ensure_dir(path)
  trimesh.exchange.export.export_mesh(mesh, path, file_type="off")


def make_watertight(mesh):
  edges_sorted = np.sort(mesh.edges, axis=1)
  # group sorted edges
  groups1 = grouping.group_rows(
    edges_sorted, require_count=1)
  x = list(mesh.edges[groups1])
  # Find all loops of edges that define the different holes
  loops = []
  while len(x) > 0:
    loop = []
    loop.append(x[0])
    del x[0]
    while loop[-1][1] != loop[0][0]:
      check = len(x)
      for index, edge in enumerate(x):
        if edge[0] == loop[-1][1]:
          loop.append(edge)
          del x[index]
          break
      if len(x) == check:
        print("Could not create loop, fixing watertightness failed")
        return mesh
    loops.append(loop)
  newfaces = mesh.faces
  newvertices = mesh.vertices

  # Create a vertice in the center of each loop, and make a face by connecting each edge in the loop to the new vertice.
  for loop in loops:
    unique_vertices_in_loop = mesh.vertices[[x[0] for x in loop]]
    barycentre = sum(unique_vertices_in_loop) / len(unique_vertices_in_loop)
    newvertices = np.append(newvertices, [barycentre], axis=0)
    for edge in loop:
      newfaces = np.append(newfaces, [[edge[0], edge[1], len(newvertices) - 1]], axis=0)
  newmesh = trimesh.Trimesh(vertices=newvertices, faces=newfaces, process=True)

  # Fix normals
  fix_normals(newmesh)
  return newmesh


def normalize_mesh(mesh):
  # Fix normals
  fix_normals(mesh)

  # Center the mass of the mesh on (0,0,0)
  mesh.apply_translation(-mesh.centroid)

  mesh = translate_eigen(mesh)
  # Get the highest value we can scale with so it still fits within the unit cube
  scale_value = 0.5 / max(abs(mesh.bounds.flatten()))
  mesh = scale_mesh(mesh, scale_value)
  return mesh


def fix_normals(mesh):
  # check if all edge pairs are unique, if not, fix normals
  if len(np.unique(mesh.edges, axis=0) != len(mesh.edges)):
    trimesh.Trimesh.fix_normals(mesh, multibody=True) if mesh.body_count > 1 else trimesh.Trimesh.fix_normals(mesh)


def eigen_values_vectors(mesh):
  covm = np.cov(mesh.vertices.T)
  values, vectors = np.linalg.eig(covm)
  # values[i] corresponds to vector[:,i}
  return values, vectors


def eigen_angle(mesh):
  x, y, z = eigen_xyz(mesh)
  return min(utils.angle(x, [1, 0, 0]), utils.angle(x, [-1, 0, 0]))


def eigen_xyz(mesh):
  values, vectors = eigen_values_vectors(mesh)
  eig_vector_x = vectors[:, np.argmax(values)]  # largest
  eig_vector_y = vectors[:, np.argsort(values)[1]]  # second largest
  eig_vector_z = np.cross(eig_vector_x, eig_vector_y)
  return eig_vector_x, eig_vector_y, eig_vector_z


def translate_eigen(mesh):
  eig_vector_x, eig_vector_y, eig_vector_z = eigen_xyz(mesh)
  for vertex in mesh.vertices:
    vc = np.copy(vertex)
    vertex[0] = np.dot(vc, eig_vector_x)
    vertex[1] = np.dot(vc, eig_vector_y)
    vertex[2] = np.dot(vc, eig_vector_z)
  return mesh


def normalize_histogram_features(features):
  df = utils.read_excel(original=False)
  features_norm = [i+"_norm" for i in features]
  subset = df[features]
  subset_norm = subset.applymap(sum_divide)
  subset_norm.columns = features_norm
  df[features_norm] = subset_norm
  utils.save_excel(df, original=False)

def sum_divide(x):
  return x / sum(x)


def process_all(show_subdivide=False, show_superdivide=False):
  # Perform all preprocessing steps on all meshes:
  df = utils.read_excel(original=True)
  i = 0
  y = 0
  z = 0
  for index, row in df.iterrows():
    path = row['path']
    print(f"preprocessing {path}")
    mesh = trimesh.load(path)
    refined_path = utils.refined_path(path)
    if not mesh.is_watertight:
      print("--------------------------------------------------")
      mesh = make_watertight(mesh)
      if not mesh.is_watertight:
        print("Make watertight operation failed")
      else:
        print("Successfully made mesh watertight")
    print("remeshing")
    if row['subsampled_outlier']:
      mesh2 = subdivision.subdivide(mesh, utils.target_vertices)
      if show_subdivide:
        main.compare([mesh, mesh2])
    elif row['supersampled_outlier']:
      mesh2 = subdivision.superdivide(mesh, utils.target_faces)
      if show_superdivide:
        main.compare([mesh, mesh2])
    else:
      mesh2 = mesh
    mesh2 = normalize_mesh(mesh2)
    save_mesh(mesh2, refined_path)
    if not mesh.is_watertight:
      i += 1
    if not mesh2.is_watertight:
      y += 1


  normalize_histogram_features(features=utils.hist_features)
  scalar_normalization(features=utils.scal_features)
  print(f'meshes filtered: {z}')
  print(f'meshes1 not watertight: {i}')
  print(f'meshes2 not watertight: {y}')

def scalar_normalization(features):
  df = utils.read_excel(original=False)
  y = (df[features]-df[features].mean())/df[features].std()
  y = (y+1) /2
  y = y.rename(columns=lambda x: x+"_norm")
  df[y.columns] = y
  vectors = np.asarray([df[features].mean(), df[features].std(), [1,1,1,1,1], [2,2,2,2,2]])
  #Store the values used for normalization so we can normalize new query objects
  with open(utils.norm_vector_path, 'wb') as f:
    np.save(f, vectors)
  utils.save_excel(df, original=False)