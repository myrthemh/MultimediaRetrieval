import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh
import pyrender

trimesh.util.attach_to_log()
logging.getLogger('matplotlib.font_manager').disabled = True

dataPath = "features/data.xlsx"
imagePath = "graphs/"

classes = [
  "Insect",  # 0
  "Farm animal",  # 1
  "People",  # 2
  "Face",  # 3
  "Building",  # 4
  "Container",  # 5
  "LampOrWatch",  # 6
  "Stabweapon",  # 7
  "Chair",  # 8
  "Table",  # 9
  "Flowerpot",  # 10
  "Tool",  # 11
  "Airplane",  # 12
  "Aircraft",  # 13
  "Spacecraft",  # 14
  "Car",  # 15
  "Chess piece",  # 16
  "DoorOrChest",  # 17
  "Satellite"  # 18
]
def scale_mesh(mesh, scale):
  #Make a vector to scale x, y and z in the mesh to this value
  scaleVector = [scale, scale, scale]

  #Create transformation matrix
  matrix = np.eye(4)
  matrix[:3, :3] *= scaleVector
  mesh.apply_transform(matrix)
  return mesh

# Step 1
def step_1():
  mesh1 = trimesh.load('testModels/db/0/m0/m0.off', force='mesh')
  mesh2 = trimesh.load('testModels/db/0/m0/m0.off')
  mesh2 = scale_mesh(mesh2, 1.001)
  material = pyrender.Material()
  mesh1 = pyrender.Mesh.from_trimesh(mesh1, smooth=False)
  mesh2 = pyrender.Mesh.from_trimesh(mesh2, wireframe=True, smooth=False, material=material)
  scene = pyrender.Scene()
  scene.add(mesh1)
  scene.add(mesh2)
  
  pyrender.Viewer(scene, use_raymond_lighting=True)

step_1()
# Step 2

# The key constraints here are that (a) the reduced database should contain at least 200 shapes; (b) you should have
# shapes of most (ideally all) of the existing class types in the database; (c) try to balance the classes, i.e.,
# do not use tens of shapes of one class and only a handful of shapes of another class. That is: Start small. When
# this works, add more shapes to your (already) functioning code, and repeat the tests.
#
# Start building a simple filter that checks all shapes in the database. The filter should output, for each shape
#
# - the class of the shape
# - the number of faces and vertices of the shape
# - the type of faces (e.g. only triangles, only quads, mixes of triangles and quads)
# - the axis-aligned 3D bounding box of the shapes

def bounding_box(vertices):
  # Find the two corners of the bounding box surrounding the mesh.
  # Bottom will contain the lowest x, y and z values, while 'top' contains the highest values in the mesh.
  bottom = vertices[0].copy()
  top = vertices[0].copy()
  for vertex in vertices:
    for dimension in [0, 1, 2]:
      bottom[dimension] = min(bottom[dimension], vertex[dimension])
      top[dimension] = max(top[dimension], vertex[dimension])
  return (bottom, top)


def filter_database():
  db = 'testModels/db'
  df = pd.DataFrame()
  # iterate over all models:
  for classFolder in os.listdir(db):
    for modelFolder in os.listdir(db + '/' + classFolder):
      for filename in os.listdir(db + '/' + classFolder + '/' + modelFolder):
        if filename.endswith('.off'):
          # Find the relevant info for the mesh:
          path = db + '/' + classFolder + '/' + modelFolder + '/' + filename
          mesh = trimesh.load(path, force='mesh')
          mesh_info = fill_mesh_info(mesh, classFolder, path)
          df = df.append(mesh_info, ignore_index=True)
  df.to_excel(dataPath)
  refine_outliers(df)


def fill_mesh_info(mesh, classFolder, path):
  face_sizes = list(map(lambda x: len(x), mesh.faces))
  mesh_info = {"class": int(classFolder), "nrfaces": len(mesh.faces), "nrvertices": len(mesh.vertices),
               "containsTriangles": 3 in face_sizes, "containsQuads": 4 in face_sizes,
               "bounding_box_corners": bounding_box(mesh.vertices), "path": f'{path}'}
  mesh_info = detect_outliers(mesh, mesh_info)
  return mesh_info


def detect_outliers(mesh, mesh_info):
  if (len(mesh.faces) < 100 and len(mesh.vertices) < 100) or (len(mesh.faces) < 100 or len(mesh.vertices) < 100):
    mesh_info["subsampled_outlier"] = True
    mesh_info["supersampled_outlier"] = False
  elif (len(mesh.faces) > 15000 or len(mesh.vertices) > 15000):
    mesh_info["supersampled_outlier"] = True
    mesh_info["subsampled_outlier"] = False
  else:
    mesh_info["subsampled_outlier"] = False
    mesh_info["supersampled_outlier"] = False
  return mesh_info


def refine_outliers(show=False):
  df = pd.read_excel(dataPath)
  undersampled = df[df["subsampled_outlier"] == True]
  for path in undersampled["path"]:
    refined_path = path[:11] + 'refined_' + path[11:]
    refine_mesh(path, refined_path)
    if show:
      mesh1 = trimesh.load(refined_path, force='mesh')
      mesh2 = trimesh.load(path, force='mesh')
      meshes = [mesh1, mesh2]
      for i, m in enumerate(meshes):
        m.apply_translation([0, 0, i * 1])

      trimesh.Scene(meshes).show()


def refine_mesh(inputfile, outputfile):
  ensure_dir(outputfile)
  command = f'java -jar ./scripts/catmullclark.jar {inputfile} {outputfile}'
  os.system(command)


def read_excel():
  # Load the excel into a pandas df
  return pd.read_excel(dataPath, index_col=0)


def meta_data(dataframe):
  # Calculate metadata on the datafram
  metadata = {}
  metadata["avgfaces"] = np.mean(dataframe.loc[:, "nrfaces"].values)
  metadata["avgvertices"] = np.mean(dataframe.loc[:, "nrvertices"].values)
  return metadata

def save_histogram(data, info):
  # the histogram of the data
  plt.hist(data, info["blocksize"], facecolor='g', alpha=0.75)
  plt.xlabel(info["xlabel"])
  plt.ylabel(info["ylabel"])
  plt.title(info["title"])
  plt.xlim(0, max(data))
  plt.grid(True)
  plt.gcf().subplots_adjust(left=0.15)
  plt.savefig(imagePath + info["title"] + '.png')
  plt.clf()


def save_all_histograms(df):
  plotInfos = [
    {"column": "class", "title": "Class distribution", "blocksize": 19, "ylabel": "#Meshes", "xlabel": "Class nr"},
    {"column": "nrfaces", "title": "Face distribution", "blocksize": 50, "ylabel": "#Meshes", "xlabel": "Number of faces"},
    {"column": "nrvertices", "title": "Vertice distribution", "blocksize": 50, "ylabel": "#Meshes", "xlabel": "Number of vertices"},
  ]
  for info in plotInfos:
    save_histogram(df.loc[:,info['column']].values, info)


def ensure_dir(file_path):
  directory = os.path.dirname(file_path)
  if not os.path.exists(directory):
    os.makedirs(directory)

def normalize_mesh(path):
  mesh = trimesh.load(path, force='mesh')

  #Center the mass of the mesh on (0,0,0)
  center_mass = mesh.center_mass
  mesh.apply_translation(-center_mass)
  
  #Get the highest value we can scale with so it still fits within the unit cube
  scale_value = 1 / max(mesh.bounds.flatten())
  mesh = scale_mesh(mesh, scale_value)
  
  # print(mesh.bounds)
  # print(mesh.center_mass)
  return mesh

#filter_database()
# df = read_excel()
# print(meta_data(df))
# save_all_histograms(df)
# normalize_mesh("testModels/db/0/m0/m0.off")

# return alle indexes van subsampled meshes
# alle filepaths ophalen van de indexes
# refine_mesh("./testModels/db/0/m0/m0.off", "./testModels/refined_db/0/m0/m0.off")
# filter_database()
#refine_outliers(show=False)
  #Define plot configuration for each plot:
