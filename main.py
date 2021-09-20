import os
import trimesh
import matplotlib.pyplot as plt
import pandas as pd
import logging
import numpy as np
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

# Step 1
def step_1():
  mesh = trimesh.load('testModels/m0/m0.off', force='mesh')
  # mesh = trimesh.load('testModels/bunny.ply', force='mesh')
  mesh.show()

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
          #Find the relevant info for the mesh:
          mesh = trimesh.load(db + '/' + classFolder + '/' + modelFolder + '/' + filename, force='mesh')
          mesh_info = fill_mesh_info(mesh, classFolder)
          df = df.append(mesh_info, ignore_index=True)
  df.to_excel(dataPath)

def fill_mesh_info(mesh, classFolder):
  mesh_info = {}
  mesh_info["class"] = int(classFolder)
  mesh_info["nrfaces"] = len(mesh.faces)
  mesh_info["nrvertices"] = len(mesh.vertices)
  face_sizes = list(map(lambda x: len(x), mesh.faces))
  mesh_info["containsTriangles"] = 3 in face_sizes
  mesh_info["containsQuads"] = 4 in face_sizes
  mesh_info["bounding_box_corners"] = bounding_box(mesh.vertices)
  mesh_info = detect_outliers(mesh, mesh_info)
  return mesh_info

def detect_outliers(mesh, mesh_info):
  if (len(mesh.faces) < 100 and len(mesh.vertices) < 100) or (len(mesh.faces) < 100 or len(mesh.vertices) < 100):
    mesh_info["outlier"] = True
  elif (len(mesh.faces) > 50000 and len(mesh.vertices) > 50000) or (
    len(mesh.faces) > 50000 or len(mesh.vertices) > 50000):
    mesh_info["outlier"] = True
  else:
      mesh_info["outlier"] = False
  return mesh_info

def refine_mesh(inputfile, outputfile):
  command = f'java -jar ./scripts/catmullclark.jar {inputfile} {outputfile}'
  os.system(command)

def read_excel():
  #Load the excel into a pandas df
  return pd.read_excel(dataPath, index_col=0) 

def meta_data(dataframe):
  #Calculate metadata on the datafram
  metadata = {}
  metadata["avgfaces"] = np.mean(dataframe.loc[:,"nrfaces"].values)
  metadata["avgvertices"] = np.mean(dataframe.loc[:,"nrvertices"].values)
  return metadata

def save_histogram(data, xlabel, ylabel, title):
  # the histogram of the data
  plt.hist(data, 50, density=True, facecolor='g', alpha=0.75)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.xlim(0, max(data))
  plt.grid(True)
  plt.show()
  plt.savefig(imagePath + title + '.png')

def save_all_histograms(df):
  histogrammable_columns = ["class", "nrfaces", "nrvertices"]
  for column in histogrammable_columns:
    save_histogram(df.loc[:,column].values, column, "Meshes", column)

def normalize_mesh(path):
  mesh = trimesh.load(path, force='mesh')

  #Center the mass of the mesh on (0,0,0)
  center_mass = mesh.center_mass
  mesh.apply_translation(-center_mass)
  
  #Get the highest value we can scale with so it still fits within the unit cube
  scale_value = 1 / max(mesh.bounds.flatten())

  #Make a vector to scale x, y and z in the mesh to this value
  scaleVector = [scale_value, scale_value, scale_value]

  #Create transformation matrix
  matrix = np.eye(4)
  matrix[:3, :3] *= scaleVector
  mesh.apply_transform(matrix)
  # print(mesh.bounds)
  # print(mesh.center_mass)
  return mesh

#refine_mesh("./testModels/db/0/m0/m0.off", "./testModels/refined_db/0/m0/m0.off")
#filter_database()
# df = read_excel()
# print(meta_data(df))
# save_all_histograms(df)'

normalize_mesh("testModels/db/0/m0/m0.off")

