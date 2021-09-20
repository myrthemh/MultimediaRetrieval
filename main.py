import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh

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
  mesh = trimesh.load('testModels/db/0/m0/m0.off', force='mesh')
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
  elif (len(mesh.faces) > 50000 and len(mesh.vertices) > 50000) or (
          len(mesh.faces) > 50000 or len(mesh.vertices) > 50000):
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
    save_histogram(df.loc[:, column].values, column, "Meshes", column)


def ensure_dir(file_path):
  directory = os.path.dirname(file_path)
  if not os.path.exists(directory):
    os.makedirs(directory)


# return alle indexes van subsampled meshes
# alle filepaths ophalen van de indexes
# refine_mesh("./testModels/db/0/m0/m0.off", "./testModels/refined_db/0/m0/m0.off")
# filter_database()
refine_outliers(show=False)

df = read_excel()
print(meta_data(df))
save_all_histograms(df)
