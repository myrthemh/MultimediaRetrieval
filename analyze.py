import logging
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh

import preprocess
import utils

trimesh.util.attach_to_log()
logging.getLogger('matplotlib.font_manager').disabled = True

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


def volume(mesh):
  subvolume = 0
  for index in range(len(mesh.faces)):
    v1 = mesh.vertices[mesh.faces[index][0]] - mesh.centroid
    v2 = mesh.vertices[mesh.faces[index][1]] - mesh.centroid
    v3 = mesh.vertices[mesh.faces[index][2]] - mesh.centroid
    subvolume += np.dot(np.cross(v1, v2), v3)
  v = 1 / 6 * abs(subvolume)

  return v


def compactness(mesh):
  c = (mesh.area ** 3) / (36 * np.pi * (volume(mesh) ** 2))
  return c


def eccentricity(mesh):
  values, _ = preprocess.eigen_values_vectors(mesh)
  ecc = values[np.argmin(values)] / values[np.argmax(values)]
  return ecc


def barycentre_distance(mesh):
  return math.sqrt(sum(mesh.centroid * mesh.centroid))


def bounding_box_volume(mesh):
  x = mesh.bounds
  volume = (x[1][0] - x[0][0]) * (x[1][1] - x[0][1]) * (x[1][2] - x[0][2])
  return volume


def filter_database(dbPath, excelPath):
  db = dbPath
  df = pd.DataFrame()
  utils.ensure_dir(excelPath)
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
  df.to_excel(excelPath)


def fill_mesh_info(mesh, classFolder, path):
  face_sizes = list(map(lambda x: len(x), mesh.faces))
  print(f"analyzing model {path}")
  mesh_info = {"class": int(classFolder), "nrfaces": len(mesh.faces), "nrvertices": len(mesh.vertices),
               "containsTriangles": 3 in face_sizes, "containsQuads": 4 in face_sizes,
               "bounding_box_corners": mesh.bounds, "path": f'{path}',
               "axis-aligned_bounding_box_distance": np.linalg.norm(mesh.bounds[0] - mesh.bounds[1]),
               "barycentre_distance": barycentre_distance(mesh),
               "volume": bounding_box_volume(mesh),
               "area": mesh.area,
               "eccentricity": eccentricity(mesh),
               "compactness": compactness(mesh)}
  mesh_info = detect_outliers(mesh, mesh_info)
  return mesh_info


def detect_outliers(mesh, mesh_info):
  if len(mesh.vertices) < 900:
    mesh_info["subsampled_outlier"] = True
    mesh_info["supersampled_outlier"] = False
  elif len(mesh.vertices) > 1100:
    mesh_info["supersampled_outlier"] = True
    mesh_info["subsampled_outlier"] = False
  else:
    mesh_info["subsampled_outlier"] = False
    mesh_info["supersampled_outlier"] = False
  return mesh_info


def meta_data(dataframe):
  # Calculate metadata on the datafram
  metadata = {}
  metadata["avgfaces"] = np.mean(dataframe.loc[:, "nrfaces"].values)
  metadata["minfaces"] = np.min(dataframe.loc[:, "nrfaces"].values)
  metadata["maxfaces"] = np.max(dataframe.loc[:, "nrfaces"].values)

  metadata["avgvertices"] = np.mean(dataframe.loc[:, "nrvertices"].values)
  metadata["minvertices"] = np.min(dataframe.loc[:, "nrvertices"].values)
  metadata["maxvertices"] = np.max(dataframe.loc[:, "nrvertices"].values)

  metadata["avgbarycentre_distance"] = np.mean(dataframe.loc[:, "barycentre_distance"].values)
  metadata["volume"] = np.mean(dataframe.loc[:, "volume"].values)

  return metadata


def save_histogram(data, info, path):
  # the histogram of the data
  if info['skip_outliers']:
    # Remove all data below the 5th percentile and 95th percentile
    p5 = np.percentile(data, 5)
    p95 = np.percentile(data, 95)
    data = data[data >= p5]
    data = data[data <= p95]
  if info["xlim"] > 0:
    bins = np.arange(0, info["xlim"], info["xlim"] / info["blocksize"])
  else:
    bins = info["blocksize"]
  plt.hist(data, bins=bins, facecolor='g', alpha=0.75)
  plt.xlabel(info["xlabel"])
  plt.ylabel(info["ylabel"])
  plt.title(info["title"])
  if info["xlim"] != 0:
    plt.xlim(0, info["xlim"])
  # plt.grid(True)
  plt.gcf().subplots_adjust(left=0.15)
  utils.ensure_dir(path)
  plt.savefig(path + info["title"] + '.png')
  plt.clf()


def save_all_histograms(df, path):
  meta_data(df)
  plotInfos = [
    {"column": "class", "title": "Class distribution", "blocksize": 19, "xlim": 18, "ylabel": "#Meshes",
     "xlabel": "Class nr", "skip_outliers": False},
    {"column": "nrfaces", "title": "Face distribution", "blocksize": 100, "xlim": 0, "ylabel": "#Meshes",
     "xlabel": "Number of faces", "skip_outliers": True},
    {"column": "nrvertices", "title": "Vertice distribution", "blocksize": 100, "xlim": 0, "ylabel": "#Meshes",
     "xlabel": "Number of vertices", "skip_outliers": True},
    {"column": "volume", "title": "Bounding box volume", "blocksize": 100, "xlim": 0, "ylabel": "#Meshes",
     "xlabel": "Bounding box volume", "skip_outliers": False},
    {"column": "barycentre_distance", "title": "Barycentre origin distance", "blocksize": 20, "xlim": 1,
     "ylabel": "#Meshes", "xlabel": "Distance barycentre to origin", "skip_outliers": False},
    {"column": "axis-aligned_bounding_box_distance", "title": "Axis-aligned bounding box distance", "blocksize": 50,
     "xlim": 3,
     "ylabel": "#Meshes", "xlabel": "Diagonal distance of axis aligned bounding box", "skip_outliers": False}
  ]
  for info in plotInfos:
    save_histogram(df.loc[:, info['column']].values, info, path)

# mesh = trimesh.load('testModels/db/0/m0/m0.off', force='mesh')
