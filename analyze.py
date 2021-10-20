import logging
import math
import os
import matplotlib.ticker as mtick

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh
from trimesh import convex
from matplotlib.ticker import PercentFormatter

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
  if mesh.area > 0:
    c = (mesh.area ** 3) / ((36 * np.pi * volume(mesh)) ** 2)
  else:
    c = 0
  return c


def diameter(mesh):
  try:
    conv_hull_points = trimesh.convex.hull_points(mesh)
    diameter = max([max((np.linalg.norm(x - y)) for y in conv_hull_points) for x in conv_hull_points])
  except:
    print("error calculating hull, reverting to brute force diameter calculation")
    diameter = max([max((np.linalg.norm(x - y)) for y in mesh.vertices) for x in mesh.vertices])
  # print(diameter == diameter_old)
  # print(f'difference new old {diameter - diameter_old}')
  return diameter


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


def filter_database(dbPath, excelPath, picklePath, features=True):
  db = dbPath
  df = pd.DataFrame()
  utils.ensure_dir(excelPath)
  utils.ensure_dir(picklePath)
  # iterate over all models:
  for classFolder in os.listdir(db):
    for modelFolder in os.listdir(db + '/' + classFolder):
      for filename in os.listdir(db + '/' + classFolder + '/' + modelFolder):
        if filename.endswith('.off'):
          # Find the relevant info for the mesh:
          path = db + '/' + classFolder + '/' + modelFolder + '/' + filename
          mesh = trimesh.load(path, force='mesh')
          mesh_info = fill_mesh_info(mesh, classFolder, path, features)
          df = df.append(mesh_info, ignore_index=True)
  df.to_excel(excelPath)
  df.to_pickle(picklePath)



def make_bins(list, lowerbound, upperbound, nrbins, plot):
  if plot:
    return list, {"blocksize": (upperbound/nrbins), "xlim": lowerbound, "ylabel": "Percentage"}
  bins = np.histogram(list, bins=nrbins, range=(lowerbound, upperbound), density=True)
  return bins[0]


def select_random_number_expection(exclude, selected_vertices):
  if len(exclude) == 1:
    new_list = [el for el in selected_vertices if not np.array_equal(el, exclude[0])]
    return new_list[np.random.randint(0, high=len(new_list))]


def check_duplicates(mesh, selected_vertices, number_vertices):
  for idx, vertice in enumerate(selected_vertices):

    if number_vertices < 2:
      if np.array_equal(vertice[0], vertice[1]):
        selected_vertices[idx, 0] = select_random_number_expection(vertice[0], mesh.vertices)
    if number_vertices < 3:
      continue

    if np.array_equal(vertice[0], vertice[2]):
      selected_vertices[idx, 0] = select_random_number_expection(vertice[0], mesh.vertices)
    if np.array_equal(vertice[1], vertice[2]):
      selected_vertices[idx, 1] = select_random_number_expection(vertice[1], mesh.vertices)

    if number_vertices < 4:
      continue

    if np.array_equal(vertice[0], vertice[3]):
      selected_vertices[idx, 0], select_random_number_expection(vertice[0], mesh.vertices)
    if np.array_equal(vertice[2], vertice[3]):
      selected_vertices[idx, 2], select_random_number_expection(vertice[2], mesh.vertices)
    if np.array_equal(vertice[1], vertice[3]):
      selected_vertices[idx, 1], select_random_number_expection(vertice[1], mesh.vertices)

  return selected_vertices


def A3(mesh, amount=1000, plot=False):
  random_vertices = mesh.vertices[np.random.randint(0, high=len(mesh.vertices), size=(amount, 3))]
  random_vertices = check_duplicates(mesh, random_vertices, 3)
  angles = [utils.angle(x[0] - x[1], x[0] - x[2]) for x in random_vertices]
  return make_bins(angles, 0, 0.5*math.pi, 10, plot)


def D1(mesh, amount=1000, plot=False):
  # Distance barycentre to random vertice
  random_vertices = mesh.vertices[np.random.randint(0, high=len(mesh.vertices), size=(amount))]
  distance_barycentre = [math.sqrt(sum(random_vertice ** 2)) for random_vertice in random_vertices]
  return make_bins(distance_barycentre, 0, 0.5, 10, plot)


def D2(mesh, amount=1000, plot=False):
  # Distance between two random vertices
  random_vertices = mesh.vertices[np.random.randint(0, high=len(mesh.vertices), size=(amount, 2))]
  random_vertices = check_duplicates(mesh, random_vertices, 2)
  distance_vertices = [math.sqrt(sum((random_vertice[0] - random_vertice[1]) ** 2)) for random_vertice in
                       random_vertices]
  return make_bins(distance_vertices, 0, 1, 10, plot)


def D3(mesh, amount=1000, plot=False):
  # Root of area of triangle given by three random vertices
  random_vertices = mesh.vertices[np.random.randint(0, high=len(mesh.vertices), size=(amount, 3))]
  random_vertices = check_duplicates(mesh, random_vertices, 3)
  area_vertices = [math.sqrt(
    (math.sqrt(sum(np.cross(random_vertice[0] - random_vertice[2], random_vertice[1] - random_vertice[2]) ** 2)) / 2))
                   for random_vertice in random_vertices]
  return make_bins(area_vertices, 0, 0.93, 10, plot)


def tetrahedon_volume(vertices):
  vector1 = vertices[0] - vertices[3]
  vector2 = vertices[1] - vertices[3]
  vector3 = vertices[2] - vertices[3]
  volume = abs(np.dot(vector1, (np.cross(vector2, vector3)))) / 6
  return volume


def D4(mesh, amount=1000, plot=False):
  # Cubic root of volume of tetahedron given by four random vertices
  random_vertices = mesh.vertices[np.random.randint(0, high=len(mesh.vertices), size=(amount, 4))]
  random_vertices = check_duplicates(mesh, random_vertices, 4)
  volumes = [tetrahedon_volume(vertices) ** (1.0 / 3) for vertices in random_vertices]
  return make_bins(volumes, 0, 0.55, 10, plot)


def fill_mesh_info(mesh, classFolder, path, features=True):
  face_sizes = list(map(lambda x: len(x), mesh.faces))
  print(f"analyzing model {path}")
  if features:
    mesh_info = {"class": int(classFolder), "nrfaces": len(mesh.faces), "nrvertices": len(mesh.vertices),
                 "containsTriangles": 3 in face_sizes, "containsQuads": 4 in face_sizes,
                 "bounding_box_corners": mesh.bounds, "path": f'{path}',
                 "axis-aligned_bounding_box_distance": np.linalg.norm(mesh.bounds[0] - mesh.bounds[1]),
                 "barycentre_distance": barycentre_distance(mesh),
                 "volume": bounding_box_volume(mesh),
                 "area": mesh.area,
                 "eccentricity": eccentricity(mesh),
                 "eigen_x_angle": preprocess.eigen_angle(mesh),
                 "diameter": diameter(mesh),
                 "compactness": compactness(mesh),
                 "A3": A3(mesh),
                 "D1": D1(mesh),
                 "D2": D2(mesh),
                 "D3": D3(mesh),
                 "D4": D4(mesh),
                 "area_faces": mesh.area_faces}
  else:
    mesh_info = {"class": int(classFolder), "nrfaces": len(mesh.faces), "nrvertices": len(mesh.vertices),
                 "containsTriangles": 3 in face_sizes, "containsQuads": 4 in face_sizes,
                 "bounding_box_corners": mesh.bounds, "path": f'{path}',
                 "axis-aligned_bounding_box_distance": np.linalg.norm(mesh.bounds[0] - mesh.bounds[1]),
                 "barycentre_distance": barycentre_distance(mesh),
                 "volume": bounding_box_volume(mesh),
                 "area": mesh.area,
                 "eigen_x_angle": preprocess.eigen_angle(mesh),
                 "area_faces": mesh.area_faces,
                 # "eccentricity": eccentricity(mesh),
                 # "compactness": compactness(mesh)
                 }
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


def histograms_all_classes(data, column):
  fig, axs = plt.subplots(6, 3, figsize = (20,15))
  for c in range(0, 18):
    for i in data.loc[data["class"] == c+1, column]:
      axs[c%6, int(c/6)].plot(i)
      axs[c%6, int(c/6)].xaxis.set_major_formatter(mtick.PercentFormatter(10))
      axs[c % 6, int(c / 6)].yaxis.set_major_formatter(mtick.PercentFormatter(50000))
    axs[c%6, int(c/6)].set_title(str(classes[c+1]))


  fig.tight_layout()
  fig.savefig(utils.refinedImagePath + "test" + '.png')


def save_histogram(data, info, path):
  # the histogram of the data

  # drop NA values if they exist
  data = data[np.isfinite(data)]
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


def save_all_histograms(df, path, features=False):
  md = meta_data(df)
  if not features:
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
       "ylabel": "#Meshes", "xlabel": "Diagonal distance of axis aligned bounding box", "skip_outliers": False},
      {"column": "eigen_x_angle", "title": "Angle largest eigenvector - x-axis", "blocksize": 50,
       "xlim": 3.2,
       "ylabel": "#Meshes", "xlabel": "Radian angle between largest eigenvector and x-axis", "skip_outliers": False}
    ]
  else:
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
       "xlim": 3, "ylabel": "#Meshes", "xlabel": "Diagonal distance of axis aligned bounding box",
       "skip_outliers": False},
      {"column": "eigen_x_angle", "title": "Angle largest eigenvector - x-axis", "blocksize": 50, "xlim": 3.2,
       "ylabel": "#Meshes", "xlabel": "Radian angle between largest eigenvector and x-axis", "skip_outliers": False},
      {"column": "compactness", "title": "Compactness", "blocksize": 50, "xlim": 0, "ylabel": "#Meshes",
       "xlabel": "Compactness", "skip_outliers": True},
      {"column": "eccentricity", "title": "Eccentricity", "blocksize": 50, "xlim": 0, "ylabel": "#Meshes",
       "xlabel": "Eccentricity", "skip_outliers": True}

    ]

  # Area_faces plot:
  all_areas = [values for values in df.loc[:, "area_faces"].values]
  all_areas = np.array([value for sublist in all_areas for value in sublist])
  plotinfo = {"title": "Face area distribution over all meshes", "blocksize": 50, "xlim": 0.0006, "ylabel": "#faces",
              "xlabel": "face area", "skip_outliers": True}
  save_histogram(all_areas, plotinfo, path)
  for info in plotInfos:
    save_histogram(df.loc[:, info['column']].values, info, path)

# mesh = trimesh.load('testModels/refined_db/9/m905/m905.off', force='mesh')
# D3(mesh, amount=1000)

def plot_shape_properties(feature, shape, classes=1):
  path = utils.refinedImagePath
  mesh = trimesh.load(shape)
  #render([mesh])
  if feature =="A3":
    bins, info = A3(mesh, plot=True)
    title = "Angle between three random points"
  if feature == "D1":
    bins, info = D1(mesh, plot=True)
    title = " Distance between the barycenter and a random point"
  if feature == "D2":
    bins, info = D2(mesh, plot=True)
    title = " Distance between two random point"
  if feature == "D3":
    bins, info = D3(mesh, plot=True)
    title = "Square  root  of the area of triangle made by three random point"
  if feature == "D4":
    bins, info = D4(mesh, plot=True)
    title = "Cube  root  of  the  volume  of  tetrahedron  made  by  three random points"
  plt.hist(bins, facecolor='g', alpha=0.75, weights=np.ones(len(bins)) / len(bins))
  plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
  plt.xlabel(feature)
  plt.ylabel(info["ylabel"])
  plt.title(title + " of class" + classes[classes])
  plt.savefig(path + feature + shape[-8:-4] +'.png')


def visualize_difference_features():
  plot_shape_properties(feature="A3", shape='testModels/refined_db/1/m102/m102.off', classes=1)
  plot_shape_properties(feature="A3", shape='testModels/refined_db/1/m105/m105.off', classes=1)
  plot_shape_properties(feature="A3", shape='testModels/refined_db/17/m1703/m1703.off', classes=17)

  plot_shape_properties(feature="D1", shape='testModels/refined_db/1/m112/m112.off')
  plot_shape_properties(feature="D1", shape='testModels/refined_db/1/m112/m112.off')
  plot_shape_properties(feature="D1", shape='testModels/refined_db/1/m112/m112.off')

  plot_shape_properties(feature="D2", shape='testModels/refined_db/1/m112/m112.off')
  plot_shape_properties(feature="D2", shape='testModels/refined_db/1/m112/m112.off')
  plot_shape_properties(feature="D2", shape='testModels/refined_db/1/m112/m112.off')

  plot_shape_properties(feature="D3", shape='testModels/refined_db/1/m112/m112.off')
  plot_shape_properties(feature="D3", shape='testModels/refined_db/1/m112/m112.off')
  plot_shape_properties(feature="D3", shape='testModels/refined_db/1/m112/m112.off')

  plot_shape_properties(feature="D4", shape='testModels/refined_db/1/m112/m112.off')
  plot_shape_properties(feature="D4", shape='testModels/refined_db/1/m112/m112.off')
  plot_shape_properties(feature="D4", shape='testModels/refined_db/1/m112/m112.off')
