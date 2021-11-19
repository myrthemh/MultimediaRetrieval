import logging
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import trimesh
from matplotlib.ticker import PercentFormatter
from scipy.spatial.distance import cdist
from trimesh import convex

import utils

trimesh.util.attach_to_log()
logging.getLogger('matplotlib.font_manager').disabled = True


def volume(mesh):
  v1s = mesh.vertices[mesh.faces][::, 0] - mesh.centroid
  v2s = mesh.vertices[mesh.faces][::, 1] - mesh.centroid
  v3s = mesh.vertices[mesh.faces][::, 2] - mesh.centroid
  v = 1 / 6 * np.abs(np.sum(np.cross(v1s, v2s) * v3s))
  return v


def compactness(mesh):
  if mesh.area > 0:
    c = np.divide(np.power(mesh.area, 3), (36 * np.pi * np.power(volume(mesh), 2)))
  else:
    c = 0
  return c if not np.isnan(c) else 0


def diameter(mesh):
  try:
    conv_hull_points = trimesh.convex.hull_points(mesh)
    d = np.max(cdist(conv_hull_points, conv_hull_points, metric='euclidean'))
  except:
    print("error calculating hull, reverting to brute force diameter calculation")
    d = np.max(cdist(mesh.vertices, mesh.vertices, metric='euclidean'))

  return d


def eccentricity(mesh):
  values, _ = utils.eigen_values_vectors(mesh)
  ecc = values[np.argmin(values)] / values[np.argmax(values)]
  return ecc


def barycentre_distance(mesh):
  return np.sqrt(np.sum(mesh.centroid * mesh.centroid))


def bounding_box_volume(mesh):
  x = mesh.bounds
  volume = (x[1][0] - x[0][0]) * (x[1][1] - x[0][1]) * (x[1][2] - x[0][2])
  return volume

def merge_bins():
  df = utils.read_excel(original=False)
  for column in df[utils.hist_features]:
    values = np.asarray(df[column])
    newvalues = list(map(lambda x: x[np.arange(0, 20, step=2)] + x[np.arange(1, 21, step=2)], values))
    df[column] = newvalues
  utils.save_excel(df, original=False)
    
def filter_database(dbPath, excelPath, picklePath, features=True):
  db = dbPath
  df = pd.DataFrame()
  utils.ensure_dir(excelPath)
  utils.ensure_dir(picklePath)
  index_to_class, class_sizes = utils.class_dictionaries()
  # iterate over all models:
  for classFolder in os.listdir(db):
    for modelFolder in os.listdir(db + '/' + classFolder):
      for filename in os.listdir(db + '/' + classFolder + '/' + modelFolder):
        if filename.endswith('.off'):
          # Find the relevant info for the mesh:
          file_number = int(filename[1:-4])
          shape_class = index_to_class[file_number]
          if shape_class in class_sizes:
            path = db + '/' + classFolder + '/' + modelFolder + '/' + filename
            mesh = trimesh.load(path, force='mesh')
            mesh_info = fill_mesh_info(mesh, shape_class, path, features)
            df = df.append(mesh_info, ignore_index=True)

  df.to_excel(excelPath)
  df.to_pickle(picklePath)


def make_bins(list, lowerbound, upperbound, nrbins, plot):
  if plot:
    return list, {"blocksize": (upperbound / nrbins), "xlim": lowerbound, "ylabel": "Percentage"}
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


def A3(mesh, amount=utils.hist_amount, plot=False):
  random_vertices =  mesh.vertices[np.random.randint(0, high=len(mesh.vertices), size=(amount, 3))]
  random_vertices = check_duplicates(mesh, random_vertices, 3)
  angles = np.arccos(
    np.clip(np.sum(utils.unit_vector(np.subtract(random_vertices[::, 0], random_vertices[::, 1]), transpose=True) *
                   utils.unit_vector(np.subtract(random_vertices[::, 0], random_vertices[::, 2]), transpose=True),
                   axis=1), -1.0, 1.0))
  return make_bins(angles, 0, math.pi, utils.nr_bins_hist, plot)


def D1(mesh, amount=utils.target_vertices, plot=False):
  # Distance barycentre to random vertice
  random_vertices = mesh.vertices[np.random.randint(0, high=len(mesh.vertices), size=(amount))]
  distance_barycentre = np.sqrt(np.sum(np.power(random_vertices, 2), axis=1))
  return make_bins(distance_barycentre, 0, 0.75, utils.nr_bins_hist, plot)


def D2(mesh, amount=utils.hist_amount, plot=False):
  # Distance between two random vertices
  random_vertices = mesh.vertices[np.random.randint(0, high=len(mesh.vertices), size=(amount, 2))]
  random_vertices = check_duplicates(mesh, random_vertices, 2)
  distance_vertices = np.sqrt(np.sum(np.power(random_vertices[::, 0] - random_vertices[::, 1], 2), axis=1))
  return make_bins(distance_vertices, 0, 1, utils.nr_bins_hist, plot)


def D3(mesh, amount=utils.hist_amount, plot=False):
  # Root of area of triangle given by three random vertices
  random_vertices = mesh.vertices[np.random.randint(0, high=len(mesh.vertices), size=(amount, 3))]
  random_vertices = check_duplicates(mesh, random_vertices, 3)
  area_vertices = np.sqrt(np.sqrt(np.sum(
    np.power(np.cross(random_vertices[::, 0] - random_vertices[::, 2], random_vertices[::, 1] - random_vertices[::, 2]),
             2), axis=1)) / 2)
  return make_bins(area_vertices, 0, 2 / 3, utils.nr_bins_hist, plot)


def D4(mesh, amount=utils.hist_amount, plot=False):
  # Cubic root of volume of tetahedron given by four random vertices
  random_vertices = mesh.vertices[np.random.randint(0, high=len(mesh.vertices), size=(amount, 4))]
  random_vertices = check_duplicates(mesh, random_vertices, 4)
  vectors1 = random_vertices[::, 0] - random_vertices[::, 3]
  vectors2 = random_vertices[::, 1] - random_vertices[::, 3]
  vectors3 = random_vertices[::, 2] - random_vertices[::, 3]
  volumes = np.power(np.divide(np.absolute(np.sum(vectors1 * (np.cross(vectors2, vectors3)), axis=1)), 6), (1.0 / 3))
  return make_bins(volumes, 0, 0.6 * 0.55, utils.nr_bins_hist, plot)


def tetrahedon_volume(vertices):
  vector1 = vertices[0] - vertices[3]
  vector2 = vertices[1] - vertices[3]
  vector3 = vertices[2] - vertices[3]
  volume = abs(np.dot(vector1, (np.cross(vector2, vector3)))) / 6
  return volume

def AABB_volume(mesh):
  x = abs(mesh.bounds[0][0] - mesh.bounds[1][0])
  y = abs(mesh.bounds[0][1] - mesh.bounds[1][1])
  z = abs(mesh.bounds[0][2] - mesh.bounds[1][2])
  return x * y * z

def fill_mesh_info(mesh, shape_class, path, features=True):
  face_sizes = list(map(lambda x: len(x), mesh.faces))
  print(f"analyzing model {path}")
  if features:
    mesh_info = {"class": shape_class, "nrfaces": len(mesh.faces), "nrvertices": len(mesh.vertices),
                 "containsTriangles": 3 in face_sizes, "containsQuads": 4 in face_sizes,
                 "bounding_box_corners": mesh.bounds, "path": f'{path}',
                 "axis-aligned_bounding_box_volume":AABB_volume(mesh),
                 "barycentre_distance": barycentre_distance(mesh),
                 "volume": volume(mesh),
                 "area": mesh.area,
                 "eccentricity": eccentricity(mesh),
                 "eigen_x_angle": utils.eigen_angle(mesh),
                 "diameter": diameter(mesh),
                 "compactness": compactness(mesh),
                 "A3": A3(mesh),
                 "D1": D1(mesh),
                 "D2": D2(mesh),
                 "D3": D3(mesh),
                 "D4": D4(mesh),
                 "area_faces": mesh.area_faces}

  else:
    mesh_info = {"class": shape_class, "nrfaces": len(mesh.faces), "nrvertices": len(mesh.vertices),
                 "containsTriangles": 3 in face_sizes, "containsQuads": 4 in face_sizes,
                 "bounding_box_corners": mesh.bounds, "path": f'{path}',
                 "axis-aligned_bounding_box_volume": np.linalg.norm(mesh.bounds[0] - mesh.bounds[1]),
                 "barycentre_distance": barycentre_distance(mesh),
                 "volume": volume(mesh),
                 "area": mesh.area,
                 "eigen_x_angle": utils.eigen_angle(mesh),
                 "area_faces": mesh.area_faces
                 }
  mesh_info = detect_outliers(mesh, mesh_info)
  return mesh_info


def detect_outliers(mesh, mesh_info):
  if len(mesh.vertices) < utils.target_vertices * 0.9:
    mesh_info["subsampled_outlier"] = True
    mesh_info["supersampled_outlier"] = False
  elif len(mesh.vertices) > utils.target_vertices * 1.1:
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
  fig, axs = plt.subplots(5, 3, figsize=(20, 15))
  index_to_class, class_sizes = utils.class_dictionaries()
  for index, c in enumerate(class_sizes.keys()):
    for i in data.loc[data["class"] == c, column]:
      axs[index % 5, int(index / 5)].plot(i)
      # axs[c % 6, int(c / 6)].xaxis.set_major_formatter(mtick.PercentFormatter(10))
    # axs[c % 6, int(c / 6)].yaxis.set_major_formatter(mtick.PercentFormatter(20000))
    axs[index % 5, int(index / 5)].set_title(c)

  fig.tight_layout()
  fig.savefig(utils.refinedImagePath + "all_classes" + column + '.png')


def save_histogram(data, info, path):
  # the histogram of the data

  # reset params
  plt.rcParams.update(plt.rcParamsDefault)
  plt.figure()
  # drop NA values if they exist
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
  if 'column' in info and info['column'] == 'class':
    bins = np.concatenate([bins, [15.0]]) -0.5
    plt.xlim(-0.5, info["xlim"] - 0.5)
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.34)
  plt.hist(data, bins=bins, facecolor='g', alpha=0.75)
  if info["ylim"] > 0:
    plt.ylim(0, info["ylim"])
  plt.xlabel(info["xlabel"])
  plt.ylabel(info["ylabel"])
  plt.title(info["title"])
  if info["xlim"] != 0 and not ('column' in info and info['column'] == 'class'):
    plt.xlim(0, info["xlim"])
  # plt.grid(True)
  plt.gcf().subplots_adjust(left=0.15)
  utils.ensure_dir(path)
  plt.savefig(path + info["title"] + '.png')
  plt.clf()
  plt.cla()


def save_all_histograms(df, path, features=False):
  plotInfos = [
    # {"column": "class", "title": "Class distribution", "blocksize": 15, "xlim": 15, "ylim": 0, "ylabel": "#Meshes",
    #   "xlabel": "Class name", "skip_outliers": False},
    {"column": "nrfaces", "title": "Face distribution", "blocksize": 25, "xlim": 0, "ylim": 750, "ylabel": "#Meshes",
      "xlabel": "Number of faces", "skip_outliers": True},
    {"column": "nrvertices", "title": "Vertice distribution", "blocksize": 25, "xlim": 0, "ylim": 800, "ylabel": "#Meshes",
      "xlabel": "Number of vertices", "skip_outliers": True},
    {"column": "volume", "title": "Mesh volume", "blocksize": 15, "xlim": 0, "ylim": 1400, "ylabel": "#Meshes",
      "xlabel": "Mesh volume", "skip_outliers": False},
    {"column": "barycentre_distance", "title": "Barycentre origin distance", "blocksize": 20, "xlim": 1, "ylim": 1400,
      "ylabel": "#Meshes", "xlabel": "Distance barycentre to origin", "skip_outliers": False},
    {"column": "axis-aligned_bounding_box_volume", "title": "Axis-aligned bounding box volume", "blocksize": 15,
      "xlim": 3, "ylim": 650, "ylabel": "#Meshes", "xlabel": "Volume of axis aligned bounding box",
      "skip_outliers": False},
    {"column": "eigen_x_angle", "ylim": 1400, "title": "Angle largest eigenvector - x-axis", "blocksize": 15,
      "xlim": 3.2, "ylabel": "#Meshes", "xlabel": "Radian angle between largest eigenvector and x-axis",
      "skip_outliers": False},
    {"column": "area", "title": "Mesh surface area", "blocksize": 15,
      "xlim": 0, "ylim": 1400, "ylabel": "#Meshes", "xlabel": "Total surface area of the mesh", "skip_outliers": True}
  ]
  if features:
    plotInfos += [
      {"column": "compactness", "title": "Compactness", "blocksize": 15, "xlim": 0, "ylim": 0, "ylabel": "#Meshes",
       "xlabel": "Compactness", "skip_outliers": True},
      {"column": "eccentricity", "title": "Eccentricity", "blocksize": 15, "xlim": 0, "ylim": 0, "ylabel": "#Meshes",
       "xlabel": "Eccentricity", "skip_outliers": True},
      {"column": "diameter", "title": "Diameter", "blocksize": 15, "xlim": 0, "ylim": 0, "ylabel": "#Meshes",
       "xlabel": "Diameter", "skip_outliers": False}
    ]

  # Area_faces plot:
  all_areas = [values for values in df.loc[:, "area_faces"].values]
  all_areas = np.array([value for sublist in all_areas for value in sublist])
  plotinfo = {"title": "Face area distribution over all meshes", "blocksize": 25, "xlim": 0.0006, "ylim": 0, "ylabel": "#faces",
              "xlabel": "face area", "skip_outliers": True}
  save_histogram(all_areas, plotinfo, path)
  for info in plotInfos:
    save_histogram(df.loc[:, info['column']].values, info, path)


def plot_shape_properties(feature, shape, classes=1):
  path = utils.refinedImagePath
  mesh = trimesh.load(shape)
  # render([mesh])
  if feature == "A3":
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
  # plt.title(title + " of class" + classes[classes])
  plt.savefig(path + feature + shape[-8:-4] + '.png')
  # plt.show()


def visualize_difference_features():
  plot_shape_properties(feature="A3", shape='testModels/refined_db/1/m102/m102.off', classes=1)
  plot_shape_properties(feature="A3", shape='testModels/refined_db/1/m105/m105.off', classes=1)
  plot_shape_properties(feature="A3", shape='testModels/refined_db/17/m1703/m1703.off', classes=17)

  plot_shape_properties(feature="D1", shape='testModels/refined_db/18/m1812/m1812.off')
  plot_shape_properties(feature="D1", shape='testModels/refined_db/18/m1814/m1814.off')
  plot_shape_properties(feature="D1", shape='testModels/refined_db/9/m909/m909.off')

  plot_shape_properties(feature="D2", shape='testModels/refined_db/16/m1601/m1601.off')
  plot_shape_properties(feature="D2", shape='testModels/refined_db/16/m1600/m1600.off')
  plot_shape_properties(feature="D2", shape='testModels/refined_db/17/m1712/m1712.off')

  plot_shape_properties(feature="D3", shape='testModels/refined_db/14/m1402/m1402.off')
  plot_shape_properties(feature="D3", shape='testModels/refined_db/14/m1403/m1403.off')
  plot_shape_properties(feature="D3", shape='testModels/refined_db/13/m1306/m1306.off')

  plot_shape_properties(feature="D4", shape='testModels/refined_db/5/m500/m500.off')
  plot_shape_properties(feature="D4", shape='testModels/refined_db/5/m507/m507.off')
  plot_shape_properties(feature="D4", shape='testModels/refined_db/7/m704/m704.off')



# mesh = trimesh.load('testModels/refined_db/1/m104/m104.off', force='mesh')
# diameter(mesh)
