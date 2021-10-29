import numpy as np
import pandas as pd
import trimesh
from annoy import AnnoyIndex
from sklearn.manifold import TSNE
from numpy.linalg import norm
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
import plotly.express as px


import analyze
import preprocess
import utils


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt


def save_map_neighbours(n_features, metric, n_trees=1000):
  # Run once, this saves a mapping file
  # metric can be one of "angular", "euclidean", "manhattan", "hamming", or "dot"
  f = n_features
  t = AnnoyIndex(f, metric)
  df = utils.read_excel(original=False)
  for index, row in df.iterrows():
    v_scal = np.asarray(row[utils.scal_features_norm])
    v_hist = np.concatenate(np.asarray(row[utils.hist_features_norm])).ravel()
    v = np.concatenate((v_scal, v_hist))
    t.add_item(index, v)
  t.build(n_trees, n_jobs=-1)
  t.save('testmodels.ann')


def load_map_neighbours(map_path, n_features, metric):
  f = n_features
  u = AnnoyIndex(f, metric)
  u.load(map_path)

  return u


def ann_distances_to_excel():
  df = utils.read_excel(False)
  save_map_neighbours(55, "euclidean", n_trees=10000)
  u = load_map_neighbours("testmodels.ann", 55, 'euclidean')
  df["ANN"] = ""
  for index, row in df.iterrows():
    tuple_list = []
    idx, distance = u.get_nns_by_item(index, 6, include_distances=True)
    # Remove distance to self
    idx = idx[1:6]
    distance = distance[1:6]
    for i, item in enumerate(distance):
      tuple_list.append((item, idx[i]))

    df.at[index, "ANN"] = tuple_list
  utils.save_excel(df, False)


def neighbours_to_paths(neighbours, distances_included=True):
  df = utils.read_excel(False).iloc[neighbours[0]] if distances_included else utils.read_excel(False).iloc[neighbours]
  return df['path']


def compute_euclidean_distance(vector1, vector2):
  return norm(vector1 - vector2)


def cosine_difference(vector1, vector2):
  return distance.cosine(vector1, vector2)


def sortmethod(x):
  return x[0]


def paths_to_meshes(paths):
  meshes = []
  for path in paths:
    meshes.append(trimesh.load(path, force='mesh'))
  return meshes

def save_similar_meshes():
  df = utils.read_excel(original=False)
  column = []
  for index, row in df.iterrows():
    if index % 10 == 0:
      print("Calculating similarities", index, '/', len(df))
    distances = find_similar_meshes(row)[:5]
    column.append(distances)
  df['similar_meshes'] = column
  utils.save_excel(df, original=False)


def find_similar_meshes(mesh_row):
  # # Analyze the mesh
  # mesh = trimesh.load(row['path'], force='mesh')
  # mesh_info = analyze.fill_mesh_info(mesh, -1, "path", features=True)
  df = utils.read_excel(original=False)

  # # Get feature vector:
  # single_vector = np.asarray([mesh_info[column] for column in utils.scal_features])
  # histograms = np.asarray([mesh_info[column] for column in utils.hist_features])
  # histogram_vector = [preprocess.sum_divide(x) for x in histograms]

  # # Standardize the scalar features:
  # with open(utils.norm_vector_path, 'rb') as f:
  #   vectors = np.load(f)
  #   single_vector -= vectors[0]
  #   single_vector /= vectors[1]
  single_vector = np.asarray(mesh_row[utils.scal_features_norm])
  histogram_vector = np.asarray(mesh_row[utils.hist_features_norm])
  distances = []
  # Compare with all meshes
  with open(utils.emd_norm_vector_path, 'rb') as f:
    emd_vector = np.load(f)
    for index, row in df.iterrows():
      if mesh_row['path'][-11:] == row['path'][-11:]:
        continue
      other_single_vector = np.asarray(row[utils.scal_features_norm])
      other_histogram_vector = np.asarray(row[utils.hist_features_norm])
      scalar_distance = compute_euclidean_distance(single_vector, other_single_vector)
      hist_distances = [wasserstein_distance(histogram_vector[i], other_histogram_vector[i]) for i in
                        range(len(histogram_vector))]

      # Standardize histogram distances:
      hist_distances /= emd_vector
      distance = scalar_distance + sum(hist_distances)
      distances.append((distance, index))
  distances.sort(key=sortmethod)
  return distances

def tsne():

  df = utils.read_excel(original=False)
  X = np.vstack(np.concatenate(   ( np.asarray(row[utils.scal_features_norm]), np.concatenate(np.asarray(row[utils.hist_features_norm])).ravel() )) for index, row in df.iterrows())

  digits_proj = TSNE().fit_transform(X)

  y = df["class"]
  scatter(digits_proj, y)
  plt.savefig(utils.refinedImagePath + 'tsne-generated.png', dpi=120)


def scatter(x, colors):
  # We choose a color palette with seaborn.
  palette = np.array(sns.color_palette("hls", 19))

  # We create a scatter plot.
  f = plt.figure(figsize=(8, 8))
  ax = plt.subplot(aspect='equal')
  sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])


  plt.xlim(-25, 25)
  plt.ylim(-25, 25)
  ax.axis('off')
  ax.axis('tight')



  # We add the labels for each digit.
  txts = []
  # for i in range(10):
  #   # Position of each label.
  #   xtext, ytext = np.median(x[colors == i, :], axis=0)
  #   txt = ax.text(xtext, ytext, str(i), fontsize=24)
  #   txt.set_path_effects([
  #     PathEffects.Stroke(linewidth=5, foreground="w"),
  #     PathEffects.Normal()])
  #   txts.append(txt)

  return f, ax, sc, txts
