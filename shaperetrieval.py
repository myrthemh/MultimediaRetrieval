import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import trimesh
from annoy import AnnoyIndex
from numpy.linalg import norm
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from sklearn.manifold import TSNE

import analyze
import preprocess
import utils

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

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
  f = len(df['A3'][0]) * len(utils.hist_features_norm) + len(utils.scal_features_norm)
  save_map_neighbours(f, "euclidean", n_trees=10000)
  u = load_map_neighbours("testmodels.ann", f, 'euclidean')
  df["ANN"] = ""
  for index, row in df.iterrows():
    tuple_list = []
    idx, distance = u.get_nns_by_item(index, len(df), include_distances=True)
    # Remove distance to self
    idx = idx[1:]
    distance = distance[1:]
    for i, item in enumerate(distance):
      tuple_list.append((item, idx[i], df.iloc[idx[i]]['class']))

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


def save_similar_meshes(weight_vector):
  df = utils.read_excel(original=False)
  with open(utils.emd_norm_vector_path, 'rb') as f:
    emd_vector = np.load(f)
    column = df.apply(lambda x: find_similar_meshes(x, weight_vector, emd_vector, df), axis=1)

    df['similar_meshes'] = column
    utils.save_excel(df, original=False)


def find_similar_meshes(mesh_row, weight_vector, emd_vector, df):
  # Find similar meshes based on an existing row in the shape database
  single_vector = np.asarray(mesh_row[utils.scal_features_norm]) * weight_vector[:6]
  histogram_vector = np.asarray(mesh_row[utils.hist_features_norm]) * weight_vector[6:]
  distances = [distance for distance in get_distances(single_vector, histogram_vector, emd_vector, df, weight_vector) if
               distance[1] != mesh_row.name]
  return distances


def get_distances(single_vector, histogram_vector, emd_vector, df, weight_vector):
  distances = []
  # Compare with all meshes
  other_single_vectors = np.asarray(df[utils.scal_features_norm]) * weight_vector[:6]
  other_histogram_vectors = np.asarray(df[utils.hist_features_norm]) * weight_vector[6:]
  scalar_distances = list(map(lambda x: compute_euclidean_distance(single_vector, x), other_single_vectors))
  hist_distancess = list(map(lambda x:
                             list(map(lambda i: wasserstein_distance(histogram_vector[i], x[i]),
                                      range(len(histogram_vector))))
                             , other_histogram_vectors))

  # Standardize histogram distances:
  hist_distancess /= emd_vector
  distances = scalar_distances + np.sum(hist_distancess, axis=1)
  distances = list(np.dstack((distances, list(range(len(df))), df['class']))[0])
  distances.sort(key=sortmethod)
  return distances


def query(meshpath):
  mesh = trimesh.load(meshpath, force='mesh')
  row = analyze.fill_mesh_info(mesh, 20, '', features=False)
  # Proprocess the mesh:
  preprocessed_mesh = preprocess.preprocess_single_mesh(mesh, row)
  features = analyze.fill_mesh_info(preprocessed_mesh, '-1', '', features=True)

  # Get feature vectores
  single_vector = np.asarray([features[x] for x in utils.scal_features])
  histogram_vector = np.asarray([features[x] for x in utils.hist_features])

  # Normalize the vectores
  with open(utils.norm_vector_path, 'rb') as f:
    vector = np.load(f)
    single_vector_norm = (single_vector - vector[0]) / vector[1]
    histogram_vector_norm = [x / sum(x) for x in histogram_vector]
  with open(utils.emd_norm_vector_path, 'rb') as f:
    emd_vector = np.load(f)
    results = get_distances(single_vector_norm, histogram_vector_norm, emd_vector, utils.read_excel(original=False), utils.weight_vectors[0])
  return results, preprocessed_mesh


def tsne():
  df = utils.read_excel(original=False)
  X = np.vstack(np.concatenate(
    (np.asarray(row[utils.scal_features_norm]), np.concatenate(np.asarray(row[utils.hist_features_norm])).ravel())) for
                index, row in df.iterrows())

  perplexity = 30
  n_iter = 2000
  digits_proj = TSNE(perplexity=perplexity, n_iter=n_iter).fit_transform(X)
  images = df["path"].str.split("/").str[-1]
  eccentricity = df["eccentricity"]
  compactness = df["compactness"]
  diameter = df["diameter"]

  scatter(digits_proj, df['class'], perplexity, n_iter, images, eccentricity, compactness, diameter)
  plt.savefig(utils.refinedImagePath + 'tsne-generated.png', dpi=120)


def scatter(x, classes, perplexity, n_iter, images, eccentricity, compactness, diameter):
  # We choose a color palette with seaborn.

  # We create a scatter plot.
  f = plt.figure(figsize=(8, 8))
  ax = plt.subplot(aspect='equal')
  test1 = x[:, 0]
  test2 = x[:, 1]

  data = pd.DataFrame({"x": test1, "y": test2, "color": classes, "image": images,
                       "eccentricity": eccentricity, "compactness": compactness, "diameter": diameter})

  sc = px.scatter(data, x="x", y="y", color="color", color_discrete_sequence=px.colors.qualitative.Dark24,
                  title="Visual map",
                  labels={
                    "x": "Reduced Feature 1",
                    "y": "Reduced Feature 2",
                    "color": "Classes"
                  },
                  hover_data=["eccentricity", "compactness", "diameter"],
                  hover_name="image"

                  )

  sc.update_layout(
    legend=dict(
      x=0,
      y=1,
      traceorder="reversed",
      title_font_family="Times New Roman",
      font=dict(
        family="Courier",
        size=12,
        color="black"
      ),
      bgcolor="LightSteelBlue",
      bordercolor="Black",
      borderwidth=2
    )
  )

  sc.show()

  return f, ax, sc
