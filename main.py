import logging
import time
import gc
from datetime import timedelta

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyrender
import trimesh
from trimesh import util

import analyze
import preprocess
import shaperetrieval
import utils
import evaluate
from PIL import Image

trimesh.util.attach_to_log(level=logging.INFO)


def scale_outward(mesh):
  epsilon = 0.00002
  normals = mesh.face_normals
  for index, face in enumerate(mesh.faces):
    for vertice in face:
      mesh.vertices[vertice] = mesh.vertices[vertice] + epsilon * normals[index]
  return mesh


def render(meshes, showWireframe=True, setcolor=True):
  scene = pyrender.Scene()
  for mesh in meshes:
    if setcolor:
      colorvisuals = trimesh.visual.ColorVisuals(mesh, [200, 200, 200, 255])
      mesh.visual = colorvisuals
    mesh1 = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh1)
  if showWireframe:
    for mesh in meshes:
      mesh = scale_outward(mesh)
      colorvisuals = trimesh.visual.ColorVisuals(mesh, [0, 0, 0, 255])
      mesh.visual = colorvisuals
      wireframe = pyrender.Mesh.from_trimesh(mesh, wireframe=True, smooth=False)
      scene.add(wireframe)
  pyrender.Viewer(scene, use_raymond_lighting=True, viewport_size=(512, 512))


def save_mesh_image(meshes, path, distance=None, showWireframe=False, setcolor=True, ):
  scene = pyrender.Scene()
  for mesh in meshes:
    if setcolor:
      colorvisuals = trimesh.visual.ColorVisuals(mesh, [200, 200, 200, 255])
      mesh.visual = colorvisuals
    mesh1 = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh1)
  if showWireframe:
    for mesh in meshes:
      mesh = scale_outward(mesh)
      colorvisuals = trimesh.visual.ColorVisuals(mesh, [0, 0, 0, 255])
      mesh.visual = colorvisuals
      wireframe = pyrender.Mesh.from_trimesh(mesh, wireframe=True, smooth=False)
      scene.add(wireframe)
  camera = pyrender.PerspectiveCamera( yfov=np.pi / 3.0, znear=0.0000001,)
  camera_pose =[[ 1.73205,  0.,       0.,       0.,     ],
                [ 0.,       1.73205,  0.,       0.,     ],
                [ 0.,       0.,      -1.,      -1.0,    ],
                [ 0.,       0.,      0.,       1,     ]]
  scene.add(camera, pose=camera_pose)

  light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=800)
  scene.add(light, pose=np.eye(4) * [-1, -1, -1, 1])
  # pyrender.Viewer(scene, use_raymond_lighting=True, viewport_size=(512,512))
  flags = pyrender.RenderFlags.RGBA
  r = pyrender.OffscreenRenderer(utils.sim_image_size, utils.sim_image_size)
  color, _ = r.render(scene, flags=flags)
  matplotlib.use('Agg')
  plt.figure(figsize=(3,3)), plt.imshow(color)
  plt.axis('off')
  if distance is not None:
    plt.text(utils.sim_image_size / 2 - 50, utils.sim_image_size - 5, "D: " + str(distance), fontsize="xx-large")
  utils.ensure_dir(path)
  plt.savefig(path, bbox_inches='tight')
  plt.clf()
  plt.cla()
  plt.close('all')


# Step 1
def step_1():
  for path in utils.shape_paths(utils.refinedDB):
    mesh = trimesh.load(path, force='mesh')
    render([mesh])


def load_from_file(path):
  return trimesh.load(path, force='mesh')


def compare(meshes, setcolor=True):
  for i, m in enumerate(meshes):
    m.apply_translation([0, 0, i * 1])
  render(meshes, setcolor=setcolor)


def compare_all():
  for path in utils.shape_paths(utils.originalDB):
    ogmesh = trimesh.load(path, force='mesh')
    rfmesh = trimesh.load(utils.refined_path(path), force='mesh')
    meshes = [ogmesh, rfmesh]
    compare(meshes)


def save_figures():
  df = utils.read_excel(original=False)
  for index, row in df.iterrows():
    if index % 10 == 0:
      print('Saving images', index, '/', len(df))
    mesh = trimesh.load(row['path'], force='mesh')
    save_path = f"{row['path'][:-4]}.png"
    save_mesh_image([mesh], save_path)

def write_html():
  df = utils.read_excel(original=False)
  html_str = """
    <html>
    <div style='display: flex; font-family: Tahoma, sans-serif;'>
    """
  for ann in [False, True]:
    html_str += "<div style='flex-grow: 1'>"
    if ann:
      html_str += "<h1> ANN: </h1>"
    else:
      html_str += "<h1> Our metric: </h1>"
    for c in range(15):
      rows = df.loc[df['class'] == utils.classes[c]]
      html_str += """<details>
        <summary style='font-size: 30px'>""" + str(utils.classes[c]) + """</summary>"""
      for index, row in rows.iterrows():
        html_str += "<div style='display: flex;'>"
        html_str += "<img src = '" + f"{row['path'][:-4]}.png" + "'>"
        if ann:
          distances = row['ANN'][:5]
        else:
          distances = row['similar_meshes'][:5]
        for distance in distances:
          html_str += "<div style='position: relative'>"
          path = f"{df.iloc[distance[1]]['path'][:-4]}.png"
          html_str += f"<img src = '{path}'>"
          html_str += f"<p style='position: absolute; bottom: 0%; left:32%; font-size: 32px; font-weight: bold'>{round(distance[0], 3)}</p>"
          html_str += "</div>"
        html_str += "</div>"
        html_str += "<br> <hr>"
      html_str += "</details>"
    html_str += "</div>"
  html_str += """
  </div>
    </html>
    """
  html_str += """
  
  """
  Html_file = open("images.html", "w")
  Html_file.write(html_str)
  Html_file.close()


def main():
  step_1()
  compare_all()
  start_time = time.monotonic()
  print("Analyze 1")
  analyze.filter_database(utils.originalDB, utils.excelPath, utils.picklePath, features=False)
  print("Preprocessing")
  preprocess.process_all()
  save_figures()
  print("Analyze 2")
  analyze.filter_database(utils.refinedDB, utils.refinedexcelPath, utils.refinedpicklePath)
  print("normalize")
  preprocess.normalize_histogram_features(utils.hist_features)
  preprocess.scalar_normalization(utils.scal_features)
  preprocess.hist_distance_normalization()
  print("Read Excel")
  originalDF = utils.read_excel(original=True)
  refinedDF = utils.read_excel(original=False)
  print("Save histograms")
  analyze.save_all_histograms(originalDF, utils.imagePath)
  analyze.save_all_histograms(refinedDF, utils.refinedImagePath, features=True)
  for column in utils.hist_features_norm:
    analyze.histograms_all_classes(refinedDF, column)
  for index, vector in enumerate(utils.weight_vectors):
    if index < 2:
      continue
    start_time = time.monotonic()
    print(index)
    shaperetrieval.save_similar_meshes(vector)
    evaluate.roc_plots(index)
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))
  shaperetrieval.ann_distances_to_excel()
  write_html()
  evaluate.boxplot_queries()
  shaperetrieval.tsne()
  evaluate.plot_ktier(refinedDF)
  end_time = time.monotonic()
  print(timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
  main()
