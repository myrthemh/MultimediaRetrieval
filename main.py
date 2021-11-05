import logging
import time
import gc
from datetime import timedelta

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyrender
import trimesh

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


def save_figures(column):
  df = utils.read_excel(original=False)
  if column == "similar_meshes":
    save_path = utils.sim_images_path
  else:
    save_path = utils.ann_images_path
  for index, row in df.iterrows():
    if index % 10 == 0:
      print('Saving images', index, '/', len(df))
    tuples = row[column][:utils.query_size]
    mesh = trimesh.load(row['path'], force='mesh')
    save_mesh_image([mesh], save_path + row['class'] + '/' + str(index) + '/0')
    for i, tuple in enumerate(tuples):
      mesh_row = df.iloc[tuple[1]]
      mesh = trimesh.load(mesh_row['path'], force='mesh')
      save_mesh_image([mesh], save_path + row['class'] + '/' + str(index) + '/' + str(i + 1),
                      distance=round(tuple[0], 4))


def write_html():
  html_str = """
    <html>
    <div style='display: flex'>
    """
  for ann in [False, True]:
    html_str += "<div style='flex-grow: 1'>"
    if ann:
      html_str += "<h1> ANN: </h1>"
    else:
      html_str += "<h1> Our metric: </h1>"
    for c in range(15):
      html_str += """<details>
        <summary style='font-size: 30px'>""" + str(utils.classes[c]) + """</summary>"""
      for index, path in enumerate(utils.image_paths(c, ann=ann)):
        if index % 6 == 0 and index != 0:
          html_str += "<br> <hr>"
        html_str += "<img src = '" + path + "'>"
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
  # step_1()
  # compare_all()
  start_time = time.monotonic()
  # print("Analyze 1")
  # analyze.filter_database(utils.originalDB, utils.excelPath, utils.picklePath, features=False)
  # print("Preprocessing")
  # preprocess.process_all()
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
  analyze.visualize_difference_features()
  shaperetrieval.save_similar_meshes()
  shaperetrieval.ann_distances_to_excel()
  # save_figures('similar_meshes')
  # save_figures('ANN')
  write_html()
  evaluate.roc_plots()
  evaluate.boxplot_queries()
  shaperetrieval.tsne()
  end_time = time.monotonic()
  print(timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
  main()
