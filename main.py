import logging
import time
from datetime import timedelta

import pyrender
import trimesh

import analyze
import preprocess
import utils
import numpy as np
trimesh.util.attach_to_log(level=logging.INFO)


def render(meshes, showWireframe=True):
  scene = pyrender.Scene()
  for mesh in meshes:
    colorvisuals = trimesh.visual.ColorVisuals(mesh, [200,200,200,255])
    mesh.visual = colorvisuals
    mesh1 = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh1)
  boxf_trimesh = trimesh.creation.box(extents=0.1*np.ones(3))
  boxf_face_colors = np.random.uniform(size=boxf_trimesh.faces.shape)
  boxf_trimesh.visual.face_colors = boxf_face_colors
  boxf_mesh = pyrender.Mesh.from_trimesh(boxf_trimesh, smooth=False)
  scene.add(boxf_mesh)
  if showWireframe:
    for mesh in meshes:
      # mesh = scale_mesh(mesh, 1.001)
      colorvisuals = trimesh.visual.ColorVisuals(mesh, [0,0,0,255])
      mesh.visual = colorvisuals
      wireframe = pyrender.Mesh.from_trimesh(mesh, wireframe=True, smooth=False)
      scene.add(wireframe)
  pyrender.Viewer(scene, use_raymond_lighting=True)


# Step 1
def step_1():
  mesh = trimesh.load('testModels/db/0/m0/m0.off', force='mesh')
  render([mesh])


start_time = time.monotonic()
# print("Analyze 1")
# analyze.filter_database(utils.originalDB, utils.excelPath)
# print("Preprocessing")
# preprocess.process_all()
# print("Analyze 2")
# analyze.filter_database(utils.refinedDB, utils.refinedexcelPath)
print("Read Excel")
originalDF = utils.read_excel(original=True)
refinedDF = utils.read_excel(original=False)
print("Save histograms")
analyze.save_all_histograms(originalDF, utils.imagePath)
analyze.save_all_histograms(refinedDF, utils.refinedImagePath)

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
