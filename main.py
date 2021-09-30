import logging

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
    mesh1 = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh1)
  boxf_trimesh = trimesh.creation.box(extents=0.1*np.ones(3))
  boxf_face_colors = np.random.uniform(size=boxf_trimesh.faces.shape)
  boxf_trimesh.visual.face_colors = boxf_face_colors
  boxf_mesh = pyrender.Mesh.from_trimesh(boxf_trimesh, smooth=False)
  scene.add(boxf_mesh)
  if showWireframe:
    for mesh in meshes:
      # Add copy of mesh in wireframe mode
      material = pyrender.Material()
      # mesh = scale_mesh(mesh, 1.001)
      wireframe = pyrender.Mesh.from_trimesh(mesh, wireframe=True, smooth=False, material=material)
      scene.add(wireframe)
  pyrender.Viewer(scene, use_raymond_lighting=True)


# Step 1
def step_1():
  mesh = trimesh.load('testModels/db/0/m0/m0.off', force='mesh')
  render([mesh])


analyze.filter_database(utils.originalDB, utils.excelPath)
preprocess.process_all()
analyze.filter_database(utils.refinedDB, utils.refinedexcelPath)
originalDF = utils.read_excel(original=True)
refinedDF = utils.read_excel(original=False)
analyze.save_all_histograms(originalDF, utils.imagePath)
analyze.save_all_histograms(refinedDF, utils.refinedImagePath)
