import logging
import time
from datetime import timedelta

import pyrender
import trimesh

import analyze
import preprocess
import utils

trimesh.util.attach_to_log(level=logging.INFO)


def scale_outward(mesh):
  epsilon = 0.0002
  normals = mesh.face_normals
  for index, face in enumerate(mesh.faces):
    for vertice in face:
      mesh.vertices[vertice] = mesh.vertices[vertice] + epsilon * normals[index]
  return mesh


def render(meshes, showWireframe=True):
  scene = pyrender.Scene()
  for mesh in meshes:
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
  pyrender.Viewer(scene, use_raymond_lighting=True)


# Step 1
def step_1():
  mesh = trimesh.load('testModels/refined_db/0/m0/m0.off', force='mesh')
  render([mesh])


def main():
  start_time = time.monotonic()
  print("Analyze 1")
  analyze.filter_database(utils.originalDB, utils.excelPath)
  print("Preprocessing")
  preprocess.process_all()
  print("Analyze 2")
  analyze.filter_database(utils.refinedDB, utils.refinedexcelPath)
  print("Read Excel")
  originalDF = utils.read_excel(original=True)
  refinedDF = utils.read_excel(original=False)
  print("Save histograms")
  analyze.save_all_histograms(originalDF, utils.imagePath)
  analyze.save_all_histograms(refinedDF, utils.refinedImagePath)

  end_time = time.monotonic()
  print(timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
  main()
