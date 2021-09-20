import numpy as np
import trimesh
import os

trimesh.util.attach_to_log()

classes = [
      "Insect", #0
      "Farm animal", #1
      "People", #2
      "Face", #3
      "Building", #4
      "Container", #5
      "LampOrWatch", #6
      "Stabweapon", #7
      "Chair", #8
      "Table", #9
      "Flowerpot", #10
      "Tool", #11
      "Airplane", #12 
      "Aircraft", #13
      "Spacecraft", #14
      "Car", #15
      "Chess piece", #16
      "DoorOrChest", #17
      "Satellite" #18
      ]

# Step 1
def step_1():
      mesh = trimesh.load('testModels/m0/m0.off', force='mesh')
      #mesh = trimesh.load('testModels/bunny.ply', force='mesh')
      mesh.show()

# Step 2

# The key constraints here are that (a) the reduced database should contain at least 200 shapes; (b) you should have
# shapes of most (ideally all) of the existing class types in the database; (c) try to balance the classes, i.e.,
# do not use tens of shapes of one class and only a handful of shapes of another class. That is: Start small. When
# this works, add more shapes to your (already) functioning code, and repeat the tests.
#
# Start building a simple filter that checks all shapes in the database. The filter should output, for each shape
#
# - the class of the shape
# - the number of faces and vertices of the shape
# - the type of faces (e.g. only triangles, only quads, mixes of triangles and quads)
# - the axis-aligned 3D bounding box of the shapes

def bounding_box(vertices):
      #Find the two corners of the bounding box surrounding the mesh.
      #Bottom will contain the lowest x, y and z values, while 'top' contains the highest values in the mesh.
      bottom = vertices[0].copy()
      top = vertices[0].copy()
      for vertex in vertices:
            for dimension in [0,1,2]:
                  bottom[dimension] = min(bottom[dimension], vertex[dimension])
                  top[dimension] = max(top[dimension], vertex[dimension])
      return (bottom, top)

def filter_database():
      db = 'testModels/db'
      # iterate over all models:
      for classFolder in os.listdir(db):
            for modelFolder in os.listdir(db + '/' + classFolder):
                  for filename in os.listdir(db + '/' + classFolder + '/' + modelFolder):
                        if filename.endswith('.off'):

                              #Find the relevant info for the mesh:
                              mesh = trimesh.load(db + '/' + classFolder + '/' + modelFolder + '/' + filename, force='mesh')
                              mesh_info = {}
                              mesh_info["class"] = int(classFolder)
                              mesh_info["nrfaces"] = len(mesh.faces)
                              mesh_info["nrvertices"] = len(mesh.vertices)
                              face_sizes = list(map(lambda x: len(x) ,mesh.faces))
                              mesh_info["containsTriangles"] = 3 in face_sizes
                              mesh_info["containsQuads"] = 4 in face_sizes
                              mesh_info["bounding_box_corners"] = bounding_box(mesh.vertices)
                              
                              #This should still be stored somewhere:
                              print(mesh_info)

filter_database()