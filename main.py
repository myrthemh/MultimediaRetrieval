import numpy as np
import trimesh

trimesh.util.attach_to_log()

# Step 1
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
