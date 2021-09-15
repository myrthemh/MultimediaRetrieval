import numpy as np
import trimesh

trimesh.util.attach_to_log()
mesh = trimesh.load('models/m0/m0.off', force='mesh')

mesh.show()