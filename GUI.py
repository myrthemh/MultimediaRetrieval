from tkinter import *
# import filedialog module
from tkinter import filedialog
from functools import partial
import trimesh
from trimesh import util

import main
import shaperetrieval
import utils
import os

class LabelGrid(Frame):
  """
  Creates a grid of labels that have their cells populated by content.
  """

  def __init__(self, master, content=([0, 0], [0, 0]), *args, **kwargs):
    Frame.__init__(self, master, *args, **kwargs)
    self.content = content
    self.content_size = (len(content), len(content[0]))
    self._create_labels()
    self._display_labels()

  def _create_labels(self):
    def __put_content_in_label(row, column):
      content = self.content[row][column]
      content_type = type(content).__name__
      if content_type in ('str', 'int'):
        self.labels[row][column]['text'] = content
      elif content_type == 'PhotoImage':
        self.labels[row][column]['image'] = content

    self.labels = list()
    for i in range(self.content_size[0]):
      self.labels.append(list())
      for j in range(self.content_size[1]):
        self.labels[i].append(Label(self))
        __put_content_in_label(i, j)

  def _display_labels(self):
    for i in range(self.content_size[0]):
      for j in range(self.content_size[1]):
        self.labels[i][j].grid(row=i, column=j)


# Function for opening the
# file explorer window
def browseFiles():
  filename = filedialog.askopenfilename(initialdir="./testModels/", title="Select a File",
                                        filetypes=(("all files", "*.*"), ("Text files", "*.txt*")))

  # Change label contents
  label_file_explorer.configure(text="File Opened: " + filename)
  return filename


def show_similar():
  path = browseFiles()
  mesh = main.load_from_file(path)
  
  distances, processed_mesh = shaperetrieval.query(path)
  colorvisuals = trimesh.visual.ColorVisuals(processed_mesh, [255, 0, 0, 200])
  processed_mesh.visual = colorvisuals
  df = utils.read_excel(original=False)
  indices = [distance[1] for distance in distances[:5]]
  rows = df.iloc[indices]
  paths = list(rows['path'])
  distances = [distance[0] for distance in distances[:5]] # todo show this in window
  meshes = [processed_mesh] + (shaperetrieval.paths_to_meshes(paths))
  main.compare(meshes, setcolor=False)

def get_image_paths(shape_class):
  classpath = utils.sim_images_path + shape_class + '/'
  paths = []
  for classFolder in os.listdir(classpath):
    images = []
    for image in os.listdir(classpath + classFolder):
      images.append(classpath + classFolder + '/' + image)
    paths.append(images)
  return paths  

#x = get_image_paths(18)
# Create the root window
window = Tk()

# Set window title
window.title('Mesh Explorer')

# Set window size
window.geometry("1920x1080")

# Set window background color
window.config(background="white")

# Create a File Explorer label
label_file_explorer = Label(window, text="Placeholder text", width=75, height=4, fg="blue")

# Create buttons
button_explore = Button(window, text="Browse files", command=browseFiles)
button_sim = Button(window, text='Show similar', command=show_similar)
button_exit = Button(window, text="Exit", command=exit)

# Grid method is chosen for placing
# the widgets at respective positions
# in a table like structure by
# specifying rows and columns
label_file_explorer.grid(column=1, row=1)
button_explore.grid(column=1, row=2)
button_sim.grid(column=1, row=4)
button_exit.grid(column=1, row=3)

# Let the window wait for any events
window.mainloop()
