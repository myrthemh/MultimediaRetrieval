from tkinter import *
# import filedialog module
from tkinter import filedialog

import trimesh

import main
import shaperetrieval


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
  mesh.visual = trimesh.visual.ColorVisuals(mesh, [255, 0, 0, 255])
  distances = shaperetrieval.find_similar_meshes(mesh)
  paths = [distance[1] for distance in distances[:5]]
  distances = [distance[0] for distance in distances[:5]]  # todo show this in window
  meshes = shaperetrieval.paths_to_meshes(paths)
  meshes.append(mesh)
  main.compare(meshes, setcolor=False)


# Create the root window
window = Tk()

# Set window title
window.title('Mesh Explorer')

# Set window size
window.geometry("500x500")

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
