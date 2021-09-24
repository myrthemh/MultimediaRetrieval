import pandas as pd
import os

excelPath = "features/original.xlsx"
refinedexcelPath = "features/refined.xlsx"
imagePath = "graphs/original/"
refinedImagePath = "graphs/refined/"
originalDB = "testModels/db"
refinedDB = "testModels/refined_db"
target_vertices = 1000


def read_excel(original=True):
  # Load the excel into a pandas df
  if original:
    return pd.read_excel(excelPath, index_col=0)
  else:
    return pd.read_excel(refinedexcelPath, index_col=0)

def ensure_dir(file_path):
  directory = os.path.dirname(file_path)
  if not os.path.exists(directory):
    os.makedirs(directory)