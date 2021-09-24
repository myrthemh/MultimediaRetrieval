import pandas as pd

excelPath = "features/data.xlsx"
imagePath = "graphs/"
originalDB = "testModels/db"
refinedDB = ""
def read_excel():
  # Load the excel into a pandas df
  return pd.read_excel(excelPath, index_col=0)