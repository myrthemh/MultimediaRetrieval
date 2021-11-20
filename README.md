# MultimediaRetrieval

All packages are installed on Python 3.8.x. Newer or older versions might not work.

### 1. Install required packages

Install the required Python Packages with PIP:

```bash
pip install -r requirements.txt
```

### 2. Additional dependencies

None so far

### 3. How to run and see results:

1. Optionally change some of the parameters in `utils.py`.
2. Run the main.py script, which will perform every step of the assignment in order. (Can take multiple hours)
3. After the running of `main.py` is concluded, the results can be found in a number of different places:
   -In the graphs folder, the histograms belonging to the original and refined databases can be found
   -In the features folder, excel files (and pickle files) contain all the information that was saved for the original and refined database (including the features). Note that the excel files are only used to view the data manually, the program only uses the pickle files. Therefore changing anything in the excel files won't have any effect on the program results.
   -When opening images.html, which should have appeared in the main folder, in the browser, the first five query results for our custom distance method can be compared to the ANN query results along with the distance values.
   -In the 'evalimages' folder, the different ROC curve graphs can be seen for every different weight vector specified in `utils.py` (the images are numbered by weight vector index)
4. `GUI.py` can be run to open a (very primitive) gui which allows online querying for shapes that don't exist in the database. When pressing the 'show similar meshes' button, a mesh file can be opened which will then be used as the query shape. Some demo models can be found in the 'demo_models' folder.
