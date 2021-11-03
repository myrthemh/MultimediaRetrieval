from matplotlib.pyplot import yscale
from numpy.lib.shape_base import dsplit
import utils
from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt
def roc():
  df = utils.read_excel(original=False)
  for class_number in range(len(utils.classes)):
    class_rows = df.loc[df['class'] == class_number]
    distances = np.asarray(class_rows['similar_meshes'])
    y_score =  np.array([1 if int(x[2]) == class_number else 0 for tuples in distances for x in tuples])
    y_true = np.array([int(x[2]) for tuples in distances for x in tuples])
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=class_number)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)",
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()
    print(lw)
roc()

