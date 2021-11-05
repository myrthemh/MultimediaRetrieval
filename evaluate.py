from matplotlib.pyplot import yscale
from numpy.lib.shape_base import dsplit
from matplotlib.ticker import PercentFormatter
import utils
from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt


def evaluate_score(DB, metric):
  metric_performance_class = list(np.repeat(0, len(utils.classes)))
  metric_performance_avg = 0

  for query_result, class_value in zip(DB["similar_meshes"], DB["class"]):
    if metric == "ktier":
      score = ktier(query_result, class_value)

    if metric == "roc":
      score = roc(query_result, class_value)

    metric_performance_class[int(class_value)] += score
    metric_performance_avg += score

  for c in range(len(utils.classes)):
    metric_performance_class[c] /= len(DB[DB["class"] == c])

  metric_performance_avg = metric_performance_avg /len(DB)
  return metric_performance_class, metric_performance_avg

def evaluate_ktier(DB):
  # info = {"xlabel": }
  plt.rcParams.update(plt.rcParamsDefault)
  for c, value in enumerate(utils.classes):
    c_len = len(DB.loc[DB["class"] == c])
    result = list(np.zeros(6))
    for query_result in DB.loc[DB["class"] == c, 'similar_meshes']:
      result = ktier(result, query_result, c, c_len)

    results = [i+1 for i in range(0, 6) for j in range(0, int(result[i]))]

    plt.hist(results, bins=np.arange(0, 7, 1),  weights=np.ones(len(results)) / len(results))

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.ylim(0, 1)
    plt.xlim(1, 6)

    plt.xlabel("Tier")
    plt.ylabel("Hoi Percentage returned in tier")
    plt.title("Percentage of total {} correctly returned in first five tiers".format(str(utils.classes[int(c)])))

    path = utils.refinedImagePath
    utils.ensure_dir(path)
    plt.savefig(path + "tierOfClass" + str(c) + '.png')
    plt.show()

def lasttier(query_result, class_value):
  tier = 0
  for q in query_result:
    if q[2] == class_value:
      tier += 1
    else:
      return tier
  return tier

def ktier(result, query_result, class_value, clen):
  tier = 0

  # for c, value in enumerate(utils.classes):
  #   for query_result in DB.loc[DB["class"] == c, 'similar_meshes']:
  for i in range(0, 6):

    for j in range(0, clen):
       if query_result[i+j][2] == class_value:
        result[i] += 1

  return result

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
#roc()

