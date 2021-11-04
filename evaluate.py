from matplotlib.pyplot import yscale
from numpy.lib.shape_base import dsplit
from matplotlib.ticker import PercentFormatter
import utils
from sklearn.metrics import auc
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
  for c, value in enumerate(utils.classes):
    results = []
    for query_result in DB.loc[DB["class"] == c, "similar_meshes" ]:
      results.append(ktier(query_result, c))

    plt.hist(results,  weights=np.ones(len(results)) / len(results))

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlim(1,5)
    plt.xlabel("Tier")
    plt.ylabel("Percentage returned in tier")
    plt.title("Percentage of total {} correctly returned in first five tiers".format(str(utils.classes[int(c)])))

    path = utils.refinedImagePath
    utils.ensure_dir(path)
    plt.savefig(path + "tierOfClass" + str(c) + '.png')
    plt.show()



def ktier(query_result, class_value):
  tier = 0
  for q in query_result:
    if q[2] == class_value:
      tier += 1
    else:
      return tier
  return tier



def roc(allclasses=False):
  df = utils.read_excel(original=False)
  if allclasses:
   plt.figure(figsize=(12, 12), dpi=80)
  for class_number in range(len(utils.classes)):
    class_rows = df.loc[df['class'] == class_number]
    distances  = class_rows['similar_meshes']
    x = list(distances)
    correct_classes = np.array(list(distances.map(lambda shape: [1 if int(x[2]) == class_number else 0 for x in shape])))
    sensitivities = []
    specificities = []
    for i in range(0, correct_classes.shape[1]):
      TP = np.sum(correct_classes[:, :i]) #Correct objects returned in the query
      FN = np.sum(correct_classes[:, i:]) #Correct objects not returned in the query
      FP = correct_classes[:, :i].size - TP #Objects returned in the query that should not have been returned
      TN = correct_classes[:, i:].size - FN #Objects not returned in the query that should indeed not be returned
      sensitivity = TP / (TP + FN)
      specificity = TN / (FP + TN)
      sensitivities.append(sensitivity)
      specificities.append(specificity)

    #Plot
    lw = 2
    if not allclasses:
      plt.figure()
    plt.plot(
        specificities,
        sensitivities,
        lw=lw,
        label=f"Class: {utils.classes[class_number]}, Area = {round(auc(sensitivities, specificities), 3)}",
    )
    if not allclasses:
      plt.plot([0, 1], [1, 0], color="navy", lw=lw, linestyle="--")
      plt.xlim([0.0, 1.01])
      plt.ylim([0.0, 1.01])
      plt.xlabel("Specificity")
      plt.ylabel("Sensitivity")
      plt.title(F"Roc curve {utils.classes[class_number]}")
      plt.legend(loc="lower left")
      path = F"{utils.eval_images_path}roc/{utils.classes[class_number]}.png"
      utils.ensure_dir(path)
      plt.savefig(path, bbox_inches='tight')
  if allclasses:
    plt.plot([0, 1], [1, 0], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.xlabel("Specificity")
    plt.ylabel("Sensitivity")
    plt.title("Receiver operating characteristic curve")
    plt.legend(loc="lower left")
    path = F"{utils.eval_images_path}roc/allclasses.png"
    utils.ensure_dir(path)
    plt.savefig(path, bbox_inches='tight')
roc()

