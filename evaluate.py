import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import auc

import shaperetrieval
import utils


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

  metric_performance_avg = metric_performance_avg / len(DB)
  return metric_performance_class, metric_performance_avg


def evaluate_ktier(DB):
  # info = {"xlabel": }
  for c, value in enumerate(utils.classes):
    results = []
    for query_result in DB.loc[DB["class"] == value, "similar_meshes"]:
      results.append(ktier(query_result, c))

    plt.hist(results, weights=np.ones(len(results)) / len(results))

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlim(1, 5)
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


def roc(correct_classes):
  sensitivities = []
  specificities = []
  for i in range(0, correct_classes.shape[1]):
    TP = np.sum(correct_classes[:, :i])  # Correct objects returned in the query
    FN = np.sum(correct_classes[:, i:])  # Correct objects not returned in the query
    FP = correct_classes[:, :i].size - TP  # Objects returned in the query that should not have been returned
    TN = correct_classes[:, i:].size - FN  # Objects not returned in the query that should indeed not be returned
    sensitivity = TP / (TP + FN)
    specificity = TN / (FP + TN)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
  return sensitivities, specificities


def roc_plots():
  df = utils.read_excel(original=False)
  plot_data = []
  for column in ["similar_meshes", "ANN"]:
    plt.clf()
    plt.figure(figsize=(15, 15), dpi=80)
    all_correct_classes = []
    for shape_class in utils.classes:
      class_rows = df.loc[df['class'] == shape_class]
      if len(class_rows) == 0:
        continue
      distances = class_rows[column]
      correct_classes = np.array(list(distances.map(lambda shape: [1 if x[2] == shape_class else 0 for x in shape])))
      all_correct_classes.append(correct_classes)
      sensitivities, specificities = roc(correct_classes)
      # Plot
      lw = 2
      plt.plot(
          sensitivities,
          specificities,
          lw=lw,
          label=f"Class: {shape_class}, Area = {round(auc(sensitivities, specificities), 3)}",
      )
      plt.plot([0, 1], [1, 0], color="navy", lw=lw, linestyle="--")
      plt.xlim([0.0, 1.01])
      plt.ylim([0.0, 1.01])
      plt.xlabel("Sensitivity")
      plt.ylabel("Specificity")
      plt.legend(loc="lower left")
    if column == "similar_meshes":
      save_path = "our_roc"
    else:
      save_path = "ann_roc"
    # Plot all classes in the same plot
    plt.title("Receiver operating characteristic curve")
    path = F"{utils.eval_images_path}{save_path}_allclasses.png"
    utils.ensure_dir(path)
    plt.savefig(path, bbox_inches='tight')
    plot_data.append(all_correct_classes)

  # plot average of all classes for ANN and Ours
  plt.clf()
  plt.figure()
  for i, data in enumerate(plot_data):
    all_correct_classes = np.concatenate(np.asarray(data))
    sensitivities, specificities = roc(all_correct_classes)
    plt.plot(
      sensitivities,
      specificities,
      lw=lw,
      label=f"{['Ours', 'ANN'][i]}, Area = {round(auc(sensitivities, specificities), 3)}",
    )
    plt.plot([0, 1], [1, 0], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.xlabel("Sensitivity")
    plt.ylabel("Specificity")
    plt.legend(loc="lower left")
    plt.title("ROC curve averaged over all queries over all classes")
    path = F"{utils.eval_images_path}{save_path}_avgallclasses.png"
    utils.ensure_dir(path)
    plt.savefig(path, bbox_inches='tight')


def time_queries(runs=100):
  ann_times = []
  custom_times = []
  df = utils.read_excel(original=False)
  len_df = len(df.index)
  u = shaperetrieval.load_map_neighbours("testmodels.ann", 106, 'euclidean')
  for i in range(runs):
    print(i)
    random_number = random.randint(0, len_df - 1)
    start = time.perf_counter()
    shaperetrieval.find_similar_meshes(df.iloc[random_number])
    end = time.perf_counter()
    custom_times.append(end - start)
    start = time.perf_counter()
    u.get_nns_by_item(i, len_df, include_distances=True)
    end = time.perf_counter()
    ann_times.append(end - start)
  return ann_times, custom_times


def boxplot_queries(show=False):
  plt.rcParams.update(plt.rcParamsDefault)
  ann_times, custom_times = time_queries()
  df = pd.DataFrame(columns=("Annoy", "Custom"))
  df = df.assign(Annoy=ann_times, Custom=custom_times)
  fig1, ax1 = plt.subplots()
  ax1.set_title('Query times for Annoy and custom')
  ax1.boxplot(df)
  plt.xticks([1, 2], ["Annoy", "Custom"])
  plt.ylabel("Time (seconds)")
  plt.xlabel("Query method")
  plt.savefig('./evalimages/querytimes.png' )
  if show:
    plt.show()

# boxplot_queries()

# roc()
