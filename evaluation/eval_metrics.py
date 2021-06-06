"""Analyze the BM25 results for accuracy metrics."""

import json
import collections
from collections import defaultdict
from tqdm import tqdm
from absl import app, flags
import numpy as np
import random

FLAGS = flags.FLAGS
 
flags.DEFINE_string("pred_result_file", "baseline_methods/BM25/results/QASC_dev_result.jsonl",
                    "The path to the prediction results.")
flags.DEFINE_string("truth_data_file", "drfact_data/datasets/QASC/linked_dev.jsonl",
                    "The path to the linked qa file.")


def hit_at_K(final_prediction):
  thresholds = list(range(10, 301, 10))
  results = defaultdict(lambda: defaultdict(lambda : 0)) 
  for _, item in final_prediction.items():
    ans = set(item["answers"])
    for K, results_at_K in item["predictions_K"].items():
      K = int(K)
      concept2rank = {}
      for ind, (concept, _) in enumerate(results_at_K):
        if concept in ans:
          concept2rank[concept] = ind
      for T in thresholds:
        if len(concept2rank) >=1 and any([r <= T for c, r in concept2rank.items()]):
          results[K][T] += 1 

  for K in results:
    K = int(K)
    for T in results[K]:
      results[K] = dict(results[K])
      results[K][T] = results[K].get(T, 0) / len(final_prediction)
  return results

def recall_at_K(final_prediction):
  thresholds = list(range(10, 301, 10))
  results = defaultdict(lambda: defaultdict(lambda : 0)) 
  for _, item in final_prediction.items():
    ans = set(item["answers"])
    for K, results_at_K in item["predictions_K"].items():
      K = int(K)
      concept2rank = {}
      for ind, (concept, _) in enumerate(results_at_K):
        if concept in ans:
          concept2rank[concept] = ind
      for T in thresholds:
        found_concepts = [c for c, r in concept2rank.items() if r <= T] 
        results[K][T] += len(found_concepts)/len(ans) 

  for K in results:
    K = int(K)
    for T in results[K]:
      results[K] = dict(results[K])
      results[K][T] = results[K].get(T, 0) / len(final_prediction)
  return results

def clean_data(item):
    clean_item = {}
    clean_item["qid"] = item["_id"]
    def get_concepts(ent_list):
      return [ent["name"] for ent in ent_list]
    clean_item["all_answers"] = get_concepts(item["all_answer_concepts"])
    return clean_item

def main(_):
  """Main fucntion."""
  truth_data = {}
  with open(FLAGS.truth_data_file, "r") as f:
    lines = f.read().splitlines()
    for line in lines:
      if not line:
        continue
      data = json.loads(line)
      data = clean_data(data)
      truth_data[data["qid"]] = data

  final_prediction = {}
  with open(FLAGS.pred_result_file, "r") as f:
    # print("Reading %s..."%f.name)
    for line in f.read().splitlines():
      if not line:
        continue
      item = json.loads(line)
      # print(item["answers"], truth_data[item["qid"]]["answers"])
      item["answers"] = truth_data[item["qid"]]["all_answers"] 
      final_prediction[item["qid"]] = item
 

  Ks = [100]
  Ts = list(range(10, 201, 10))
  hk_accs = hit_at_K(final_prediction)
  line = []
  
  for K in Ks: 
    for T in Ts:
      line.append("%.4f"%(hk_accs[K][T]))
  print(",".join(line))


  rk_accs = recall_at_K(final_prediction)
  line = []
  for K in Ks: 
    for T in Ts:
      line.append("%.4f"%(rk_accs[K][T]))
  print(",".join(line))
 

if __name__ == "__main__":
  app.run(main)
