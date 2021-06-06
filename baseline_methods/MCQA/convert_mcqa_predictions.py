"""Converts the BM25/DPR + MCQA results (in npy format) over test for normal format."""
import json
import collections
from collections import defaultdict
from tqdm import tqdm
from absl import app, flags
import pickle
import numpy as np


FLAGS = flags.FLAGS
 
flags.DEFINE_string("pred_result_file", "baseline_methods/MCQA/data/ARC/test.DPR.npy",
                    "The path to the linked qa file's BM25/DPR results.")  
flags.DEFINE_string("mcqa_file", "baseline_methods/MCQA/data/ARC/test.DPR.jsonl",
                    "The path to the linked qa file's BM25/DPR results.") 
flags.DEFINE_string("reference_result_file", "baseline_methods/DPR/results/ARC_dev_max_result.jsonl",
                    "The path to the linked qa file's BM25/DPR results.") 
flags.DEFINE_string("converted_result_file", "baseline_methods/MCQA/results/ARC_DPR_MCQA_result.jsonl",
                    "The path to the linked qa file's BM25/DPR results.")                   


def main(_):
  """Main fucntion.""" 
  instances = []
  with open(FLAGS.mcqa_file) as f:
    for line in f.read().splitlines():
      if line:
        instances.append(json.loads(line))
  with open(FLAGS.pred_result_file, "rb") as f:
    predictions = pickle.load(f)
  assert len(predictions) == len(instances)
  
  original_result = []
  with open(FLAGS.reference_result_file) as f:
    for line in f.read().splitlines():
      if line:
        original_result.append(json.loads(line))
  assert len(predictions) == len(original_result)

  for ind, (inst, pred) in enumerate(zip(instances, predictions)):
    choices = inst["question"]["choices"]
    assert len(choices) == len(pred)
    ranked_list = [(c["text"], float(p)) for c, p in zip(choices, pred)]
    ranked_list.sort(key=lambda x: x[1], reverse=True)
    for key in original_result[ind]["predictions_K"].keys():
      original_result[ind]["predictions_K"][key] = ranked_list[:int(key)]
  with open(FLAGS.converted_result_file, "w") as f:
    for inst in original_result:
      f.write(json.dumps(inst) + "\n")
    print("Finish", f.name)


if __name__ == "__main__":
  app.run(main)