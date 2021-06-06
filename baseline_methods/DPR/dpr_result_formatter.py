"""Analyze DPR retrived results."""

import json

from absl import app, flags
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import collections
import numpy as np
import pickle

import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True
})

FLAGS = flags.FLAGS
flags.DEFINE_string("linked_qa_file", "",
                    "The path to the linked qa files.")
flags.DEFINE_string("dpr_result_file", "",
                    "The path to the linked qa files.")
flags.DEFINE_string("drfact_format_gkb_file", "",
                    "The path to the linked qa files.")
flags.DEFINE_string("output_prefix", "DPR",
                    "The path to the linked qa files.")


def evaluate_DPR_hits(DPR_results, corpus_dict):
  """Formats the retrieved results."""
  with open(FLAGS.linked_qa_file) as f:
    instances = [json.loads(line) for line in f.read().split("\n") if line]
  assert len(DPR_results) == len(instances)
  num_questions = len(instances)

  def get_results(question_id):
    assert 0 <= question_id < num_questions
    result = DPR_results[question_id]
    ret_fids = []
    ret_scores = []
    for fid, score in zip(result[0], result[1]):
      fact_id = "gkb-best#%s" % fid
      if fact_id not in corpus_dict:
        continue
      ret_fids.append(fact_id)
      ret_scores.append(float(score))
    return ret_fids, ret_scores 
  
  result_data = []
  # num_top_correct = 0
  for question_id, ins in tqdm(
          enumerate(instances[:]),
          total=num_questions, desc=FLAGS.linked_qa_file):
    ret_fids, ret_scores = get_results(question_id)
    all_ret_facts = [(fid, s) for fid, s in zip(ret_fids, ret_scores)]
    DPR_res = {}
    DPR_res["all_ret_facts"] = all_ret_facts
    ins["results"] = DPR_res

    result_data.append(ins) 
    
  with open(FLAGS.linked_qa_file+"."+FLAGS.output_prefix+".jsonl", "w") as f:
    f.write("\n".join([json.dumps(r) for r in result_data])) 


def main(_):
  """Run the results formatting."""

  if "_with_ans" in FLAGS.linked_qa_file:
    FLAGS.linked_qa_file = FLAGS.linked_qa_file.replace("_with_ans", "")
    FLAGS.output_prefix += "_with_ans"

  with open(FLAGS.dpr_result_file, "rb") as f:
    print("Reading", f.name)
    DPR_results = pickle.load(f)
  with open(FLAGS.drfact_format_gkb_file) as f:
    corpus_dict = {}
    for line in f.read().split("\n"):
      if line:
        instance = json.loads(line)
        corpus_dict[instance["id"]] = instance

  evaluate_DPR_hits(DPR_results, corpus_dict)
 

if __name__ == "__main__":
  app.run(main)
