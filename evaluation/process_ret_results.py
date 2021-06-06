"""Post-prorocessing the BM25/DPR results for computing the metics."""

import json
import collections
from collections import defaultdict
from tqdm import tqdm
from absl import app, flags
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string("linked_qa_ret_file", "",
                    "The input path to the linked qa file's retrieved results.")
flags.DEFINE_string("pred_result_file", "", # Only for BM25/DPR methods because they need concept aggregation for ranking concepts.
                    "The input/output path to the linked qa file's prediction results.") # output for BM25/DPR, and input for other methods.
flags.DEFINE_string("eval_result_file", "",
                    "The output path to save the evaluation results.")
flags.DEFINE_string("ent_agg", "max", "max | mean | sum")
flags.DEFINE_string("drfact_format_gkb_file", "drfact_data/knowledge_corpus/gkb_best.drfact_format.jsonl", "Path to gkb corpus.")
# flags.DEFINE_boolean("need_process", False, "set --need_process for BM25 and DPR.")  # not set for DrFact/DrKIT and x+Reranker


def produce_prediction(instances, facts_dict, agg_method=np.max):
  
  final_prediction = {}
  thresholds = [20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
  for ins in tqdm(instances, desc=FLAGS.linked_qa_ret_file):
    # question_concepts = set([m["kb_id"] for m in ins["entities"]]) 
    answer_concepts = list([a["name"] for a in ins["answer_concepts"]])
        
    all_ret_facts = ins["results"]["all_ret_facts"]

    ret_concept_scores = defaultdict(list)
    thre_id = 0
    predictions_K = {}
    total_count = len(all_ret_facts)
    for ind, (fid, score) in enumerate(all_ret_facts):
      fact = facts_dict[fid] 
      for m in fact["mentions"]:
        c = m["kb_id"]
        ret_concept_scores[c].append(score)
      if ind + 1 == thresholds[thre_id] or ind+1 == total_count:
        # Aggregate
        agg_res = {c: agg_method(scores) for c, scores in ret_concept_scores.items()}
        # Sort
        concept_predictions = [(k, v) for k, v in sorted(agg_res.items(), key=lambda x: x[1], reverse=True)][:1000]
        predictions_K[thresholds[thre_id]] = concept_predictions
        thre_id += 1
        if thre_id == len(thresholds):
          break
    final_prediction[ins["_id"]] = dict(qid = ins["_id"], \
                                        question = ins["question"], \
                                        predictions_K = predictions_K, \
                                        answers = answer_concepts, \
                                        )
    # metadata[ins["_id"]] = dict(qid = ins["_id"], \
    #                             question = ins["question"], \
    #                             answers = remove_nested(choice2concepts[ans]), \
    #                             distractors = list(distractors))
  return final_prediction



def main(_):
  """Main fucntion."""  
  # for BM25 and DPR, we need aggregate their concept scores for ranking.
  instances = []
  with open(FLAGS.linked_qa_ret_file) as f:
    print("Reading %s ..."%f.name)
    for line in tqdm(f):
      if line.strip():
        instances.append(json.loads(line))

  
  with open(FLAGS.drfact_format_gkb_file) as f:
    print("Reading %s ..."%f.name)
    facts_dict = {}
    for line in f:
      if line.strip():
        instance = json.loads(line)
        facts_dict[instance["id"]] = instance
        
  # Compute concept output.
  if FLAGS.ent_agg == "mean":
    agg_method = np.mean
  elif FLAGS.ent_agg == "max":
    agg_method = np.max
  elif FLAGS.ent_agg == "sum":
    agg_method = np.sum
  else:
    print("Wrong agg method")
    exit()
    
  final_prediction = produce_prediction(instances, facts_dict, agg_method)
  # Save to file.
  with open(FLAGS.pred_result_file, "w") as f:
    print("Writing %s..."%f.name)
    for _, pred_res in final_prediction.items():
      f.write(json.dumps(pred_res) + "\n")
    f.write("\n")  




if __name__ == "__main__":
  app.run(main)
