"""Converts the BM25/DPR result over train/dev/test for mcqa format."""
  
import json
import collections
from collections import defaultdict
from tqdm import tqdm
from absl import app, flags
import numpy as np
import random
import copy

random.seed(42)

FLAGS = flags.FLAGS
 
flags.DEFINE_string("eval_result_file", "baseline_methods/BM25/results/ARC_max_result.jsonl",
                    "The path to the linked qa file's BM25/DPR results.") 
flags.DEFINE_integer("top_K", 1000,
                    "The path to the linked qa file's BM25/DPR results.") 
flags.DEFINE_integer("num_distractors", 4,
                    "The path to the linked qa file's BM25/DPR results.") 
flags.DEFINE_string("mcqa_file", "baseline_methods/MCQA/data/ARC_BM25.jsonl",
                    "The path to the linked qa file's BM25/DPR results.") 
flags.DEFINE_boolean("if_test", False,
                    "The path to the linked qa file's BM25/DPR results.") 
flags.DEFINE_string("concept_vocab", "drfact_data/knowledge_corpus/gkb_best.vocab.txt",
                    "The path to the linked qa file's BM25/DPR results.") 



def main(_):
  """Main fucntion.""" 
  all_concepts = []
  with open(FLAGS.concept_vocab) as f:
    for line in f.read().splitlines():
      all_concepts.append((line, -1))
  with open(FLAGS.eval_result_file) as f:
    lines = f.read().splitlines()
  output_lines = []
  num_empty = 0
  for line in tqdm(lines, desc="Processing %s"%FLAGS.eval_result_file):
    if not line:
      continue
    data = json.loads(line) 
    qid = data["qid"]
    question_text = data["question"]
    # if str(FLAGS.top_K) not in data["predictions_K"]: 
    all_keys = [int(k) for k in data["predictions_K"].keys()]
    max_k = max(all_keys)
    ranked_list_concepts = data["predictions_K"][str(max_k)]
    # else:
    #   ranked_list_concepts = data["predictions_K"][str(FLAGS.top_K)]
    answer_concepts = data["answers"]
    if FLAGS.if_test:
      if len(ranked_list_concepts) == 0:
        print(qid)
        # print(len(choices), len(ranked_list_concepts), data["predictions_K"].keys())
        num_empty += 1
        ranked_list_concepts = all_concepts
      choices = [ {"text": t[0] , "label": str(i+1)} for i, t in enumerate(ranked_list_concepts[:FLAGS.top_K])]
      if len(choices) < FLAGS.top_K:
        print(qid, len(choices))
        choices += [ {"text": t[0] , "label": str(i+1)} for i, t in enumerate(all_concepts[:FLAGS.top_K-len(choices)])]
      assert len(choices) == FLAGS.top_K
      
      answerKey = "-1"
    else:
      distractors = [t[0] for t in random.sample(ranked_list_concepts[:FLAGS.top_K], k=FLAGS.num_distractors) \
        if t not in answer_concepts] 
      choices = [ {"text": d , "label": str(i+1)} for i, d in enumerate(distractors)]
      choices.append( {"text": answer_concepts[0] , "label": str(len(choices)+1)})
      choice_ind = random.randint(0, len(choices)-1)
      tmp_choice =  copy.deepcopy(choices[choice_ind])
      choices[choice_ind]["text"] = choices[-1]["text"]
      choices[-1]["text"] = tmp_choice["text"]
      answerKey = tmp_choice["label"]
    qas_instance = dict(id=qid, question={"stem":question_text, "choices": choices}, answerKey=answerKey)
    output_lines.append(json.dumps(qas_instance))
  with open(FLAGS.mcqa_file, "w") as f:
    for line in output_lines:
      f.write(line + "\n")
    print("Empty instances at ", f.name, ":", num_empty)
    

if __name__ == "__main__":
  app.run(main)