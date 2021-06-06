# Lint as: python3
"""Convert OpenCSR dataset with BM25/DPR(nq) results to be DPR data for learning retriever."""
import json

from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm
import random 

random.seed(42)

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_name", None, "Path to dataset file.")
flags.DEFINE_string("linked_qas_file", None, "Path to dataset file.")
flags.DEFINE_string("drfact_format_gkb_file", None, "Path to gkb corpus.")
flags.DEFINE_string("ret_result_file", None, "Path to dataset file.") # retrieved file
flags.DEFINE_string("output_file", None, "Path to dataset file.")
flags.DEFINE_integer("cut_off", 1000, "Path to dataset file.")
flags.DEFINE_integer("max_pos_facts", 10, "Path to dataset file.")
flags.DEFINE_integer("max_neg_facts", 10, "Path to dataset file.")
flags.DEFINE_integer("max_hard_neg_facts", 10, "Path to dataset file.")


def main(_):
  """Main funciton."""
  logging.set_verbosity(logging.INFO)

  with open(FLAGS.drfact_format_gkb_file) as f:
    logging.info("Reading %s..."%f.name)
    facts_dict = {}
    for line in f.read().split("\n"):
      if line:
        instance = json.loads(line)
        facts_dict[instance["id"]] = instance
  
  with open(FLAGS.ret_result_file) as f:
    logging.info("Reading %s..."%f.name)
    ret_data = [json.loads(line) for line in f.read().splitlines()]

  logging.info("Reading QAS(-formatted) data...")
  with open(FLAGS.linked_qas_file) as f:
    jsonlines = f.read().split("\n")
  qas_data = [json.loads(jsonline) for jsonline in jsonlines if jsonline]

  assert len(ret_data) == len(qas_data)

  dpr_format_data = []
  
  for ind, ins in tqdm(enumerate(qas_data), desc=FLAGS.linked_qas_file, total=len(qas_data)):
    dpr_ins = {}
    dpr_ins["dataset"] = FLAGS.dataset_name
    dpr_ins["question"] = ins["question"]
    dpr_ins["answers"] = [c["kb_id"] for c in ins["all_answer_concepts"]]
    all_positive_ctxs = []  # facts with answer concepts
    all_negative_ctxs = []  # facts w/o answer concepts
    all_ret_facts = ret_data[ind]["results"]["all_ret_facts"]
    
    # question_concepts = set([c["kb_id"] for c in ins["entities"]])
    answer_concepts = set(dpr_ins["answers"])
    
    for fid, score in all_ret_facts[:FLAGS.cut_off]:
      fact = facts_dict[fid]
      fact_concepts = set([m["kb_id"] for m in fact["mentions"]])
      contain_answer = fact_concepts & answer_concepts
      ctx = {}
      ctx["text"] = fact["context"]
      ctx["title"] = fact["url"] # not used in fact
      ctx["passage_id"] = fact["url"] # not used in fact
      ctx["score"] = score  # not used in fact
      ctx["title_score"] = 0  # not used in fact
      if contain_answer:
        all_positive_ctxs.append(ctx)
      else:
        all_negative_ctxs.append(ctx)
    
    dpr_ins["positive_ctxs"] = all_positive_ctxs[:FLAGS.max_pos_facts]
    dpr_ins["hard_negative_ctxs"] = all_negative_ctxs[:FLAGS.max_hard_neg_facts]
    k = min(FLAGS.max_neg_facts, max(len(all_negative_ctxs)-FLAGS.max_hard_neg_facts, 0))
    dpr_ins["negative_ctxs"] = random.sample(all_negative_ctxs[FLAGS.max_hard_neg_facts:], min(k, len(all_negative_ctxs)))
    
    
    if len(dpr_ins["positive_ctxs"]) >= 1:
      dpr_format_data.append(dpr_ins)

  with open(FLAGS.output_file, "w") as f:
    logging.info("Converted percentage %.2f", len(dpr_format_data)/len(qas_data))
    logging.info("Writing to %s", f.name)
    f.write("\n".join([json.dumps(i) for i in dpr_format_data])+"\n")
  logging.info("Done.")

if __name__ == "__main__":
  app.run(main)
