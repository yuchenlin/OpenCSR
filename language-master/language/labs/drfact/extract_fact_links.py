# Lint as: python3
"""Adds pre-computed results as initial facts for OpenCSR data."""
import json

from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string("sup_facts_file", None, "Path to dataset file.")
flags.DEFINE_string("fact_links_file", None, "Path to dataset file.")


def main(_):
  """Main funciton."""
  logging.set_verbosity(logging.INFO)
  sup_facts_data = [] 
  with open(FLAGS.sup_facts_file) as f:
    jsonlines = f.read().splitlines()
  sup_facts_data = [json.loads(jsonline) for jsonline in jsonlines]
  
  with open(FLAGS.fact_links_file, "r") as f:
    logging.info("Reading %s", FLAGS.fact_links_file)
    fact_links = []
    for line in f.read().splitlines():
      if not line:
        continue
      f1, f2 = line.split()
      fact_links.append((f1, f2))
    fact_links = set(fact_links)
  for ind, ins in tqdm(enumerate(sup_facts_data), desc=FLAGS.sup_facts_file, total=len(sup_facts_data)):
    new_fid1  = set()
    new_fid2  = set()
    for sup_item in sup_facts_data[ind]["sup_facts"]:
      item_supfacts = sup_item[0] # a list of facts
      if len(item_supfacts) == 1: # One-hop quesiton
        fid = item_supfacts[0][0]
        score = item_supfacts[0][1]
        # TODO: self-link
        fact_links.add((fid, fid))
        # new_fid1.add(fid)
        # new_fid2.add(fid)
      elif len(item_supfacts) == 2: # Two-hop quesiton
        fid_1 = item_supfacts[0][0]
        score_1 = item_supfacts[0][1]
        fid_2 = item_supfacts[1][0]
        score_2 = item_supfacts[1][1]
        new_fid1.add(fid_1)
        new_fid2.add(fid_2)
        # if (fid_1, fid_2) not in fact_links:
        #   fact_links.add((fid_1, fid_2))
    for f1 in new_fid1:
      for f2 in new_fid2:
        fact_links.add((f1, f2))


  with open(FLAGS.fact_links_file, "w") as f: 
    for f1, f2 in list(fact_links):
      line = "%s\t%s\n"%(f1, f2)
      f.write(line)
    logging.info("Done! %s", FLAGS.sup_facts_file)
 
if __name__ == "__main__":
  app.run(main)
