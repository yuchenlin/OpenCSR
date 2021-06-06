"""Converts the qas format json to csv."""
import sys
import json
from tqdm import tqdm
import spacy

nlp = spacy.load("en_core_web_sm")
nlp.pipeline = [('tagger', nlp.tagger)]

filename = sys.argv[1]
append_answer = sys.argv[2]
if append_answer == "yes":
  output_name = filename.replace(".jsonl", "_with_ans.csv")
else:
  output_name = filename.replace(".jsonl", ".csv")

with open(filename) as f:
  lines = f.read().split("\n")
instances = [json.loads(l) for l in lines if l]
tsv = ""
for ind, ins in tqdm(enumerate(instances), total=len(instances), desc=filename):
  question = ins["question"].replace("\t", " ")
  answer = ins["answer"]
  answer_concepts = [a["name"] for a in ins["answer_concepts"]]
  answer_list_str = json.dumps(answer_concepts)
  if append_answer == "yes":
    # Add the answer as well to the query for collecting justification facts
    question_text = " ".join([t.text.lower() for t in nlp(question + " " + " ".join(answer_concepts))])
  else:
    question_text = " ".join([t.text.lower() for t in nlp(question)])
  tsv += "%s\t%s\n"%(question_text, answer_list_str)
with open(output_name, "w") as f:
  f.write(tsv)
