import json
import sys 
from tqdm import tqdm
prediction_file = sys.argv[1]
output_file = sys.argv[2]



outputs = []
with open(prediction_file) as f:
  print("Reading", f.name)
  lines = f.read().splitlines()
  
for line in tqdm(lines[1:], desc="Processing %s"%f.name):
  instance = json.loads(line)
  qid = instance["qas_id"]
  pred = instance["predictions"]
  concept_predictions = pred["top_5000_predictions"]
  predictions_K = {100: concept_predictions}  # TODO: add more
  output = dict(qid = qid, \
                question = pred["question"], \
                predictions_K = predictions_K
                )
  outputs.append(output)

with open(output_file, "w") as f:
  print("Writing", f.name)
  for output in outputs:
    f.write(json.dumps(output) + "\n")

    