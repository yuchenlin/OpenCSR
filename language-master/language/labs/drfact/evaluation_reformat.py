import json
import sys 
from tqdm import tqdm
prediction_file = sys.argv[1]
meta_data_file = sys.argv[2]
output_file = sys.argv[3]

ans = dict()
dis = dict()
with open(meta_data_file, "r") as f:
  print("Reading", f.name)
  lines = f.read().splitlines()
  for line in lines:
    if line:
      item = json.loads(line)
      ans[item["qid"]] = item["answers"]
      # dis[item["qid"]] = item["distractors"]

outputs = []
with open(prediction_file) as f:
  print("Reading", f.name)
  lines = f.read().splitlines()
  assert len(lines)-1 == len(ans)
for line in tqdm(lines[1:], desc="Processing %s"%f.name):
  instance = json.loads(line)
  qid = instance["qas_id"]
  assert qid in ans
  pred = instance["predictions"]
  concept_predictions = pred["top_5000_predictions"]
  predictions_K = {100: concept_predictions}
  output = dict(qid = qid, \
                question = pred["question"], \
                predictions_K = predictions_K, \
                answers = ans[qid], \
                # distractors = dis[qid]
                )
  outputs.append(output)

with open(output_file, "w") as f:
  print("Writing", f.name)
  for output in outputs:
    f.write(json.dumps(output) + "\n")

    