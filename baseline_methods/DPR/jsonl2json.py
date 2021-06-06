import json
import sys 

assert sys.argv[1].endswith(".jsonl")
data = []
with open(sys.argv[1]) as f:
  lines = f.read().split("\n")
  for line in lines:
    if line:
      data.append(json.loads(line))
with open(sys.argv[1].replace(".jsonl", ".json"), "w")  as f:
  json.dump(data, f)