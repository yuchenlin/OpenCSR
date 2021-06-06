"""Converts the gkb json to tsv."""
import sys
import json
from tqdm import tqdm

filename = sys.argv[1]
output_name = filename.replace(".jsonl", ".tsv")

print(filename, output_name)

with open(filename) as f:
  lines = f.read().split("\n")
instances = [json.loads(l) for l in lines if l]
tsv = "id\ttext\ttitle\n"
for ind, ins in tqdm(enumerate(instances), total=len(instances)):  
  kid = ins["sent_id"]
  assert kid.endswith(str(ind))
  text = ins["sentence"]
  title = ins["remark"]["title"]
  tsv += "%d\t%s\t%s\n"%(ind, text, title)
with open(output_name, "w") as f:
  f.write(tsv)
