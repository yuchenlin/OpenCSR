import sys
import json
rows = []
cols = []
vals = []

with open(sys.argv[1]) as f: 
  for line in f.read().splitlines():
    if not line:
      continue
    f1, f2 = line.split()
    rows.append(f1)
    cols.append(f2)
    vals.append(1)

with open(sys.argv[2], "w") as f:
  json.dump(dict(rows=rows, cols=cols, vals=vals), f)