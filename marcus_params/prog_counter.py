import json
import sys

fn =sys.argv[-1]

f=open(fn,"r")
data=json.loads(f.read())
f.close()

print(len(data))
