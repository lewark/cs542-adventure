import json
import sys

import matplotlib.pyplot as plt


filename = sys.argv[1]
out_filename = sys.argv[2]
title = sys.argv[3]

loss = []

with open(filename, "r") as in_file:
    for line in in_file:
        doc = json.loads(line.replace("'", '"'))
        loss.append(doc["loss"])

plt.plot(loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(title)
plt.savefig(out_filename)
