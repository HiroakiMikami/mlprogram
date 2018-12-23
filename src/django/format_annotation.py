import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument(
    '--annotation-file',
    type=str,
    default=os.path.join("dataset", "raw", "django", "all.anno"))
parser.add_argument(
    '--destination',
    type=str,
    default=os.path.join("dataset", "raw", "django", "all_formatted.anno"))

args = parser.parse_args()

with open(args.annotation_file) as f:
    annotations = f.readlines()

annots = []
for annotation in annotations:
    m = re.search(r"^.*\.   ", annotation)
    if m is None or len(annots) == 0:
        annots.append(annotation.strip())
    else:
        annots[-1] = "{} {}".format(annots[-1], m.group(0)).strip()
        annots.append(annotation[m.end():].strip())
annotations = list(filter(lambda x: len(x) > 0, annots))

with open(args.destination, mode='w') as f:
    f.write("\n".join(annotations))
