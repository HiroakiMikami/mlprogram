import argparse
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--train', type=int, default=16000)
parser.add_argument('--test', type=int, default=1000)
parser.add_argument('--valid', type=int, default=1805)
parser.add_argument(
    '--annotation-file',
    type=str,
    default=os.path.join("dataset", "raw", "django", "all_formatted.anno"))
parser.add_argument(
    '--code-file',
    type=str,
    default=os.path.join("dataset", "raw", "django", "all.code"))
parser.add_argument(
    '--destination', type=str, default=os.path.join("dataset", "django"))

args = parser.parse_args()

# Open annotation and code file
with open(args.annotation_file) as f:
    annotations = f.readlines()
with open(args.code_file) as f:
    code = f.readlines()

assert (len(annotations) == len(code))
num_dataset = min(len(annotations), len(code))
assert (args.train + args.test + args.valid == num_dataset)

order = list(range(0, num_dataset))
if args.seed >= 0:
    random.seed(args.seed)
    random.shuffle(order)

# Split dataset into train/test/valid
os.makedirs(os.path.join(args.destination, "train"))
os.makedirs(os.path.join(args.destination, "test"))
os.makedirs(os.path.join(args.destination, "valid"))


def create(i, dir):
    with open(os.path.join(dir, "{}.anno".format(i)), mode='w') as f:
        f.write(annotations[i])
    with open(os.path.join(dir, "{}.code".format(i)), mode='w') as f:
        f.write(code[i])


for i in order[0:args.train]:
    create(i, os.path.join(args.destination, "train"))
for i in order[args.train:args.train + args.test]:
    create(i, os.path.join(args.destination, "test"))
for i in order[args.train + args.test:]:
    create(i, os.path.join(args.destination, "valid"))
