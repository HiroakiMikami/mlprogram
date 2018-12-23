import ast
import argparse
import json
import string
import os
import numpy as np
from src.python.grammar import to_sequence

parser = argparse.ArgumentParser()
parser.add_argument('--ids', type=str, nargs='+')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--train-num', type=int, default=100)
parser.add_argument('--test-num', type=int, default=10)
parser.add_argument('--valid-num', type=int, default=10)
parser.add_argument(
    '--directory', type=str, default=os.path.join("dataset", "toy"))

args = parser.parse_args()
train_dir = os.path.join(args.directory, "train")
test_dir = os.path.join(args.directory, "test")
valid_dir = os.path.join(args.directory, "valid")
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
if not os.path.exists(valid_dir):
    os.makedirs(valid_dir)

rng = np.random.RandomState(args.seed)


def create_dataset(dir, num):
    for id in range(num):
        var0 = ''.join(rng.choice(list(string.ascii_lowercase), size=5))
        var1 = ''.join(rng.choice(list(string.ascii_lowercase), size=5))
        if rng.randint(2) == 0:
            # return
            annotation = "return ####0####"
            code = "return {}".format(var0)
            mapping = {"####0####": var0}
        else:
            # assign
            annotation = "assign ####0#### to ####1####"
            code = "{} = {}".format(var1, var0)
            mapping = {"####0####": var0, "####1####": var1}

        # Create train dataset
        with open(os.path.join(dir, "{}.code".format(id)), mode='w') as f:
            f.write(code)
        with open(os.path.join(dir, "{}.anno".format(id)), mode='w') as f:
            f.write(annotation)
        with open(
                os.path.join(dir, "{}.mapping.json".format(id)),
                mode='w') as f:
            json.dump(mapping, f)
        with open(
                os.path.join(dir, "{}.reference_seq.json".format(id)),
                mode='w') as f:
            node = ast.parse(code).body[0]
            sequence = to_sequence(node)
            json.dump(sequence, f)


create_dataset(train_dir, args.train_num)
create_dataset(test_dir, args.test_num)
create_dataset(valid_dir, args.valid_num)
