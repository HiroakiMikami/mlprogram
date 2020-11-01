import argparse
import json
import os
import pstats
from contextlib import redirect_stdout

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--cprofile", type=str, required=True)
parser.add_argument("--torchprofile", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

stats = pstats.Stats(args.cprofile)
prof = torch.load(args.torchprofile)

os.makedirs(args.output, exist_ok=True)
prof.export_chrome_trace(os.path.join(args.output, "trace.json"))
with open(os.path.join(args.output, "trace.json")) as file:
    trace = json.load(file)
trace = [event for event in trace if "mlprogram" in event["name"]]
with open(os.path.join(args.output, "trace.json"), "w") as file:
    json.dump(trace, file)

with open(os.path.join(args.output, "cumtime_stats.txt"), "w") as file:
    with redirect_stdout(file):
        stats.sort_stats("cumtime")
        stats.print_stats()
with open(os.path.join(args.output, "tottime_stats.txt"), "w") as file:
    with redirect_stdout(file):
        stats.sort_stats("tottime")
        stats.print_stats()
