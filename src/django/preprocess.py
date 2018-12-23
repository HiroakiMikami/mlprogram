import argparse
import os
import re
import json
from nltk import tokenize

parser = argparse.ArgumentParser()
parser.add_argument('--ids', type=str, nargs='+')
parser.add_argument(
    '--directory',
    type=str,
    default=os.path.join("dataset", "django", "train"))

args = parser.parse_args()

tokenizer = tokenize.WhitespaceTokenizer()

for id in args.ids:
    # Read the annotation file
    with open(os.path.join(args.directory, "{}.anno".format(id))) as f:
        annotation = "\n".join(f.readlines())

    # Create the backup of the original file
    with open(
            os.path.join(args.directory, "{}.anno.orig".format(id)),
            mode='w') as f:
        f.write(annotation)

    # Preprocess annotation
    def placeholder(id):
        return "####{}####".format(id)

    # Replace quoted string literals with placeholders
    mappings = {}
    literal = r'\'\\\'\'|\"[^\"]+\"|\'[^\']+\'|`[^`]+`|"""[^"]+"""'
    while True:
        m = re.search(literal, annotation)
        if m is not None:
            p = placeholder(len(mappings))
            annotation = annotation[:m.start()] + p + annotation[m.end():]
            w = m.group(0)[1:len(m.group(0)) - 1]
            assert (not ("####" in w))
            mappings[p] = str(w)
        else:
            break

    words = tokenizer.tokenize(annotation)
    preprocessed_words = []
    for word in words:
        preprocessed_words.append(word)
        vars = list(filter(lambda x: len(x) > 0,
                           word.split('.')))  # split by '.'
        if len(vars) > 1:
            for v in vars:
                preprocessed_words.append(v)
            continue
    with open(
            os.path.join(args.directory, "{}.anno".format(id)), mode='w') as f:
        f.write(" ".join(preprocessed_words))
    with open(
            os.path.join(args.directory, "{}.mapping.json".format(id)),
            mode='w') as f:
        json.dump(dict(mappings), f)
