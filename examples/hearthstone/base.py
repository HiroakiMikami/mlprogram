import mlprogram
import mlprogram.datasets.hearthstone as hearthstone
import mlprogram.languages.python as python

parser = python.Parser(split_value=hearthstone.SplitValue())
extract_reference = hearthstone.TokenizeQuery()
dataset = hearthstone.download()
is_subtype = python.IsSubtype()
metrics = {
    "accuracy": mlprogram.metrics.use_environment(
        metric=mlprogram.metrics.Accuracy(),
        in_keys=[["ground_truth", "expected"], "actual"],
        value_key="actual",
    ),
    "bleu": mlprogram.metrics.use_environment(
        metric=python.metrics.Bleu(),
        in_keys=[["ground_truth", "expected"], "actual"],
        value_key="actual",
    ),
}
