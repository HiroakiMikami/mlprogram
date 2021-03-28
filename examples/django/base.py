import mlprogram
import mlprogram.datasets.django as django
import mlprogram.languages.python as python

parser = django.Parser(split_value=django.SplitValue())
extract_reference = django.TokenizeQuery()
is_subtype = python.IsSubtype()
dataset = django.download()
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
