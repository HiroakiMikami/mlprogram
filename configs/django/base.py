parser = mlprogram.datasets.django.Parser(
    split_value=mlprogram.datasets.django.SplitValue(),
)
extract_reference = mlprogram.datasets.django.TokenizeQuery()
is_subtype = mlprogram.languages.python.IsSubtype()
dataset = mlprogram.datasets.django.download()
metrics = {
    "accuracy": mlprogram.metrics.use_environment(
        metric=mlprogram.metrics.Accuracy(),
        in_keys=[["ground_truth", "expected"], "actual"],
        value_key="actual",
    ),
    "bleu": mlprogram.metrics.use_environment(
        metric=mlprogram.languages.python.metrics.Bleu(),
        in_keys=[["ground_truth", "expected"], "actual"],
        value_key="actual",
    ),
}
