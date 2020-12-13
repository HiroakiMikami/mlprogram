parser = mlprogram.languages.python.Parser(
    split_value=mlprogram.datasets.hearthstone.SplitValue(),
)
extract_reference = mlprogram.datasets.hearthstone.TokenizeQuery()
dataset = mlprogram.datasets.hearthstone.download()
is_subtype = mlprogram.languages.python.IsSubtype()
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
