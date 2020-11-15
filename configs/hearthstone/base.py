parser = mlprogram.languages.python.Parser(
    split_value=mlprogram.datasets.hearthstone.SplitValue(),
)
extract_reference = mlprogram.datasets.hearthstone.TokenizeQuery()
dataset = mlprogram.datasets.hearthstone.download()
is_subtype = mlprogram.languages.python.IsSubtype()
metrics = {
    "accuracy": mlprogram.metrics.Accuracy(),
    "bleu": mlprogram.languages.python.metrics.Bleu(),
}
