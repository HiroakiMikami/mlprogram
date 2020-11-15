parser = mlprogram.datasets.django.Parser(
    split_value=mlprogram.datasets.django.SplitValue(),
)
extract_reference = mlprogram.datasets.django.TokenizeQuery()
is_subtype = mlprogram.languages.python.IsSubtype()
dataset = mlprogram.datasets.django.download()
metrics = {
    "accuracy": mlprogram.metrics.Accuracy(),
    "bleu": mlprogram.languages.python.metrics.Bleu(),
}
