parser = mlprogram.languages.bash.Parser(
    split_value=mlprogram.datasets.nl2bash.SplitValue(),
)
extract_reference = mlprogram.datasets.nl2bash.TokenizeQuery()
is_subtype = mlprogram.languages.bash.IsSubtype()
dataset = mlprogram.datasets.nl2bash.download()
metrics = {"accuracy": mlprogram.metrics.Accuracy(), "bleu": mlprogram.metrics.Bleu()}
