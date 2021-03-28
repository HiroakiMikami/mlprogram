import mlprogram
import mlprogram.datasets.nl2bash as nl2bash
import mlprogram.languages.bash as bash

parser = bash.Parser(split_value=nl2bash.SplitValue())
extract_reference = nl2bash.TokenizeQuery()
is_subtype = bash.IsSubtype()
dataset = nl2bash.download()
metrics = {
    "accuracy": mlprogram.metrics.use_environment(
        metric=mlprogram.metrics.Accuracy(),
        in_keys=[["ground_truth", "expected"], "actual"],
        value_key="actual",
    ),
    "bleu": mlprogram.metrics.use_environment(
        metric=mlprogram.metrics.Bleu(),
        in_keys=[["ground_truth", "expected"], "actual"],
        value_key="actual",
    ),
}
