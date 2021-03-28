import examples.hearthstone.baseline_base as base
from examples.nl2prog_lstm.evaluate import run
from mlprogram.launch import global_options

run(
    dataset=base.base.dataset,
    parser=base.base.parser,
    extract_reference=base.base.extract_reference,
    is_subtype=base.base.is_subtype,
    metric_top_n=global_options.metric_top_n,
    metrics=base.base.metrics,
    setup_synthesizer=lambda x: x,
)
