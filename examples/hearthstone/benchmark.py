import mlprogram
from mlprogram.launch import global_options

global_options["timeout_sec"] = 10


def setup_synthesizer(synthesizer):
    return mlprogram.synthesizers.SynthesizerWithTimeout(
        synthesizer=synthesizer,
        timeout_sec=global_options.timeout_sec,
    )
