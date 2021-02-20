eval_params = select(
    key=benchmark_size,
    options={
        "small": {
            "timeout_sec": 15,
        },
        "large": {
            "timeout_sec": 180,
        },
    },
)

synthesizer = mlprogram.synthesizers.SynthesizerWithTimeout(
    synthesizer=_synthesizer,
    timeout_sec=eval_params.timeout_sec,
)
