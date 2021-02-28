benchmark_timeout = {
    "short": 15,
    "long": 180,
}

synthesizer = mlprogram.synthesizers.SynthesizerWithTimeout(
    synthesizer=base_synthesizer,
    timeout_sec=select(options=benchmark_timeout, key=benchmark),
)
