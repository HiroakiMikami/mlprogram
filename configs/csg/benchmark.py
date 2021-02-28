benchmark_timeout = {
    "short": 30,
    "long": 360,
}

synthesizer = mlprogram.synthesizers.FilteredSynthesizer(
    synthesizer=mlprogram.synthesizers.SynthesizerWithTimeout(
        synthesizer=base_synthesizer,
        timeout_sec=select(options=benchmark_timeout, key=benchmark),
    ),
    score=mlprogram.metrics.use_environment(
        metric=mlprogram.metrics.TestCaseResult(
            interpreter=interpreter,
            metric=mlprogram.metrics.use_environment(
                metric=mlprogram.metrics.Iou(),
                in_keys=["expected", "actual"],
                value_key="actual",
            ),
        ),
        in_keys=["test_cases", "actual"],
        value_key="actual",
    ),
    threshold=0.9,
)
