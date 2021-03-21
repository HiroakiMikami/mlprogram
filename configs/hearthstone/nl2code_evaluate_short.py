imports = ["nl2code_base.py", "benchmark.py"]
benchmark = "short"
device = torch.device(type_str="cpu", index=0)
main = mlprogram.entrypoint.evaluate(
    input_dir=train_artifact_dir,
    output_dir=evaluate_artifact_dir,
    valid_dataset=valid_dataset,
    model=model,
    synthesizer=synthesizer,
    metrics=metrics,
    top_n=params.metric_top_n,
    device=device,
)
