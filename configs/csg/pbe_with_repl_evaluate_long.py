imports = ["pbe_with_repl_base.py", "benchmark.py"]
benchmark = "long"
output_dir = "output/output"
device = torch.device(
    type_str="cpu",
    index=0,
)
main = mlprogram.entrypoint.evaluate(
    workspace_dir="output/workspace",
    input_dir=output_dir,
    output_dir=output_dir,
    valid_dataset=valid_dataset,
    model=model,
    synthesizer=synthesizer,
    metrics={},
    top_n=[],
    device=device,
)
