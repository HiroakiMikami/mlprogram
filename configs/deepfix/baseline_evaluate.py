imports = ["baseline_base.py"]
output_dir = "output/output"
device = torch.device(
    type_str="cuda",
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
    n_process=params.n_evaluate_process,
)
