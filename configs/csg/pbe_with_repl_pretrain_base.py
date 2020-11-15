imports = ["pbe_with_repl_base.py"]
device = torch.device(
    type_str="cuda",
    index=0,
)
batch_size = 32
output_dir = "output/output"
optimizer = torch.optim.Optimizer(
    optimizer_cls=torch.optim.Adam(),
    model=model,
)
loss_fn = torch.nn.Sequential(
    modules=collections.OrderedDict(
        items=[
            [
                "loss",
                mlprogram.nn.action_sequence.Loss(
                    reduction="sum",
                ),
            ],
            [
                "normalize",
                mlprogram.nn.Apply(
                    in_keys=[["output@action_sequence_loss", "lhs"]],
                    out_key="output@action_sequence_loss",
                    module=mlprogram.nn.Function(
                        f=Div(),
                    ),
                    constants={"rhs": batch_size},
                ),
            ],
            [
                "pick",
                mlprogram.nn.Function(
                    f=Pick(
                        key="output@action_sequence_loss",
                    ),
                ),
            ],
        ],
    ),
)
main = mlprogram.entrypoint.train_supervised(
    workspace_dir="output/workspace_pretraining",
    output_dir=output_dir,
    dataset=train_dataset,
    model=model,
    optimizer=optimizer,
    loss=loss_fn,
    evaluate=None,
    metric="none",
    collate=collate_fn,
    batch_size=batch_size,
    length=mlprogram.entrypoint.train.Iteration(
        n=option.n_pretrain_iteration,
    ),
    evaluation_interval=mlprogram.entrypoint.train.Iteration(
        n=1000,
    ),
    snapshot_interval=mlprogram.entrypoint.train.Iteration(
        n=1000,
    ),
    device=device,
)
