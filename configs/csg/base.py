dataset_option = {
    "train_max_object": 13,
    "evaluate_max_object": 30,
    "size": 16,
    "resolution": 1,
    "n_evaluate_dataset": 30,
}

parser = mlprogram.languages.csg.Parser()
n_pixel = mul(x=dataset_option.size, y=dataset_option.resolution)
n_feature_pixel = intdiv(lhs=n_pixel, rhs=2)
dataset = mlprogram.languages.csg.Dataset(
    canvas_size=dataset_option.size,
    min_object=1,
    max_object=dataset_option.train_max_object,
    length_stride=1,
    degree_stride=15,
)
encoder = with_file_cache(
    path=os.path.join(
        args=[output_dir, "encoder.pt"],
    ),
    config=mlprogram.encoders.ActionSequenceEncoder(
        samples=mlprogram.languages.csg.get_samples(
            dataset=dataset, parser=parser, reference=reference
        ),
        token_threshold=0,
    ),
)
train_dataset = mlprogram.utils.data.transform(
    dataset=dataset,
    transform=Apply(
        module=mlprogram.languages.csg.transforms.AddTestCases(
            interpreter=interpreter,
        ),
        in_keys=["ground_truth"],
        out_key="test_cases",
        is_out_supervision=False,
    ),
)
test_dataset = mlprogram.utils.data.transform(
    dataset=mlprogram.utils.data.to_map_style_dataset(
        dataset=mlprogram.languages.csg.Dataset(
            canvas_size=dataset_option.size,
            min_object=dataset_option.train_max_object,
            max_object=dataset_option.evaluate_max_object,
            length_stride=1,
            degree_stride=15,
            seed=10000,
        ),
        n=dataset_option.n_evaluate_dataset,
    ),
    transform=Apply(
        module=mlprogram.languages.csg.transforms.AddTestCases(
            interpreter=interpreter,
        ),
        in_keys=["ground_truth"],
        out_key="test_cases",
        is_out_supervision=False,
    ),
)
valid_dataset = mlprogram.utils.data.transform(
    dataset=mlprogram.utils.data.to_map_style_dataset(
        dataset=mlprogram.languages.csg.Dataset(
            canvas_size=dataset_option.size,
            min_object=dataset_option.train_max_object,
            max_object=dataset_option.evaluate_max_object,
            length_stride=1,
            degree_stride=15,
            seed=20000,
        ),
        n=dataset_option.n_evaluate_dataset,
    ),
    transform=Apply(
        module=mlprogram.languages.csg.transforms.AddTestCases(
            interpreter=interpreter,
        ),
        in_keys=["ground_truth"],
        out_key="test_cases",
        is_out_supervision=False,
    ),
)
interpreter = mlprogram.languages.csg.Interpreter(
    width=dataset_option.size,
    height=dataset_option.size,
    resolution=dataset_option.resolution,
    delete_used_reference=True,
)
