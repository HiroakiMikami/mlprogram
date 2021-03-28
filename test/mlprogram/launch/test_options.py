from mlprogram.launch.options import Options


def test_define_option():
    opt = Options()
    opt["foo"] = 10
    opt["bar"] = "bar"
    assert opt.foo == 10
    assert opt.bar == "bar"
    assert opt._values == {"foo": 10, "bar": "bar"}


def test_options():
    opt = Options()
    opt["foo"] = 10
    opt["bar"] = "bar"
    assert opt.options == {"foo": int, "bar": str}


def test_overwrite_options_by_args():
    opt = Options()
    opt["foo"] = 10
    opt["bar"] = "bar"
    opt.overwrite_options_by_args(["--foo", 0])
    assert opt.foo == 0
    assert opt.bar == "bar"
