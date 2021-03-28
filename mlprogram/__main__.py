import argparse
import tempfile

from mlprogram.launch.launch import launch_multiprocess


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--file", required=True, type=str)
    parser.add_argument("--option", choices=["test", "profile"])
    parser.add_argument("--n_process", type=int, default=0)
    args, argv = parser.parse_known_args()
    with tempfile.TemporaryDirectory() as tmpdir:
        launch_multiprocess(
            args.file,
            args.option,
            tmpdir,
            args.n_process,
            argv
        )


if __name__ == "__main__":
    main()
