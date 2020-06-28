import argparse

from mlprogram.entrypoint.parse import parse_config_file


def launch(config_file: str):
    configs = parse_config_file(config_file)
    configs["/main"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-c", required=True, type=str)
    args = parser.parse_args()
    launch(args.config_file)


if __name__ == "__main__":
    main()
