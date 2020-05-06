import gin
import argparse
from typing import Callable

import gin.torch.external_configurables  # noqa
import nl2prog.gin.external_configurables  # noqa


@gin.configurable
def entrypoint(task: Callable[[], None]) -> None:
    task()


def launch(config_file: str) -> None:
    gin.parse_config_file(config_file)
    entrypoint()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-c", required=True, type=str)
    args = parser.parse_args()
    launch(args.config_file)


if __name__ == "__main__":
    main()
