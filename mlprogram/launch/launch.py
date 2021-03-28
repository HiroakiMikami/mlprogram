import cProfile
import importlib
import logging as L
import os
import random
from typing import List, Optional

import numpy as np
import torch
from torch import multiprocessing
from torch.autograd.profiler import profile

from mlprogram import distributed, logging
from mlprogram.launch.options import Options, global_options
from mlprogram.tasks.train import Epoch, Iteration

logger = logging.Logger(__name__)


def setup_test_options(tmpdir: str, options: Options):
    for key, _type in options.options.items():
        if key == "device_type":
            options.overwrite_option(key, "cpu")
        elif key.endswith("_artifact_dir"):
            name = key.replace("_artifact_dir", "")
            value = f"{tmpdir}/{name}"
            options.overwrite_option(key, value)
        elif _type in set(Epoch, Iteration):
            options.overwrite_option(key, Iteration(2))
        elif key == "n_validate_sample":
            options.overwrite_option(key, 1)
        elif key == "n_test_sample":
            options.overwrite_option(key, 1)
        elif key == "max_step_size":
            options.overwrite_option(key, 2)
        elif key == "timeout_sec":
            options.overwrite_option(key, 0.5)


def setup_profile_options(tmpdir: str, options: Options):
    for key, _type in options.options.items():
        if key == "device_type":
            options.overwrite_option(key, "cpu")
        elif key.endswith("_artifact_dir"):
            name = key.replace("_artifact_dir", "")
            value = f"{tmpdir}/{name}"
            options.overwrite_option(key, value)
        elif _type in set(Epoch, Iteration):
            options.overwrite_option(key, Iteration(2))
        elif key == "n_validate_sample":
            options.overwrite_option(key, 1)
        elif key == "n_test_sample":
            options.overwrite_option(key, 0)
        elif key == "max_step_size":
            options.overwrite_option(key, 2)
        elif key == "timeout_sec":
            options.overwrite_option(key, 2)


def launch(
    file: str,
    option: Optional[str],
    tmpdir: str,
    rank: Optional[int],
    n_process: Optional[int],
    args: List[str],
):
    logging.set_level(L.INFO)

    logger.info(f"Load file {file}")
    spec = importlib.util.spec_from_file_location("module.name", file)
    main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main)

    if option == "test":
        logger.info("Setup options for testing")
        setup_test_options(tmpdir, global_options)
    elif option == "profile":
        logger.info("Setup options for profiling")
        setup_profile_options(tmpdir, global_options)
    logger.info("Setup options from command line arguments")
    global_options.overwrite_options_by_args(args)

    distributed.initialize(tmpdir, rank, n_process)

    rank = distributed.rank()
    seed = random.randint(0, 2**31) + rank
    logger.info(f"Fix seed={seed}")
    rng = np.random.RandomState(seed)
    torch.manual_seed(rng.randint(0, 2**32 - 1))
    np.random.seed(rng.randint(0, 2**32 - 1))
    random.seed(rng.randint(0, 2**32 - 1))

    logger.info("Run main")
    if option == "profile":
        cprofile = cProfile.Profile()
        cprofile.enable()
        with profile() as torch_prof:
            main.main()
        cprofile.disable()
        torch.save(torch_prof, os.path.join("profile", f"torch_profiler-{rank}.pt"))
        cprofile.dump_stats(os.path.join("profile", f"cprofile-{rank}.pt"))
    else:
        main.main()


def launch_multiprocess(
    config_file: str,
    option: Optional[str],
    tmpdir: str,
    n_process: int,
    args: List[str],
):
    if n_process == 0:
        return launch(config_file, option, tmpdir, None, None, args)
    else:
        ps = []
        for i in range(n_process):
            p = multiprocessing.Process(
                target=launch,
                args=(config_file, option, tmpdir, i, n_process, args))
            p.start()
            ps.append(p)
        for p in ps:
            p.join()
