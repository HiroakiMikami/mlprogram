import torch
import json
from torch.utils.data import Dataset, DataLoader
from typing \
    import Callable, Any, Optional, Tuple, cast, Iterable, List, Mapping, Set
import os
import logging
import shutil
from math import ceil
from mlprogram.gin import workspace
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask
from mlprogram.metrics import Metric
from mlprogram.utils import evaluate as eval, TopKModel
from mlprogram.utils.data import to_eval_dataset, ListDataset
from mlprogram.synthesizers import synthesize, BeamSearchSynthesizer


logger = logging.getLogger(__name__)


def train(dataset_key: str, model_key: str, optimizer_key: str,
          encoder_keys: Set[str],
          workspace_dir: str, output_dir: str,
          prepare_dataset: Callable[[str], None],
          prepare_encoder: Callable[[], None],
          prepare_model: Callable[[str], None],
          prepare_optimizer: Callable[[str], None],
          transform_cls: Callable[[], Callable[[Dataset], Dataset]],
          loss_fn: Callable[[PaddedSequenceWithMask, PaddedSequenceWithMask,
                             PaddedSequenceWithMask, PaddedSequenceWithMask],
                            torch.Tensor],
          score_fn: Callable[[PaddedSequenceWithMask, PaddedSequenceWithMask,
                              PaddedSequenceWithMask, PaddedSequenceWithMask],
                             torch.Tensor],
          collate_fn: Callable[[Any], Any],
          batch_size: int, num_epochs: int,
          num_checkpoints: int = 2, num_models: int = 3,
          device: torch.device = torch.device("cpu"),
          progress_bar: Optional[Callable[[Iterable], Iterable]] = None) \
        -> None:
    with workspace.use_workspace():
        os.makedirs(workspace_dir, exist_ok=True)

        logger.info("Prepare dataset")
        prepare_dataset(dataset_key)
        raw_dataset = workspace.get(dataset_key)
        assert raw_dataset is not None
        raw_train_dataset = raw_dataset["train"]

        logger.info("Prepare encoder")
        encoder_path = os.path.join(workspace_dir, "encoder.pt")
        if os.path.exists(encoder_path):
            logger.info(f"Load encoder from {encoder_path}")
            for key, value in torch.load(encoder_path).items():
                workspace.put(key, value)
        else:
            prepare_encoder()
            logger.info("Save encoder")
            torch.save({key: workspace.get(key) for key in encoder_keys},
                       encoder_path)

        logger.info("Transform the dataset")
        dataset = transform_cls()(raw_train_dataset)

        logger.info("Prepare model")
        prepare_model(model_key)
        model = workspace.get(model_key)
        assert model is not None
        model.to(device)
        model.train()

        logger.info("Prepare optimizer")
        prepare_optimizer(optimizer_key)
        optimizer = workspace.get(optimizer_key)
        assert optimizer is not None

        # Load checkpoint
        checkpoint_dir = os.path.join(workspace_dir, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoints = [(os.path.join(checkpoint_dir, checkpoint),
                        int(checkpoint.replace(".pt", "")))
                       for checkpoint in os.listdir(checkpoint_dir)]
        checkpoints.sort(key=lambda x: x[1])
        if len(checkpoints) != 0:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1][0])
            logger.info(f"Load checkpoint from {checkpoint_path}")
            ckpt = \
                torch.load(checkpoint_path, map_location=torch.device("cpu"))
            start_epoch = ckpt["epoch"]
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
        else:
            start_epoch = 0

        # Load log
        log_path = os.path.join(workspace_dir, "log.json")
        if os.path.exists(log_path):
            logger.info(f"Load log {log_path}")
            with open(log_path, "r") as file:
                logs = json.load(file)
        else:
            logs = []

        # Prepare TopKModel
        model_dir = os.path.join(workspace_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        top_k_model = TopKModel(num_models, model_dir)

        logger.info(f"Strat training from {start_epoch} epoch")
        for epoch in range(start_epoch, ceil(num_epochs)):
            # TODO num_workers > 0 causes the RuntimeError
            loader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0,
                                collate_fn=collate_fn)
            if progress_bar is not None:
                loader = progress_bar(loader)
            if num_epochs - epoch < 1:
                n_iter = max(1, int(len(loader) * (num_epochs - epoch)))
            else:
                n_iter = -1

            avg_loss = 0.0
            avg_score = 0.0
            model.train()
            for i, batch in enumerate(loader):
                if i == n_iter:
                    break
                input = batch[:-1]
                ground_truth = batch[-1]
                output = cast(Tuple[PaddedSequenceWithMask,
                                    PaddedSequenceWithMask,
                                    PaddedSequenceWithMask],
                              model(*input))
                loss = loss_fn(*output, ground_truth)
                score = score_fn(*output, ground_truth)
                model.zero_grad()
                loss.backward()
                optimizer.step()

                avg_loss += loss.item() / len(loader)
                avg_score += score.item() / len(loader)

            logger.info(
                f"Epoch {epoch} : Loss = {avg_loss} Score = {avg_score}")
            logger.info("Save checkpoint")
            checkpoint = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            checkpoint_path = os.path.join(checkpoint_dir, f"{epoch}.pt")
            torch.save(checkpoint, checkpoint_path)
            checkpoints.append((checkpoint_path, epoch))
            while len(checkpoints) > num_checkpoints:
                path, _ = checkpoints.pop(0)
                logger.info(f"Remove {path}")
                os.remove(path)
            logger.info("Save log")
            logs.append({
                "epoch": epoch, "loss": avg_loss, "score": avg_score
            })
            with open(log_path, "w") as file:
                json.dump(logs, file)
            top_k_model.save(avg_score, f"{epoch}", model)

        logger.info("Save encoder to output_dir")
        os.makedirs(output_dir, exist_ok=True)
        encoder_path = os.path.join(output_dir, "encoder.pt")
        torch.save({key: workspace.get(key) for key in encoder_keys},
                   encoder_path)

        logger.info("Copy log to output_dir")
        shutil.copyfile(log_path, os.path.join(output_dir, "log.json"))

        logger.info("Copy models to output_dir")
        out_model_dir = os.path.join(output_dir, "model")
        if os.path.exists(out_model_dir):
            shutil.rmtree(out_model_dir)
        shutil.copytree(model_dir, out_model_dir)


def evaluate(dataset_key: str, synthesizer_key: str, encoder_keys: Set[str],
             input_dir: str, workspace_dir: str, output_dir: str,
             prepare_dataset: Callable[[str], None],
             prepare_synthesizer: Callable[[str], None],
             metrics: Mapping[str, Metric],
             main_metric: Tuple[int, str],
             top_n: List[int] = [1],
             device: torch.device = torch.device("cpu"),
             n_samples: Optional[int] = None,
             progress_bar: Optional[Callable[[Iterable], Iterable]] = None) \
        -> None:
    with workspace.use_workspace():
        os.makedirs(workspace_dir, exist_ok=True)

        logger.info("Prepare dataset")
        prepare_dataset(dataset_key)
        raw_dataset = workspace.get(dataset_key)
        assert raw_dataset is not None
        raw_test_dataset = raw_dataset["test"]
        raw_valid_dataset = raw_dataset["valid"]
        if n_samples is not None:
            raw_test_dataset = ListDataset(raw_test_dataset[:n_samples])
            raw_valid_dataset = ListDataset(raw_valid_dataset[:n_samples])
        test_dataset = to_eval_dataset(raw_test_dataset)
        valid_dataset = to_eval_dataset(raw_valid_dataset)

        encoder_path = os.path.join(input_dir, "encoder.pt")
        logger.info(f"Load encoder from {encoder_path}")
        for key, value in torch.load(encoder_path).items():
            workspace.put(key, value)

        logger.info("Prepare synthesizer")
        prepare_synthesizer(synthesizer_key)
        synthesizer = cast(BeamSearchSynthesizer,
                           workspace.get(synthesizer_key))
        assert synthesizer is not None

        model_dir = os.path.join(input_dir, "model")
        results_path = os.path.join(workspace_dir, "results.pt")
        if os.path.exists(results_path):
            logger.info(f"Load results from {results_path}")
            results = torch.load(results_path)
        else:
            results = {"test": {}}
        for name in os.listdir(model_dir):
            if name in results:
                continue
            path = os.path.join(model_dir, name)
            state_dict = \
                torch.load(path, map_location=torch.device("cpu"))["model"]
            logger.info(f"Start evaluation (test dataset): {name}")
            synthesizer.load_state_dict(state_dict)

            test_data = test_dataset
            if progress_bar is not None:
                test_data = progress_bar(test_data)

            result = eval(test_data,
                          lambda query: synthesize(query, synthesizer),
                          metrics=metrics, top_n=top_n)
            logger.info(f"{name}: {result.metrics}")
            results["test"][name] = result
            torch.save(results, results_path)

        logger.info("Find best model")
        best_model: Optional[str] = None
        best_score: float = 0.0
        for name, result in results["test"].items():
            m = result.metrics[main_metric[0]][main_metric[1]]
            if best_score < m:
                best_model = name
                best_score = m

        if best_model is not None:
            logger.info(f"Start evaluation (valid dataset): {best_model}")
            path = os.path.join(model_dir, best_model)
            state_dict = \
                torch.load(path, map_location=torch.device("cpu"))["model"]
            synthesizer.load_state_dict(state_dict)

            test_data = valid_dataset
            if progress_bar is not None:
                test_data = progress_bar(test_data)

            result = eval(test_data,
                          lambda query: synthesize(query, synthesizer),
                          metrics=metrics, top_n=top_n)
            logger.info(f"{name}: {result.metrics}")
            results["best_model"] = best_model
            results["valid"] = result
            torch.save(results, results_path)

        logger.info("Copy log to output_dir")
        os.makedirs(output_dir, exist_ok=True)
        shutil.copyfile(results_path, os.path.join(output_dir, "results.pt"))
