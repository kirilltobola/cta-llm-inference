import random
import numpy as np
import torch
import yaml
import pandas as pd

from datetime import datetime

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from datasets import load_dataset
from torcheval.metrics.functional import multiclass_f1_score

from response_model import ResponseModel


def set_determinism(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # If using CUDA

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Setting benchmark to False can also help ensure determinism


def truncate_input_to_n_words(input: str, n: int):
    input_lenght = len(input)
    cnt = 0
    space_idx = 0
    for i in range(space_idx, input_lenght):
        if input[i].isspace() or i == input_lenght-1:
            cnt += 1
            space_idx = i
        if cnt >= n:
            break
    return input[:space_idx + 1]


def read_config(path="config.yaml"):
    with open(path, mode="r") as f:
        config = yaml.safe_load(f)
    return config


def get_sampling_params(config) -> SamplingParams:
    sampling = config["sampling"]
    return SamplingParams(
        **sampling,
        structured_outputs=StructuredOutputsParams(json=ResponseModel.model_json_schema())
    )


def load_model(config, model_config) -> LLM:
    return LLM(
        model=model_config["model_name"],
        tokenizer=model_config["model_name"],
        quantization=model_config["quantization"], 
        max_model_len=model_config["max_model_len"],
        max_num_seqs=config["optim"]["max_num_seqs"], 
        cpu_offload_gb=config["optim"]["cpu_offload_gb"],
        gpu_memory_utilization=config["optim"]["gpu_memory_utilization"],
        trust_remote_code=True
    )


def read_dataset(path: str) -> pd.DataFrame:
    return pd.DataFrame(load_dataset(path, split="test"))


def read_labels(config):
    return pd.read_csv(config["labels_path"])[config["labels_column"]].tolist()


def save_results(model_config, suffix: str, logits: list, labels: list) -> str:
    model_short_name = model_config["model_short_name"]
    infernece_result = pd.DataFrame({
        "logits": logits,
        "labels": labels
    })

    filename = f"results/{model_short_name}-{suffix}-{datetime.now():%Y-%m-%d %H:%M:%S}.csv"
    infernece_result.to_csv(
        filename,
        sep="|",
        index=False
    )
    return filename


def get_inference_score(filename: str, labels_filename: str, labels_column: str, num_tests: int, average_list: list[str]) -> None:
    sem_types = pd.read_csv(labels_filename)[labels_column].tolist()
    sem_types_dict = {}
    for i in range(len(sem_types)):
        sem_types_dict[sem_types[i]] = i
    sem_types_dict

    df = pd.read_csv(filename, sep="|")
    preds = df["logits"].tolist()
    labels = df["labels"].tolist()

    tests = np.array([0 for _ in range(num_tests)], dtype=float)
    for average in average_list:
        for i in range(num_tests):
            preds_map = list(map(lambda x: sem_types_dict.get(x, random.randint(0, len(sem_types) - 1)), preds))
            preds_tensor = torch.tensor(preds_map, dtype=torch.int64)
            labels_map = list(map(lambda x: sem_types_dict[x], labels))
            labels_tensor = torch.tensor(labels_map, dtype=torch.int64)

            score = multiclass_f1_score(
                preds_tensor,
                labels_tensor,
                num_classes=len(sem_types),
                average=average
            )
            tests[i] = score.numpy()
        print(f"<< \n < Inference score ({average}): min = {tests.min():.4f} / mean = {tests.mean():.4f} / max = {tests.max():.4f}")
