import yaml
import pandas as pd

from datetime import datetime

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from datasets import load_dataset

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


def save_results(model_config, suffix: str, logits: list, labels: list):
    model_short_name = model_config["model_short_name"]
    infernece_result = pd.DataFrame({
        "logits": logits,
        "labels": labels
    })

    infernece_result.to_csv(
        f"results/{model_short_name}-{suffix}-{datetime.now():%Y-%m-%d %H:%M:%S}.csv",
        sep="|",
        index=False
    )
    print("> Job done >>> Exiting...")
