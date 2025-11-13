import pandas as pd
from vllm import LLM

from functions import get_sampling_params, read_config, read_dataset, create_prompt, load_model, save_results, set_determinism
from response_model import ResponseModel

def inference(config, llm: LLM, prompts: list) -> list:
    logits = []

    outputs = llm.generate(prompts, get_sampling_params(config))
    for output in outputs:
        try:
            response = ResponseModel.model_validate_json(output.outputs[0].text)
        except Exception:
            response = "error"
        logits.append(str(response).lower())
    return logits


if __name__ == "__main__":
    set_determinism(seed=42)

    config = read_config()

    dataset = read_dataset(config["dataset"]["path"])
    data_column = config["dataset"]["data_column"]
    label_column = config["dataset"]["label_column"]

    # Sort columns with larger string length.
    dataset.sort_values(
        config["dataset"]["data_column"], 
        ascending=False, 
        inplace=True,
        key=lambda x: x.str.len()
    )
    targets = dataset[label_column].tolist()
    sem_types = pd.read_csv("sem_types.csv")["label"].tolist()
    
    model_config = read_config(f"model_config/{config['model']}.yaml")
    prompts = [
        create_prompt(config["model"], col, config["input_text_max_len"], ", ".join(sem_types), len(sem_types)) 
        for col in dataset["column_data"]
    ]
    llm = load_model(config, model_config)
    preds = inference(config, llm, prompts)
    save_results(model_config, preds, targets)
