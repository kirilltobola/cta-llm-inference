from vllm import LLM

from functions import get_sampling_params, read_config, read_dataset, create_prompt, load_model, save_results
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
    
    prompts = [create_prompt(col, max_len=config["input_text_max_len"]) for col in dataset["column_data"]]
    llm = load_model(config)

    targets = dataset[label_column].tolist()
    preds = inference(config, llm, prompts)
    save_results(config, preds, targets)
