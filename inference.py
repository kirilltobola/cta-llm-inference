import pandas as pd
from vllm import LLM

from functions import get_inference_score, get_sampling_params, read_config, read_dataset, load_model, save_results, set_determinism, truncate_input_to_n_words
from prompts.prompt_renderer import PromptRenderer
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
    model_config = read_config(f"model_config/{config['model']}.yaml")

    set_determinism(seed=config["seed"])

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
    
    prompt_filename = config["prompt"]
    renderer = PromptRenderer(
        prompt_filename,
        config["model"],
        model_config["render_rules"]
    )
    renderer.pre_render(pd.read_csv("sem_types.csv")["label"].tolist())
    prompts = [
        renderer.render(truncate_input_to_n_words(input, n=config["input_text_max_len"]))
        for input in dataset["column_data"]
    ]

    llm = load_model(config, model_config)
    preds = inference(config, llm, prompts)
    suffix = prompt_filename.split("/")[-1]
    inference_filename = save_results(model_config, suffix, preds, targets)

    get_inference_score(inference_filename, config["labels_path"], config["labels_column"], num_tests=100)
