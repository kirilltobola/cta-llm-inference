from functions import *


def inference(model, labels):
    logits = []
    targets = []

    for batch in dataloader:
        data = batch["data"]
        labels = batch["label"]
        for i in data:
            prompt = create_prompt(i)
            label = get_label(model, prompt, labels)
            logits.append(str(label))
        targets.extend(labels)
    assert len(logits) == len(targets)
    return logits, targets


if __name__ == "__main__":
    config = read_config()

    dataset, dataloader = get_dataset_dataloader(
        config["dataset"]["path"],
        config["dataloader"]["batch_size"]
    )

    logits, labels = inference(load_model(config), read_labels(config))
    save_results(config, logits, labels)
