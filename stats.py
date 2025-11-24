import argparse

from functions import get_inference_score, read_config


if __name__ == "__main__":
    config = read_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("inference_filename", help="Inference result filename")
    args = parser.parse_args()

    get_inference_score(
        args.inference_filename, 
        config["labels_path"], 
        config["labels_column"], 
        config["inference_score"]["n_tests"],
        config["inference_score"]["average"]
    )
