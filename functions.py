import yaml
import pandas as pd

from datetime import datetime

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from datasets import load_dataset

from response_model import ResponseModel


def create_prompt(input_text: str, max_len: int) -> str:
    input_text = input_text[:max_len]
    sys_message = """System message: Be a helpful, accurate assistant for data discovery and exploration desiged to output valid JSON.
    """
    
    usr_message = f"""User message: Consider input text is: {input_text}
    
    There are a list of russian 170 valid types for given input text: авиакомпания, автомобиль, адрес, актер, альбом, аннотация, арена, атлет, аэропорт, база, банк, боксер, борец, ввп, ведущий, вес, вещество, владелец, возраст, высота, гимнаст, глубина, год, гонка, город, группа, дата, двигатель, день, директор, документ, дорога, жанр, животное, журнал, закон, звук, здание, игра, идентификатор, идеология, издатель, изображение, инструмент, камера, канал, категория, класс, классификация, клуб, книга, код, количество, колледж, команда, компания, компонент, континент, конькобежец, корабль, кратер, лейбл, лес, лига, локомотив, марка, место, местонахождение, модель, мотоцикл, музей, муниципалитет, награда, название, население, национальность, область, образование, опера, оператор, описание, организация, остров, отель, отношение, партия, партнер, период, персона, персонаж, песня, пилот, площадь, поезд, позиция, пол, порт, порядок, правительство, премия, префектура, примечание, провинция, программа, продолжительность, продукт, продюсер, проект, производитель, происхождение, пьеса, работа, работодатель, размер, район, ракета, ранг, расстояние, результат, река, религия, роман, сайт, самолет, символ, сингл, служба, событие, создатель, сообщество, состояние, спорт, спортсмен, ссылка, стадион, стандарт, станция, статистика, статус, статья, столица, страна, структура, судья, сценарист, театр, теннисист, территория, техника, тип, транспорт, требование, трек, тренер, турнир, улица, университет, устройство, фигурист, фильм, фирма, флаг, формат, футболист, художник, цвет, цитата, шахматист, элемент, язык.
    
    Your task is to choose only ONE type from the list to annotate the given input text.
    
    [INST]Solve this task by following these steps: 1. Choose only one valid type from the given list of types. 2. Check that the type MUST be in the given list of valid types. 3. Give the answer in valid JSON format.[/INST]"""

    prompt = sys_message + usr_message
    return prompt


def read_config(path="config.yaml"):
    with open(path, mode="r") as f:
        config = yaml.safe_load(f)
    return config


def get_sampling_params(config) -> SamplingParams:
    sampling = config["sampling"]
    return SamplingParams(
        temperature=sampling["temperature"], 
        top_p=sampling["top_p"],
        max_tokens=sampling["max_tokens"],
        structured_outputs=StructuredOutputsParams(json=ResponseModel.model_json_schema())
    )


def load_model(config) -> LLM:
    model_name = config["model_name"]
    optim = config["optim"]
    return LLM(
        model=model_name,
        quantization=optim["quantization"], 
        max_num_seqs=optim["max_num_seqs"], 
        max_model_len=optim["max_model_len"],
        cpu_offload_gb=optim["cpu_offload_gb"],
        gpu_memory_utilization=optim["gpu_memory_utilization"],
    )


def read_dataset(path: str) -> pd.DataFrame:
    return pd.DataFrame(load_dataset(path, split="test"))


def read_labels(config):
    return pd.read_csv(config["labels_path"])[config["labels_column"]].tolist()


def save_results(config, logits: list, labels: list):
    model_short_name = config["model_short_name"]
    infernece_result = pd.DataFrame({
        "logits": logits,
        "labels": labels
    })

    infernece_result.to_csv(
        f"results/{model_short_name}-inference-results-{datetime.now():%Y-%m-%d %H:%M:%S}.csv",
        sep="|",
        index=False
    )

    print("> Job done >>> Exiting...")
