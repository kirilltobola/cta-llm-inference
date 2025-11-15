

class PromptRenderer:
    """TODO"""
    def __init__(self, prompt_filename:str, model_name: str, render_rules: dict):
        with open(prompt_filename, mode="r") as f:
            self.prompt = "".join(f.readlines())

        self.model_name = model_name
        self.render_rules = render_rules

    def pre_render(self, labels: list[str]) -> None:
        self.render_rules["<len_labels>"] = str(len(labels))
        self.render_rules["<labels>"] = ", ".join(labels)

        prompt_copy = self.prompt[:]
        for k, v in self.render_rules.items():
            prompt_copy = prompt_copy.replace(k, v)
        self.prompt = prompt_copy

    def render(self, input_text: str) -> str:
        return self.prompt.replace("<input_text>", input_text)
