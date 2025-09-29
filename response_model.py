from pydantic import BaseModel

class ResponseModel(BaseModel):
    type: str

    def __str__(self):
        return f"{self.type}"
