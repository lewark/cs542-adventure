from typing import Annotated, Optional
from pydantic import BaseModel, Field


class Room(BaseModel):
    name: str
    objects: list["GameObject"]


class GameObject(BaseModel):
    name: str
    notes: list[str]

    def to_str(self):
        if len(self.notes) > 0:
            adjs = ", ".join(self.notes)
            return f"{self.name} ({adjs})"
        return self.name



room_schema = Room.model_json_schema()
object_schema = GameObject.model_json_schema()
