from typing import Annotated, Optional
from pydantic import BaseModel, Field

directions = ["north", "south", "east", "west", "up", "down"]
opposite_directions = {"north":"south", "south":"north", "east":"west", "west":"east", "up":"down", "down": "up"}
short_directions = {direction[0]: direction for direction in directions}
dir_ids = {dirname: index for index, dirname in enumerate(directions)}


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


class RoomNode:
    model: Room
    num: int
    exits: dict[str, Optional["RoomNode"]]

    def __init__(self, model: Room, num: int):
        self.exits = {}
        self.model = model
        self.num = num

    def describe(self):
        items = ["## " + self.model.name]
        if len(self.model.objects) > 0:
            obj_strs = [obj.to_str() for obj in self.model.objects]
            items.append("Objects: " + ", ".join(obj_strs))
        if len(self.exits) > 0:
            exit_strs = ["{} ({})".format(direction, room.model.name) for direction, room in self.exits.items() if room is not None]
            items.append("Exits: " + ", ".join(exit_strs))
        unexplored_exits = [direction for direction in directions if direction not in self.exits]
        if len(unexplored_exits) > 0:
            items.append("Unexplored exits: " + ", ".join(unexplored_exits))

        return "\n\n".join(items)
