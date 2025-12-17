from typing import Optional

from jericho import FrotzEnv, ZObject
from langchain_core.documents import Document

from .schema import Room


# Obtain room names from the first line of their descriptions.
# This works for Zork but may be incorrect for other games.
NAMES_FROM_DESCRIPTIONS = True


directions = ["north", "south", "east", "west", "up", "down"]
opposite_directions = {"north":"south", "south":"north", "east":"west", "west":"east", "up":"down", "down": "up"}
short_directions = {direction[0]: direction for direction in directions}
dir_ids = {dirname: index for index, dirname in enumerate(directions)}


class RoomNode:
    model: Room
    num: int
    description: str
    exits: dict[str, Optional["RoomNode"]]
    visited: bool
    objects: list["RoomObject"]

    def __init__(self, model: Room, num: int):
        self.exits = {}
        self.model = model
        self.num = num
        self.visited = False

    def describe(self):
        items = ["## " + self.model.name]

        if self.visited:
            if len(self.model.objects) > 0:
                obj_strs = [obj.to_str() for obj in self.model.objects]
                items.append("Objects: " + ", ".join(obj_strs))
            if len(self.exits) > 0:
                exit_strs = ["{} ({})".format(direction, room.model.name) for direction, room in self.exits.items() if room is not None]
                items.append("Exits: " + ", ".join(exit_strs))
        else:
            items.append("Unexplored")
        #unexplored_exits = [direction for direction in directions if direction not in self.exits]
        #if len(unexplored_exits) > 0:
        #    items.append("Unexplored exits: " + ", ".join(unexplored_exits))

        return "\n\n".join(items)

    def to_document(self) -> Document:
        exits: list[str] = []
        exit_directions: list[str] = []

        for direction, other_room in self.exits.items():
            if other_room is not None:
                exits.append(other_room.get_doc_id())
                exit_directions.append(direction)

        doc_id = self.get_doc_id()
        return Document(id=doc_id, page_content=self.description, metadata={"name": self.model.name, "exits": exits, "exit_directions": exit_directions})

    def get_doc_id(self) -> str:
        return str(self.num)


class RoomObject:
    def __init__(self, name: str, description: str, room: RoomNode):
        self.name = name
        self.description = description
        self.room = room

    def get_doc_id(self) -> str:
        room_id = self.room.get_doc_id()
        doc_id = room_id + self.name
        return doc_id

    def to_document(self) -> Document:
        doc_id = self.get_doc_id()
        room_id = self.room.get_doc_id()
        return Document(id=doc_id, page_content=self.description, metadata={"name": self.name, "room_id": room_id})


RoomDict = dict[int, RoomNode]


def update_exits(command: str, last_loc: Optional[ZObject], loc: ZObject, rooms: RoomDict):
    room = rooms[loc.num]
    command_split = command.split()

    if len(command_split) > 0:
        if command_split[0] == "warp":
            pass
        elif last_loc is not None and last_loc.num != loc.num:
            direction = command_split[-1].lower()
            if direction in short_directions:
                direction = short_directions[direction]
            if direction in directions:
                last_room = rooms[last_loc.num]
                new_room = rooms[loc.num]

                last_room.exits[direction] = new_room
        else:
            direction = command_split[-1].lower()
            if direction in short_directions:
                direction = short_directions[direction]
            if direction in directions:
                room.exits[direction] = None


def discover_exits(env: FrotzEnv, loc: ZObject, rooms: RoomDict):
    room = rooms[loc.num]

    state = env.get_state()

    #print(loc)

    for direction in directions:
        obs, reward, done, info = env.step(direction)

        new_loc = env.get_player_location()
        #print(direction, new_loc)
        assert new_loc is not None

        if new_loc.num != loc.num:
            if new_loc.num in rooms:
                new_room = rooms[new_loc.num]
            else:
                loc_desc, _, _, _ = env.step("look")
                if NAMES_FROM_DESCRIPTIONS:
                    loc_name: str = loc_desc.split("\n")[0].strip()
                else:
                    loc_name = loc.name

                model = Room(name=loc_name, objects=[])
                new_room = RoomNode(model, new_loc.num)
                new_room.description = loc_desc
                rooms[new_room.num] = new_room

            room.exits[direction] = new_room
        else:
            room.exits[direction] = None

        env.set_state(state)


def find_room_by_name(rooms: RoomDict, name: str) -> Optional[RoomNode]:
    name = name.lower()
    for room in rooms.values():
        if room.model.name.lower() == name:
            return room
    return None
