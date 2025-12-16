from typing import Optional

from jericho import FrotzEnv, ZObject

from .schema import Room


directions = ["north", "south", "east", "west", "up", "down"]
opposite_directions = {"north":"south", "south":"north", "east":"west", "west":"east", "up":"down", "down": "up"}
short_directions = {direction[0]: direction for direction in directions}
dir_ids = {dirname: index for index, dirname in enumerate(directions)}


class RoomNode:
    model: Room
    num: int
    exits: dict[str, Optional["RoomNode"]]
    visited: bool

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


RoomDict = dict[int, RoomNode]


def update_exits(command: str, last_loc, loc, rooms: RoomDict):
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
                model = Room(name=new_loc.name, objects=[])
                new_room = RoomNode(model, new_loc.num)
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
