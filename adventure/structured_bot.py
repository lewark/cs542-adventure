from typing import Optional

import ollama
from jericho import FrotzEnv, ZObject

from .schema import Room, GameObject, RoomNode, room_schema, directions, short_directions, opposite_directions, dir_ids

MODEL = "llama3.2:3b"
GAME = "z-machine-games-master/jericho-game-suite/zork1.z5"

ROOM_PROMPT = """Describe this room, including all interactive objects. The room's name is typically listed on the first line.

{}"""

UPDATE_ROOM_PROMPT = """The previous description of this room is as follows.

```json
{}
```

If needed, update the description of the room or its interactive objects based on the following command and its result.
Please list any new information pertaining to objects that may be important to know later.
If an item was taken, remove it from the room description.
If an operation acts on an object not listed, create an entry for it.
If a particular operation cannot be done on an object, make a note of this.

Command: {}

Result:
{}"""



RoomDict = dict[int, RoomNode]

def main():
    env = FrotzEnv(GAME)

    rooms: RoomDict = {}

    obs, info = env.reset()
    #print(obs, info)

    command = ""
    commands = []

    last_loc = None

    while True:
        loc = env.get_player_location()
        assert loc is not None

        if loc.num not in rooms:
            room_model = Room(name=loc.name, objects=[])
            room = RoomNode(room_model, loc.num)
            rooms[room.num] = room

        room = rooms[loc.num]
        if room.visited:
            pass
            #room_model = update_room_model(room.model, command, obs)
            #room.model = room_model
            #print(room_model)
        else:
            discover_exits(env, loc, rooms)
            #room_model = extract_room_model(obs)
            #room.model = room_model
            room.visited = True

        update_exits(command, last_loc, loc, rooms)
        last_loc = loc

        prompt = get_prompt(env, obs, False, rooms, commands[-10:])
        print(prompt)
        #print(obs)

        command = get_next_command(prompt)
        print(">", command)
        #command = input("> ")
        #commands.append(command)

        command_split = command.split()
        if len(command_split) > 0 and command_split[0] == "warp":
            warp_command(rooms, env, loc, command_split)
        else:
            obs, reward, done, info = env.step(command)

        #break

    env.close()


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


def warp_command(rooms: RoomDict, env: FrotzEnv, loc: ZObject, command_split: list[str]):
    room_name = " ".join(command_split[1:])
    end_room = find_room_by_name(rooms, room_name)
    if end_room is None:
        obs = "Unable to find room " + room_name
    else:
        room = rooms[loc.num]
        path = find_path(room, end_room, rooms)
        if len(path) > 0:
            for step in path:
                print(">", step)
                obs, reward, done, info = env.step(step)
                print(obs)
        else:
            obs = "Unable to find path to " + room_name


def extract_room_model(obs: str) -> Room:
    prompt = ROOM_PROMPT.format(obs)
    result = ollama.generate(model=MODEL, prompt=prompt, format=room_schema)

    return Room.model_validate_json(result.response)


def update_room_model(room_model: Room, command: str, obs: str):
    model_str = room_model.model_dump_json(indent=2)
    prompt = UPDATE_ROOM_PROMPT.format(model_str, command, obs)
    result = ollama.generate(model=MODEL, prompt=prompt, format=room_schema)

    return Room.model_validate_json(result.response)


def get_next_command(prompt: str):
    result = ollama.generate(model=MODEL, prompt=prompt)
    return result.response


def find_room_by_name(rooms: RoomDict, name: str) -> Optional[RoomNode]:
    name = name.lower()
    for room in rooms.values():
        if room.model.name.lower() == name:
            return room
    return None


def find_path(start_room: RoomNode, end_room: RoomNode, rooms: RoomDict) -> list[str]:
    visited: dict[int, tuple[int, int]] = {} # room: (in_direction, from_room)
    next_rooms: list[int] = []

    cur_room = start_room
    while True:
        for direction, room in cur_room.exits.items():
            if room is None:
                continue

            if room.num not in visited:
                dir_id = dir_ids[direction]
                visited[room.num] = (dir_id, cur_room.num)
                next_rooms.append(room.num)

            if room is end_room:
                return backtrace(start_room, end_room, visited, rooms)

        if len(next_rooms) == 0:
            return []
        cur_room = rooms[next_rooms.pop(0)]


def backtrace(start_room: RoomNode, end_room: RoomNode, visited: dict[int, tuple[int, int]], rooms: RoomDict) -> list[str]:
    path = []
    cur_room = end_room
    while cur_room is not start_room:
        in_dir, from_room = visited[cur_room.num]
        in_dir_str = directions[in_dir]
        path.insert(0, in_dir_str)
        cur_room = rooms[from_room]

    return path



def get_prompt(env: FrotzEnv, obs: str, done: bool, rooms: RoomDict, commands: list[str]):
    items=[]
    state = env.get_state()

    loc = env.get_player_location()
    assert loc is not None
    room = rooms[loc.num]

    if len(rooms) > 0:
        items.append("# Known rooms")
        items.extend([other_room.describe() for other_room in rooms.values() if other_room is not room])

    if not done:
        inv_desc, reward, done, info = env.step("inventory")
        items.append("# Inventory\n" + inv_desc.strip())

    if not done:
        look_desc, reward, done, info = env.step("look")
        if not obs.endswith(look_desc):
            items.append("# Current location\n" + look_desc.strip())

    items.append("# Current observation\n" + obs.strip())

    items.append("# Current room")
    items.append(room.describe())

    items.append("# Task")
    items.append("""Your goal is to win the game.
Output a short command, one to two words, describing your next action.
You can move between rooms by entering the direction to move.
To travel to a specific room, enter 'warp' followed by the name of the room.""")
    #if len(commands) > 0:
    #    items.append("Previous commands: " + ", ".join(commands))


    # if not done:
    #     valid_actions = env.get_valid_actions()
    #     bullets = ["- " + action for action in valid_actions]
    #     items.append("##Available actions\n" + "\n".join(bullets))

    # env.set_state(state)

    return "\n\n".join(items)



if __name__ == "__main__":
    main()
