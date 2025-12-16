import time

import ollama
from jericho import FrotzEnv

from .schema import Room, room_schema
from .graph import RoomNode, RoomDict, discover_exits, update_exits
from .pathfind import warp_command
from .metrics import ScoreTracker

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



def main():
    env = FrotzEnv(GAME)

    rooms: RoomDict = {}

    obs, info = env.reset()
    #print(obs, info)

    command = ""
    commands = []

    score_tracker = ScoreTracker(env)

    last_loc = None

    while True:
        start_time = time.time()
        loc = env.get_player_location()
        assert loc is not None

        if loc.num not in rooms:
            room_model = Room(name=loc.name, objects=[])
            room = RoomNode(room_model, loc.num)
            rooms[room.num] = room

        room = rooms[loc.num]
        if room.visited:
            #pass
            room_model = update_room_model(room.model, command, obs)
            room.model = room_model
            #print(room_model)
        else:
            discover_exits(env, loc, rooms)
            room_model = extract_room_model(obs)
            room.model = room_model
            room.visited = True

        update_exits(command, last_loc, loc, rooms)
        last_loc = loc

        prompt = get_prompt(env, obs, False, rooms, commands[-10:])
        print(prompt)
        #print(obs)

        command = get_next_command(prompt)
        end_time = time.time()

        print(">", command)
        #command = input("> ")
        #commands.append(command)

        command_split = command.split()
        if len(command_split) > 0 and command_split[0] == "warp":
            obs, info = warp_command(rooms, env, loc, command_split, info)
        else:
            obs, reward, done, info = env.step(command)

        score_tracker.update(info, start_time, end_time)
        score_tracker.get_stats(env, info)

        #break

    env.close()


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
