from jericho import FrotzEnv, ZObject

from adventure.graph import RoomNode, RoomDict, directions, dir_ids, find_room_by_name


def warp_command(rooms: RoomDict, env: FrotzEnv, loc: ZObject, command_split: list[str], info: dict):
    room_name = " ".join(command_split[1:])
    end_room = find_room_by_name(rooms, room_name)

    obs = ""

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

    return obs, info


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
