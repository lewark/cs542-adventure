import sys

from jericho import FrotzEnv

story_file = sys.argv[1]

env = FrotzEnv(story_file)

obs, info = env.reset()
print(obs)

try:
    while True:
        command = input("> ")

        obs, reward, done, info = env.step(command)

        loc = env.get_player_location()
        #print(loc)

        obj_num = loc.child
        while obj_num > 0:
            x = env.get_object(obj_num)
            print(x)
            obj_num = x.sibling

        print(obs)
        if done:
            break

finally:
    env.close()
