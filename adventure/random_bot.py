import random
import sys
import time

from jericho import FrotzEnv

from adventure.metrics import ScoreTracker

story_file = sys.argv[1]

env = FrotzEnv(story_file)

obs, info = env.reset()
print(obs)

score_tracker = ScoreTracker(env)

try:
    for i in range(100):
        start_time = time.time()
        actions = env.get_valid_actions()
        action = random.choice(actions)
        end_time = time.time()

        print(">", action)

        obs, reward, done, info = env.step(action)

        score_tracker.update(info, start_time, end_time)

        print(obs)
        if done:
            break

    score_tracker.get_stats(env, info)
finally:
    env.close()
