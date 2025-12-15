from jericho import FrotzEnv

class ScoreTracker:
    def __init__(self, env: FrotzEnv):
        self.env = env
        self.unique_hashes = set()
        self.unique_rooms = set()
        self.unique_items = set()

        self.prev_score = -1
        self.total_steps = 0
        self.retries = 0
        self.retries_per_score = []
        self.generate_times = []

        self.update_from_env()

    def update_from_env(self):
        loc = self.env.get_player_location()
        if loc is None:
            raise ValueError("No initial player location is set")

        self.unique_rooms.add(loc.name)
        self.unique_hashes.add(self.env.get_world_state_hash())
        for item in self.env.get_inventory():
            self.unique_items.add(item.name)

    def update(self, info: dict, start_time: float, end_time: float):
        self.update_from_env()

        self.retries += 1
        if info["score"] != self.prev_score:
            self.retries_per_score.append(self.retries)
            self.retries = 0
        self.prev_score = info["score"]
        self.total_steps += 1

        self.generate_times.append(end_time - start_time)

    def get_stats(self, env: FrotzEnv, info: dict):
        stats = {
            'unique_rooms': len(self.unique_rooms),
            'unique_hashes': len(self.unique_hashes),
            'unique_items': len(self.unique_items),
            'score': info['score'],
            'max_score': env.get_max_score(),
            'avg_retries': sum(self.retries_per_score) / max(len(self.retries_per_score), 1),
            'avg_generate_time': sum(self.generate_times) / len(self.generate_times),
        }
        print(stats)
        return stats
