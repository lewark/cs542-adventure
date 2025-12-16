import time
from typing import Optional

from graph_retriever.strategies import Eager
from langchain.agents import create_agent
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_graph_retriever import GraphRetriever
from jericho import FrotzEnv, ZObject

from adventure.metrics import ScoreTracker
from adventure.pathfind import warp_command
from adventure.schema import Room
from adventure.structured_bot import extract_room_model, update_room_model

from .graph import RoomDict, RoomNode, discover_exits, update_exits

CHAT_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"
GAME = "z-machine-games-master/jericho-game-suite/zork1.z5"

PROMPT_TEMPLATE = """# Location

{}

# Inventory

{}

# Observation

Previous command: {}

Result:
{}

What action do you take next?"""
SYSTEM_PROMPT = "You are playing a text adventure game. Output short one to two word commands to advance through the game."


class Game:
    def __init__(self):
        self.model = ChatOllama(model=CHAT_MODEL)
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.vector_store = InMemoryVectorStore(self.embeddings)

        self.agent = create_agent(model=self.model, tools=None, system_prompt=SYSTEM_PROMPT)

        # https://docs.langchain.com/oss/python/integrations/retrievers/graph_rag#inmemory
        self.traversal_retriever = GraphRetriever(
            store=self.vector_store,
            edges = [],
            strategy = Eager(k=5, start_k=1, max_depth=2)
        )

        self.env = FrotzEnv(GAME)

        self.rooms: RoomDict = {}

    def run_game(self):
        obs, info = self.env.reset()
        #print(obs, info)

        command = ""
        commands = []

        score_tracker = ScoreTracker(self.env)

        last_loc = None

        while True:
            start_time = time.time()

            loc = self.env.get_player_location()
            assert loc is not None

            self.update_rooms(command, loc, last_loc)

            last_loc = loc

            prompt = self.get_prompt(obs, command)
            print(prompt)
            #print(obs)

            command = self.get_next_command(prompt)
            end_time = time.time()

            print(">", command)
            #command = input("> ")
            #commands.append(command)

            command_split = command.split()
            if len(command_split) > 0 and command_split[0] == "warp":
                obs, info = warp_command(self.rooms, self.env, loc, list(command_split), info)
            else:
                obs, reward, done, info = self.env.step(command)

            score_tracker.update(info, start_time, end_time)
            score_tracker.get_stats(self.env, info)

            #break

        env.close()
            #room_model = extract_room_mode


    def update_rooms(self, command: str, loc: ZObject, last_loc: Optional[ZObject]):
        if loc.num not in self.rooms:
            room_model = Room(name=loc.name, objects=[])
            room = RoomNode(room_model, loc.num)
            self.rooms[room.num] = room

        room = self.rooms[loc.num]
        if room.visited:
            pass
            #room_model = update_room_model(room.model, command, obs)
            #room_model = Room()
            #room.model = room_model
            #print(room_model)
        else:
            discover_exits(self.env, loc, self.rooms)
            #room_model = extract_room_model(obs)
            #room_model = Room(name=lo)
            #room.model = room_model
            room.visited = True

        update_exits(command, last_loc, loc, self.rooms)


    def get_prompt(self, obs: str, prev_command: str):
        room_desc, _, _, _ = self.env.step("look")
        inv_desc, _, _, _ = self.env.step("inventory")

        return PROMPT_TEMPLATE.format(room_desc.strip(), inv_desc.strip(), prev_command, obs.strip())

    def get_next_command(self, prompt: str):
        events = []
        for event in self.agent.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            stream_mode="values"
        ):
            events.append(event)

        content = events[-1]["messages"][-1].content
        print(content)
        return content


if __name__ == "__main__":
    game = Game()
    game.run_game()
