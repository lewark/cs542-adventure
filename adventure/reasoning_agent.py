import os
import sys
import time

import ollama
import jericho
import re
import json
import random

import time
from .metrics import ScoreTracker

os.environ['PATH'] = f'{os.environ["PATH"]}:./ollama/bin' # Add local ollama


def n_steps(turn_func, env, n=100):
    score_tracker = ScoreTracker(env)

    for _ in range(n):

        # Turn
        start = time.time()
        done, info = turn_func()
        end = time.time()
        
        score_tracker.update(info, start, end)

        if done:
            break
        score_tracker.get_stats(env, info)

    return score_tracker.get_stats(env, info)

def agent(model='qwen3'):
    
    game = 'zork1.z5'
    GAMES_DIR = "z-machine-games-master/jericho-game-suite"
    env = jericho.FrotzEnv(f"{GAMES_DIR}/{game}")


    system_prompt = (
        f'Think step by step. You are playing {game}, an interactive fiction game. You must analyze the scenario the game presents to you and choose an action that will make progress. Your goal is to finish the game\n'
        'Use the tools provided to you to take actions, view possible actions for your current location, and view the game walkthrough if necessary'
    )

    game_response, info = env.reset()
    print(game_response)
    done = False

    memory = [
        ollama.Message(role='system', content=system_prompt)
    ]
    max_memory = 20

    def do_game_action(action: str) -> str:
        """Perform an action in the active text adventure game and see the result"""
        """
        Args:
          action: game action string

        Returns:
          The game's response after performing the action
        """
        nonlocal done, info
        game_response, reward, done, info = env.step(action)
        return game_response
    
    def view_possible_actions() -> str:
        """View a list of the actions that can be performed in the game's current state"""
        """
        Returns:
          String containg actions separated by commas
        """
        return ', '.join(env.get_valid_actions())
    
    def view_walkthrough():
        """View the full game walkthrough as a list of actions"""
        """
        Returns:
          String containing actions separated by newlines
        """
        return env.get_walkthrough()
    
    available_functions = {
        'do_game_action': do_game_action,
        'view_possible_actions': view_possible_actions,
        'view_walkthrough': view_walkthrough
    }

    def turn():
        nonlocal game_response, memory
        memory.append(ollama.Message(role='user', content=f'Remember that you are the player and there is no human in the loop.\nGame:\n{game_response}'))
            
        response = ollama.chat(model=model, messages=memory, think=True, tools=[do_game_action, view_possible_actions, view_walkthrough], options={'num_ctx': 2048})
        memory.append(response.message)

        print("Thinking: ", response.message.thinking)
        print("Content: ", response.message.content)

        if response.message.tool_calls:
            for tc in response.message.tool_calls:
                if tc.function.name in available_functions:
                    print(f"Calling {tc.function.name} with arguments {tc.function.arguments}")
                    result = available_functions[tc.function.name](**tc.function.arguments)
                    print(f"Result: {result}")
                    # add the tool result to the messages
                    memory.append({'role': 'tool', 'tool_name': tc.function.name, 'content': str(result)})
                    if len(memory) > max_memory:
                        memory.pop(1) # Don't remove system prompt
        return done, info
              
    result = n_steps(turn, env)              
    print(result)
    print('Memory at end:')
    print(json.dumps(memory, indent=4, default=str))
    return result


print(agent())