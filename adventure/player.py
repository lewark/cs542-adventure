import time

import jericho

from .prompt import get_prompt

def run_game(model, tokenizer, game_filename: str, n_steps: int, history: int):
    env = jericho.FrotzEnv(game_filename)

    messages = []
    
    obs, info = env.reset()
    print(obs)
    
    unique_hashes = set()
    unique_rooms = set()
    unique_items = set()
    
    unique_rooms.add(env.get_player_location().name)
    unique_hashes.add(env.get_world_state_hash())
    for item in env.get_inventory():
        unique_items.add(item.name)

    prev_score = -1
    total_steps = 0
    retries = 0
    retries_per_score = []
    generate_times = []

    for i in range(n_steps):
        if len(messages) > history:
            messages.pop(0)
            messages.pop(0)

        prompt = get_prompt(env, obs, done)
        messages.append(make_message("user", prompt))
        
        start = time.time()
        response = generate_response(model, tokenizer, messages)
        generate_times.append(time.time() - start)

        print(">", response)
        messages.append(make_message("assistant", response))
        
        obs, reward, done, info = env.step(response)
        print(obs)

        unique_rooms.add(env.get_player_location().name)
        unique_hashes.add(env.get_world_state_hash())
        for item in env.get_inventory():
            unique_items.add(item.name)

        retries += 1
        if info["score"] != prev_score:
            retries_per_score.append(retries)
            retries = 0
        prev_score = info["score"]
        total_steps += 1

        if done:
            break
    
    stats = {
        'unique_rooms': len(unique_rooms),
        'unique_hashes': len(unique_hashes),
        'unique_items': len(unique_items),
        'score': info['score'],
        'max_score': env.get_max_score(),
        'avg_retries': sum(retries_per_score) / max(len(retries_per_score), 1),
        'avg_generate_time': sum(generate_times) / len(generate_times),
    }
    print(stats)
    
    env.close()
    return stats


def make_message(role, content):
    return {"role": role, "content": content}


def get_input_ids(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")


def generate_response(model, tokenizer, messages):
    input_ids = get_input_ids(tokenizer, messages)

    output_ids = model.generate(input_ids,
        max_new_tokens = 128,
    )
    out = tokenizer.batch_decode(output_ids)

    #print(out)
    
    out_line = out[0]
    start_token = "<|end_header_id|>"
    end_token = tokenizer.eos_token
    
    start_index = out_line.rindex(start_token) + len(start_token)
    end_index = out_line.rindex(end_token)
    
    return out_line[start_index : end_index].strip()


if __name__ == "__main__":
    from .model import load_model
    from unsloth import FastLanguageModel
    model, tokenizer = load_model("grpo_model_2", "llama-3.2")
    FastLanguageModel.for_inference(model)
    
    run_game(model, tokenizer, "./z-machine-games-master/jericho-game-suite/zork1.z5", 100, history=10)
