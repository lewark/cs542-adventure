import time

from jericho import FrotzEnv

from .prompt import get_prompt
from .metrics import ScoreTracker


def run_game(model, tokenizer, game_filename: str, n_steps: int, history: int, include_actions: bool = True):
    env = FrotzEnv(game_filename)

    messages = []

    obs, info = env.reset()
    print(obs)

    score_tracker = ScoreTracker(env)
    done = False

    for i in range(n_steps):
        if len(messages) > history:
            messages.pop(0)
            messages.pop(0)

        prompt = get_prompt(env, obs, done, include_actions)
        messages.append(make_message("user", prompt))

        start_time = time.time()
        response = generate_response(model, tokenizer, messages)
        end_time = time.time()

        print(">", response)
        messages.append(make_message("assistant", response))

        obs, reward, done, info = env.step(response)
        print(obs)

        score_tracker.update(info, start_time, end_time)

        if done:
            break

    env.close()
    return score_tracker.get_stats(env, info)


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
    model, tokenizer = load_model("grpo_model_result", "llama-3.2")
    FastLanguageModel.for_inference(model)

    run_game(model, tokenizer, "./z-machine-games-master/jericho-game-suite/zork1.z5", 100, history=10)
