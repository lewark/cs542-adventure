import jericho
from datasets import Dataset, IterableDataset

from .model import load_model, save_model


def make_grpo_dataset(env: jericho.FrotzEnv):
    states = {}
    next_actions = []
    
    initial_obs, info = env.reset()
    initial_hash = env.get_world_state_hash()
    #walkthrough = env.get_walkthrough()

    prompts = []
    hashes = []

    max_prompts = 10
    
    def generate_dataset():
        obs = initial_obs
        #state_hash = env.get_world_state_hash()
        state_hash = initial_hash
        prompts.append({"role": "user", "content": obs})

        print("Yielding first result")

        last_result = {"prompt": prompts, "state_hash": state_hash}
        yield last_result

        print("Yielded first result")

        while True:
            if len(next_actions) == 0:
                print("Warning: Next action list is empty")
                yield last_result
                continue

            state_hash, action, obs = next_actions.pop(0)
            
            prompts.append({"role": "assistant", "content": action})
            prompts.append({"role": "user", "content": obs})

            if len(prompts) > max_prompts:
                prompts.pop(0)
                prompts.pop(0)

            last_result = {"prompt": prompts, "state_hash": state_hash}
            yield last_result

        #env.close()

    return IterableDataset.from_generator(generate_dataset), states, next_actions


def shorten_response(response: str):
    return response.split("\n")[0]


# Workaround for segfault in FrotzEnv.set_state
# Create + reuse a separate env within the GRPO reward function
reward_env = None
global_var = None

def get_reward_func(game_file: str, states: dict, next_actions: list):

    def reward_func(prompts, completions, state_hashes, **kwargs): #(prompts, completions, **kwargs):
        global reward_env
        global global_var
        global_var = "hello"
        #env.set_state(states[state_hash])
        #env.step
        if reward_env is None:
            reward_env = jericho.FrotzEnv(game_file)
        #local_env = env.copy()
        scores = []
        best_score = None
        best_action = None
        best_state = None
        best_hash = None
        best_obs = None
        for prompt, completion, state_hash in zip(prompts, completions, state_hashes):
            reward_env.set_state(states[state_hash])
            command = completion[0]["content"]

            cur_inv_size = len(reward_env.get_inventory())
            obs, reward, done, info = reward_env.step(command)
            new_inv_size = len(reward_env.get_inventory())
            new_hash = reward_env.get_world_state_hash()

            if new_hash == state_hash:
                # Punish taking an invalid action
                reward -= 1.0
            #if reward_env.get_world_state_hash() != state_hash:
            #    # Reward taking a valid action
            #    reward += 1.0
            if new_inv_size > cur_inv_size:
                # Reward picking up items
                reward += new_inv_size - cur_inv_size

            short_desc = shorten_response(prompt[0]["content"])
            short_obs = shorten_response(obs)
            #print(f"'{short_desc}': '{command}' -> '{short_obs}' {reward}")
            #print(info)

            if best_score is None or reward > best_score:
                best_score = reward
                best_action = command
                best_state = reward_env.get_state()
                best_hash = new_hash
                best_obs = obs

            scores.append(reward)
        #local_env.close()

        states[best_hash] = best_state
        next_actions.append((best_hash, best_action, best_obs))

        #print(scores)
        return scores

    return reward_func


# https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo
# https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Advanced_Llama3_2_(3B)_GRPO_LoRA.ipynb
from trl import GRPOConfig, GRPOTrainer


def make_grpo_trainer(model, tokenizer, grpo_dataset, reward_func):
    max_seq_length = 2048
    max_prompt_length = 287 + 1

    grpo_training_args = GRPOConfig(
        use_vllm = True, # use vLLM for fast inference!
        learning_rate = 5e-6,
        #adam_beta1 = 0.9,
        #adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        logging_steps = 1,
        per_device_train_batch_size = 1,
        #bf16 = is_bfloat16_supported(),
        #fp16 = not is_bfloat16_supported(),
        gradient_accumulation_steps = 4, # Increase to 4 for smoother training, decrease if OOM
        num_generations = 4, # Decrease if out of memory
        max_prompt_length = max_prompt_length,
        max_completion_length = max_seq_length - max_prompt_length,
        max_steps = 5,
        save_steps = 250,
        max_grad_norm = 1.0,
        report_to = "none",
        output_dir = "outputs",
        dispatch_batches = False
    )

    # https://csolab.research.google.com/github/unslothai/notebooks/blob/main/nb/Advanced_Llama3_2_(3B)_GRPO_LoRA.ipynb

    grpo_trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            reward_func
        ],
        args = grpo_training_args,
        train_dataset = grpo_dataset,
    )

    return grpo_trainer


if __name__ == "__main__":
    model, tokenizer = load_model("lora_model", "llama-3.2")
    env_file = "./z-machine-games-master/jericho-game-suite/zork1.z5"
    
    env = jericho.FrotzEnv(env_file)
    grpo_dataset, states, next_actions = make_grpo_dataset(env)
    env.close()

    reward_func = get_reward_func(env_file, states, next_actions)
    grpo_trainer = make_grpo_trainer(model, tokenizer, grpo_dataset, reward_func)
    grpo_trainer.train()
    print(global_var)
    save_model(model, tokenizer, "grpo_model_2")
