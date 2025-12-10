import jericho

from .model import load_model


def make_grpo_dataset(env: jericho.FrotzEnv):
    states = {}
    
    initial_obs, info = env.reset()
    walkthrough = env.get_walkthrough()

    prompts = []
    hashes = []
    
    obs = initial_obs
    for step in walkthrough:
        prompts.append([{"role": "user", "content": obs}])
        
        state_hash = env.get_world_state_hash()
        hashes.append(state_hash)
        if state_hash not in states:
            states[state_hash] = env.get_state()
        
        obs, reward, done, info = env.step(step)

        if done:
            break

    env.close()

    return Dataset.from_dict({"prompt": prompts, "state_hashes": hashes}), states


def shorten_response(response: str):
    return response.split("\n")[0]


# Workaround for segfault in FrotzEnv.set_state
# Create + reuse a separate env within the GRPO reward function
reward_env = None

def get_reward_func(game_file: str):

    def reward_func(prompts, completions, state_hashes, **kwargs): #(prompts, completions, **kwargs):
        global reward_env
        #env.set_state(states[state_hash])
        #env.step
        if reward_env is None:
            reward_env = jericho.FrotzEnv(main_game_file)
        #local_env = env.copy()
        scores = []
        for prompt, completion, state_hash in zip(prompts, completions, state_hashes):
            reward_env.set_state(states[state_hash])
            command = completion[0]["content"]

            cur_inv_size = len(reward_env.get_inventory())
            obs, reward, done, info = reward_env.step(command)
            new_inv_size = len(reward_env.get_inventory())

            if reward_env.get_world_state_hash() == state_hash:
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

            scores.append(reward)
        #local_env.close()

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
        max_steps = 500,
        save_steps = 250,
        max_grad_norm = 1.0,
        report_to = "none",
        output_dir = "outputs",
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
    model, tokenizer = load_model("lora_model")
    grpo_dataset, states = make_grpo_dataset(env)
    reward_func = get_reward_func("./z-machine-games-master/jericho-game-suite/zork1.z5")
    grpo_trainer = make_grpo_trainer(model, tokenizer, grpo_dataset, reward_func)
    grpo_trainer.train()
