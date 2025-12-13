import jericho
from .prompt import get_prompt


def get_steps(filename: str):
    env = jericho.FrotzEnv(filename)
    
    initial_obs, info = env.reset()
    walkthrough = env.get_walkthrough()

    steps = []
   
    done = False
    obs = initial_obs
    for step in walkthrough:
        prompt = get_prompt(env, obs, done)
        steps.append((prompt, step))
        #print(obs, step)
        obs, reward, done, info = env.step(step)
        #print(reward)
        if done:
            break

    env.close()

    return steps


from datasets import Dataset
#from unsloth import standardize_sharegpt

def steps_to_dataset(steps: list[list[tuple[str, str]]], length: int, overlap: bool = True):
    convos = []

    for game in steps:
        convo = []
        n = 0
        
        for step in game:
            convo.append({"role": "user", "content": step[0]})
            convo.append({"role": "assistant", "content": step[1]})
            n += 1
            if overlap:
                if length > 0 and n > length:
                    n -= 1
                    convo.pop(0)
                    convo.pop(0)
                    
                convos.append(list(convo))
            else:
                if length > 0 and n >= length:
                    n = 0
                    convos.append(convo)
                    convo = []

        if len(convo) > 0:
            convos.append(convo)

    return Dataset.from_dict({"conversations": convos})


def get_dataset(game_files: list[str], length: int, overlap: bool):
    steps = []
    for game_file in game_files:
        steps.append(get_steps(game_file))
    dataset = steps_to_dataset(steps, length=length, overlap=overlap)

    return dataset


def format_dataset(tokenizer, dataset):
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt = False)
            for convo in convos
        ]
        return {'text': texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset
