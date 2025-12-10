import jericho

def run_game(game_filename: str, n_steps: int, with_history: bool):
    env = jericho.FrotzEnv(game_filename)

    messages = []
    
    obs, info = env.reset()
    print(obs)
    
    for i in range(n_steps):
        if not with_history:
            messages.clear()

        messages.append(make_message("user", obs))
        
        response = generate_response(messages)
        print(">", response)
        messages.append(make_message("assistant", response))
        
        obs, reward, done, info = env.step(response)
        print(obs)
        if done:
            break
    
    env.close()

def make_message(role, content):
    return {"role": role, "content": content}


def get_input_ids(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")


def generate_response(model, tokenizer, messages):
    input_ids = get_input_ids(messages)

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
