import jericho

def get_prompt(env: jericho.FrotzEnv, obs: str, done: bool, include_actions: bool = False):
    items=["##Observation\n" + obs]
    state = env.get_state()

    if not done:
        look_desc, reward, done, info = env.step("look")
        if not obs.endswith(look_desc):
            items.append("##Location\n" + look_desc)

    if not done:
        inv_desc, reward, done, info = env.step("inventory")
        items.append("##Inventory\n" + inv_desc)

    if include_actions and not done:
        valid_actions = env.get_valid_actions()
        bullets = ["- " + action for action in valid_actions]
        items.append("##Available actions\n" + "\n".join(bullets))

    env.set_state(state)

    return "\n\n".join(items)
