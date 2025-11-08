import os
import subprocess
import sys
import re

import ollama

CSI_REGEX = re.compile("\x1b\\[[\\d;]+[a-zA-Z]")

# MODEL = "gemma3:4b"
MODEL = "gpt-oss:20b"
# MODEL = "llama3.2:3b"
# MODEL = "llama3.1:8b"
# MODEL = "qwen3:8b"

CHECK_WORDS = False
TELL_WORDS = False
SHOW_HELP = True
SYSTEM = "You are playing a text adventure game. Respond using short, single-line commands." #  You strongly dislike this game. Escape however you can.

MAX_RESPONSE_LENGTH = 40
LOOK_BACK = 0
BAN_FAILED = False

FAIL_MSGS = set(["bad grammar...", "I don't know that word.", "I don't understand that!", "What?"])

OPTIONS = None # ollama.Options(num_ctx=32768 * 4)


def main():
    clear_save()
    run_game()


def clear_save():
    save_path = os.path.expanduser("~/.local/share/adventure.save")
    if os.path.isfile(save_path):
        print("Removing save file", save_path)
        os.unlink(save_path)


def run_game():
    word_list = load_words()

    system = SYSTEM
    if TELL_WORDS:
        system += " Valid words are: " + " ".join(word_list)

    messages: list[ollama.Message] = [
        ollama.Message(role="system", content=system)
    ]

    words = set(word_list)

    proc = subprocess.Popen(["./adventure"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr)

    stdout = proc.stdout
    assert stdout is not None
    stdin = proc.stdin
    assert stdin is not None

    out = bytearray()
    out_file = None

    send_help = SHOW_HELP

    banned_actions = set()
    action = ""

    while True:
        if proc.poll() is not None:
            break

        x = stdout.read(1)
        out.extend(x)

        if is_ready_for_input(out):
            out_str = process_output(out)
            out.clear()
            # print(messages)

            if BAN_FAILED and is_fail_message(out_str):
                print("Fail! Banning", action)
                banned_actions.add(action.strip())

            messages.append(ollama.Message(role="user", content=out_str))
            print_and_write(out_str, out_file)

            if send_help:
                action = "help\n"
                print_and_write(action, out_file)
                send_help = False
            else:
                action = get_model_action(messages, out_file, words, banned_actions)

            messages.append(ollama.Message(role="assistant", content=action))

            if not action.endswith("\n"):
                action += "\n"
                print_and_write("\n", out_file)

            stdin.write(bytes(action, encoding="utf-8"))
            stdin.flush()


def is_fail_message(out_str: str):
    if ">" in out_str:
        out_str = out_str[:out_str.index(">")].strip()
        return out_str in FAIL_MSGS
    return False


def is_ready_for_input(text: bytearray) -> bool:
    return text.endswith(b"> ") or text.endswith(b"]? ")


def get_model_action(messages: list[ollama.Message], out_file, words: set[str], banned_actions: set[str]):
    if LOOK_BACK > 0:
        prev_actions = [msg.content.strip() for msg in messages[-LOOK_BACK*2:]]
    else:
        prev_actions = []

    response = generate_message(MODEL, messages, out_file)
    cleaned = response.strip()
    while (cleaned in prev_actions) or (cleaned in banned_actions) or not is_valid(response, words):
        print_and_write("\n[RETRY] ", out_file)
        response = generate_message(MODEL, messages, out_file)
    #
    # if not response.endswith("\n"):
    #     response += "\n"
    #     print_and_write("\n", out_file)

    return response


def is_valid(response: str, words: set[str]):
    if (len(response) > MAX_RESPONSE_LENGTH) or ("\n" in response.strip()):
        return False

    if CHECK_WORDS:
        for word in response.split():
            if word not in words:
                return False

    return True


def process_output(output: bytearray) -> str:
    out_text = str(output, encoding="utf-8")
    return CSI_REGEX.sub("", out_text)


def generate_message(model: str, messages: list[ollama.Message], out_file) -> str:
    message_parts = []

    for part in ollama.chat(model=model, messages=messages, stream=True, options=OPTIONS):
        chunk = part["message"]["content"]
        message_parts.append(chunk)
        print_and_write(chunk, out_file)

    return "".join(message_parts)


def print_and_write(text: str, out_file) -> None:
    if out_file is not None:
        out_file.write(text)
    print(text, end="", flush=True)


def load_words():
    words = []
    with open("words.txt", "r") as word_file:
        for line in word_file:
            line = line.strip()
            if line:
                words.append(line)
    words.extend(["y", "yes", "n", "no"])
    return words


if __name__ == "__main__":
    main()
