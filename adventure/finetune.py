import glob

import unsloth
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

from .dataset import get_dataset, format_dataset
from .model import load_model, save_model, max_seq_length


def make_trainer(model, tokenizer, dataset):
    enable_bf16 = is_bfloat16_supported()

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences
        #formatting_func = formatting_prompts_func,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 240, #60,
            # num_train_epochs = 1,
            learning_rate = 2e-4,
            fp16 = not enable_bf16,
            bf16 = enable_bf16,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        )
    )

    # https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    return trainer


if __name__ == "__main__":
    #game_files = [
    #    name
    #    for name in glob.glob("./z-machine-games-master/jericho-game-suite/*.z5")
    #    if "zork1" not in name
    #]
    game_files = ["./z-machine-games-master/jericho-game-suite/zork1.z5"]
    dataset = get_dataset(game_files)

    model, tokenizer = load_model("unsloth/Llama-3.2-3B-Instruct-bnb-4bit", "llama-3.2")
    dataset = format_dataset(tokenizer, dataset)

    trainer = make_trainer(model, tokenizer, dataset)
    trainer_stats = trainer.train()

    save_model(model, tokenizer, "lora_model")

    from unsloth import FastLanguageModel
    model, tokenizer = load_model("lora_model", "llama-3.2")
    FastLanguageModel.for_inference(model)

    from .player import run_game
    run_game(model, tokenizer, "./z-machine-games-master/jericho-game-suite/zork1.z5", 100, True)

