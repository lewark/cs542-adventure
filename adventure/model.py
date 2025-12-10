from typing import Optional

import unsloth
# Taken from this article:
# https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama
# https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Advanced_Llama3_2_(3B)_GRPO_LoRA.ipynb
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch

max_seq_length = 2048
dtype = None
load_in_4bit = True
lora_rank = 64 # Larger rank = smarter, but slower

def load_model(model_name: str, chat_template_name: Optional[str]):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        fast_inference = True, # required for vLLM, https://github.com/huggingface/open-r1/issues/572
        max_lora_rank = lora_rank,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # any number > 0, suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0, # supports any, but 0 is optimized
        bias = "none", # supports any, but "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False, # We support rank stabilized LoRA
        loftq_config = None # And LoftQ
    )
    
    if chat_template_name is not None:
        tokenizer = get_chat_template(tokenizer, chat_template = chat_template_name)

    return model, tokenizer


def save_model(model, tokenizer, model_name):
    model.save_pretrained(model_name)
    tokenizer.save_pretrained(model_name)


#model, tokenizer = load_model(train_model_name)
