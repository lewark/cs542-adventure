# cs542-adventure

## Planning

1. Directly Prompting Off-The-Shelf LLMs
    - Prompt engineering
    - Multiple prompts (agentic)
        - Give it questions to answer to build a "train of thought", then prompt it to actually give the action to take
    - Can we use reasoning models like Quen, GPT-OSS to accomplish the above for better performance?
2. Fine-tuning
    - LoRA, UnSloth (more control, lower resource usage), Axlotl (potentially better for beginners), TorchTune
    - Fine-tuning based on game walkthroughs (Jericho provides it)
    - Fine-tuning based on dataset from Q*BERT for question-answers (qa-jericho)
3. Q*BERT testing
4. Reinforcement Learning
    - Input as word-embeddings? How do we do action space?
    - Stable Baselines (try it)
5. RAG