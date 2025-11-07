# cs542-adventure

## Planning

1. Directly Prompting Off-The-Shelf LLMs
    - Prompt engineering
    - Multiple prompts (agentic)
        - Give it questions to answer to build a "train of thought", then prompt it to actually give the action to take
    - Can we use reasoning models like Qwen, GPT-OSS to accomplish the above for better performance?
2. Fine-tuning
    - LoRA, UnSloth (more control, lower resource usage), Axlotl (potentially better for beginners), TorchTune
    - Fine-tuning based on game walkthroughs (Jericho provides it)
    - Fine-tuning based on dataset from Q*BERT for question-answers (qa-jericho)
3. Q*BERT testing
4. Reinforcement Learning
    - Input as word-embeddings? How do we do action space?
    - Stable Baselines (try it)
5. RAG
    1. Get room prompt
    2. Turn into embedding vector
    3. Use vector to access vector DB for relevant info we've learned
    4. Build out our action prompt by taking room prompt, any relevant info from DB, and whatever final prompt we want to give the LLM
    5. Semantically split room prompt and put into database
    6. Send prompt to LLM and take action
    - Vector databases
        - Redis
        - Postgres with pgvector
        - sqlite with sqlite-vec <- Tyson's favorite idea
        - Lots of specialized options, like FAISS, QDrant, Chroma
