1. Get room prompt
2. Turn into embedding vector
3. Use vector to access vector DB for relevant info we've learned
4. Build out our action prompt by taking room prompt, any relevant info from DB, and whatever final prompt we want to give the LLM
5. Semantically split room prompt and put into database
6. Send prompt to LLM and take action
