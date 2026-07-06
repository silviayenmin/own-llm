# 💾 How to Save and Export Your Custom Model

You might be wondering: *"I spent all this time training the model, but where is it? How do I save it?"*

The great news is: **Your model is already saved!** PyTorch automatically saves the neural network's brain (the weights and biases) during the training process.

## 1. Where is the model saved?

Every time you ran the training scripts, the code automatically saved your model checkpoints into the `experiments/checkpoints/` folder.

You will find two main files there:
1. `best_model.pt` -> This is your **Base Model** (trained on Shakespeare text).
2. `instruct_model.pt` -> This is your **Instruct Model** (fine-tuned on your Q&A dataset).

> **NOTE:** The `.pt` extension stands for **PyTorch Tensor**. This single file contains millions of float numbers representing what your AI learned.

## 2. What exactly is inside the `.pt` file?

If you open the `.pt` file in a text editor, it will look like gibberish because it is a binary file. Inside this file, our code saved a Python Dictionary containing three crucial things:
* `"model_state_dict"`: The actual trained weights of all the Transformer blocks.
* `"optimizer_state_dict"`: The state of the AdamW optimizer (in case you want to pause and resume training later).
* `"config"`: The hyperparameters (Layers, Heads, Embeddings) from `config.yaml` so the model knows its own shape when loading.

## 3. How to Package and Share Your Model

If you want to send this model to a friend, or use it in a completely different Python project, you need to share **two things**: The Model Brain (Weights) and the Dictionary (Tokenizer).

**Follow these steps to export your model:**

1. Create a new folder on your Desktop (e.g., `My_Custom_AI`).
2. Copy the model file you want (e.g., `experiments/checkpoints/instruct_model.pt`) into that folder.
3. Copy the entire Tokenizer folder (`data/tokenizer/`) into that folder.
4. Zip the `My_Custom_AI` folder.

You can now send this Zip file to anyone! As long as they have PyTorch installed, they can load your `.pt` file and generate text using your AI.

## 4. Future Step: Hugging Face (.safetensors)

Right now, we are saving models in PyTorch's native `.pt` format. 
When we move to training "Huge Parameter" models (like 1 Billion parameters), we will convert this `.pt` file into a format called `.safetensors` and upload it to **Hugging Face** (The GitHub for AI models). That way, anyone in the world can download and run your model using standard libraries!
