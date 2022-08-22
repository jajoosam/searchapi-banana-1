# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface GPTJ model

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b3", use_cache=True)
    AutoTokenizer.from_pretrained("bigscience/bloom-1b3")


if __name__ == "__main__":
    download_model()