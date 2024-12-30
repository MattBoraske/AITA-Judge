from huggingface_hub import login
import os
from dotenv import load_dotenv, find_dotenv
from datasets import load_dataset


load_dotenv(find_dotenv())
print(os.getenv('HUGGINGFACE_TOKEN'))
login(token=os.getenv('HUGGINGFACE_TOKEN'))

HF_DATASET = 'MattBoraske/reddit-AITA-submissions-and-comments-multiclass'

hf_dataset = load_dataset(HF_DATASET)