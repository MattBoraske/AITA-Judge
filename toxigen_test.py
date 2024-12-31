from transformers import pipeline
import torch
from multiprocessing import freeze_support

def run_toxigen():
    # load toxigen roberta model
    if torch.cuda.is_available():
        toxigen_roberta = pipeline("text-classification", 
                                 model="tomh/toxigen_roberta", 
                                 truncation=True, 
                                 device_map='cuda')
    else:
        toxigen_roberta = pipeline("text-classification", 
                                 model="tomh/toxigen_roberta", 
                                 truncation=True, 
                                 device_map='cpu')

    # Test both individual and batched inputs
    test_inputs = ['I love you', 'I hate you']

    for text in test_inputs:
        result = toxigen_roberta(text)
        print(f"Text: {text}")
        print(f"Result: {result}\n")

if __name__ == '__main__':
    freeze_support()
    run_toxigen()