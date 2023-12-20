'''
Inference utilities. 
'''

import torch
import os
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

from data import *

PARAMETERS = {
    'max_new_tokens': 1000,
    'do_sample': True,
    'top_k': 10,
    'num_return_sequences': 2,
    'return_full_text': False,
    'device': torch.device('cuda')
}

# ----------------------- Inference utils ----------------------- #

def generate(model_name, 
             model_path, 
             data_path,
             output_path=None):
    '''
    Loads a model and generates summaries for all the files in the path.
    Can be used for either 
    - Generator: Generate clinical notes from patient summaries
    - Direct: Generate clinical notes from patient-doctor conversations

    Arguments: 
        - model_name: Name of the model to be loaded.
        - model_path: Path to the model.
        - data_path: Path to the data file with dialog or patient summaries. 
        - output_path: Path to the output file with generated notes
    '''

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found at {model_path}.')
    
    try: 
        print(f"Loading model {model_name} from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(model_path, use_cache=True, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_cache=False)
    except Exception as e:
        raise ValueError(f"Error when loading model and tokenizer from {model_path}:\n{e}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Loading data from {data_path}...")
    dataset = load_file(data_path)
    dataset['model_name'] = model_name

    try: 
        print(f"Initalizing pipeline...")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, 
                        eos_token_id=tokenizer.eos_token_id, **PARAMETERS)
    except Exception as e:
        raise ValueError(f"Error when initializing pipeline:\n{e}")
    
    if output_path is None:
        output_path = data_path.replace('.jsonl', f'-{model_name}.jsonl')

    os.makedirs(os.path.join(DATA_DIR, 'inference'), exist_ok=True)
    
    for i, row in tqdm(dataset.iterrows(), total=len(dataset), 
                       desc=f"Generating answers from {model_name}"):
        try: 
            answer = pipe(row['prompt'])['generated_text']
            dataset.loc[i, 'pred'] = answer
        except Exception as e:
            print(f"Error in generating answer for {row['prompt']}: {e}")
        if i % 10 == 0: 
            save_file(dataset, output_path)
    save_file(dataset, output_path)
    return dataset
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', 
                        type=str, 
                        default='meditron-7b', 
                        help='Model name to be loaded.')
    parser.add_argument('--model_path', 
                        type=str, 
                        default='/pure-mlo-scratch/make_project/trial-runs/meditron-7b-summarizer/hf_checkpoint/', 
                        help='Path to the model.')
    parser.add_argument('--data_path', 
                        type=str, 
                        default='data/direct_train.jsonl',
                        help='Path to the data file with dialog or patient summaries.')
    parser.add_argument('--output_path', 
                        type=str, 
                        default=None, 
                        help='Path to the output file with generated notes. If None, saved to directory of data_path.')
    args = parser.parse_args()

    generate(args.model_name, args.model_path, args.prompt_path, args.data_path, args.output_path)