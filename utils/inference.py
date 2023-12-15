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

# ----------------------- Generic utils ----------------------- #

def load_file(path): 
    '''
    Given a .csv or .json or .jsonl or .txt file,
    load it into a dataframe or string.
    '''
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist.")
    if '.csv' in path:
        data = pd.read_csv(path)
    elif '.jsonl' in path:
        data = pd.read_json(path, lines=True)
    elif '.json' in path:
        data = pd.read_json(path)
    elif '.txt' in path:
        with open(path, 'r') as f:
            data = f.read()
    else: 
        raise ValueError(f"Provided path {path} is not a valid file.")
    return data

def save_file(df, path):
    '''
    Given a dataframe, save it to a .csv or .json or .jsonl file.
    '''
    if '.csv' in path:
        df.to_csv(path, index=False)
    elif '.jsonl' in path:
        df.to_json(path, orient='records', lines=True)
    elif '.json' in path:
        df.to_json(path, orient='records')
    else: 
        raise ValueError(f"Provided path {path} is not a .csv, .json or .jsonl file.")
    return df

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

    # Load model and tokenizer
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found at {model_path}.')
    model = AutoModelForCausalLM.from_pretrained(model_path, use_cache=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_cache=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load instruction and data
    dataset = load_file(data_path)
    dataset['model_name'] = model_name
    if output_path is None:
        output_path = data_path.replace('.jsonl', f'-{model_name}.jsonl')

    parameters = {
        'max_new_tokens': 1000,
        'do_sample': True,
        'top_k': 10,
        'num_return_sequences': 2,
        'eos_token_id': tokenizer.eos_token_id,
        'return_full_text': False
    }
    pipe = pipeline("text-generation", model=model, tokenizer= tokenizer, **parameters)
    for i, row in tqdm(dataset.iterrows(), total=len(dataset), 
                       desc=f"Generating answers from {model_name}"):
        answer = pipe(row['prompt'])['generated_text']
        dataset.loc[i, 'pred'] = answer
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