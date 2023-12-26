'''
Inference utilities. 
'''

import torch
import os
import argparse
import pandas as pd
import json as json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline, StoppingCriteria

from data import *

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = []):
      StoppingCriteria.__init__(self), 

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops = []):
      self.stops = stops
      for i in range(len(stops)):
        self.stops = self.stops[i]



SUMMARIZER_PARAMETERS = {
    'max_new_tokens': 1536,
    'do_sample': True,
    'top_k': 10,
    'num_return_sequences': 1,
    'return_full_text': False,





}

GENERATOR_PARAMETERS = {
    'max_new_tokens': 1024,
    'do_sample': True,
    'top_k': 10,
    'num_return_sequences': 1,
    'return_full_text': False
}

# ----------------------- Summarizer inference utilities ----------------------- #

def load_template(template_path):
    '''
    Loads the JSON patient summary template from the path. 
    '''
    if not os.path.exists(template_path):
        raise FileNotFoundError(f'Template not found at {template_path}.')
    with open(template_path) as f:
        template = json.dumps(json.load(f), indent=4)
    return template

def complete_json(text): 
    ''' 
    Format a (potentially partial) JSON string. 
    Removes the last character until the string is valid.
    '''
    json_string = text
    while True:
        if not json_string:
            return None
        try:
            data = json.loads(json_string + '}')
        except json.decoder.JSONDecodeError:
            json_string = json_string[:-1]
            continue
        break
    return data

def check_summary(answer, prev_answer, template_path): 
    '''
    Temporary fix for limited context length. 
    
    Loads the JSON patient summary template from the path. 
    Given the (potentially partial) model output, check whether all fields are filled. 
    Otherwise, outputs the features that are missing. 
    '''
    if not os.path.exists(template_path):
        raise FileNotFoundError(f'Template not found at {template_path}.')
    with open(template_path) as f:
        template = json.load(f)

    answer = complete_json(answer)
    if answer is None:
        return False, template
    prev_answer.update(answer)

    missing = {}
    for key in template.keys():
        if key not in prev_answer.keys():
            missing[key] = template[key]
    valid = (len(missing) == 0)
    return valid, missing, prev_answer


# ----------------------- Running inference ----------------------- #
    

def generate(
        model_name,
        model_path, 
        data_path,
        output_path=None, 
        num_samples=None, 
        mode='summarizer',
        template_path=None):
    '''
    Loads a model and generates clinical notes. 
    Can be used for either 
    - Generator: Generate clinical notes from patient summaries
    - Direct: Generate clinical notes from patient-doctor conversations

    Arguments: 
        - model_name: Name of the model to be loaded.
        - model_path: Path to the model.
        - data_path: Path to the data file with dialog or patient summaries. 
        - output_path: Path to the output file with generated notes
        - num_samples: Number of samples to generate (default: None --> all)
        - mode: 'summarizer' or 'generator'
        - template_path: Path to the template file (.json), only for summarizer mode

    '''
    print(f"Loading model {model_name} from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found at {model_path}.')
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, use_cache=True, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_cache=False)
        print(f"Model is running on {torch.cuda.device_count()} GPUs.")
    except Exception as e:
        raise ValueError(f"Error when loading model and tokenizer from {model_path}:\n{e}")
    model.eval()

    print(f"\nLoading data from {data_path}...")
    dataset = load_file(data_path)
    dataset['model_name'] = model_name
    dataset['pred'] = ''

    print(f"\nInitalizing pipeline...")
    if mode == 'summarizer':
        gen_parameters = SUMMARIZER_PARAMETERS
        if template_path is None:
            raise ValueError(f"Template path must be specified for summarizer mode.")
        stop_word = '}\n}'
        stop_words_ids = tokenizer.encode(stop_word, add_prefix_space = False)
        stopping_criteria = StoppingCriteriaSub(stop_words_ids)

    elif mode == 'generator':
        gen_parameters = GENERATOR_PARAMETERS
        stopping_criteria = None

    else:
        raise ValueError(f"Invalid mode {mode}. Must be 'summarizer' or 'generator'.")
    
    pipe = pipeline("text-generation", 
                    model=model, 
                    tokenizer=tokenizer, 
                    eos_token_id=tokenizer.eos_token_id, 
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria,
                    **gen_parameters)
    
    if output_path is None:
        output_path = data_path.replace('.jsonl', f'-{model_name}.jsonl')

    for i, row in tqdm(dataset.iterrows(), total=len(dataset), 
                       desc=f"Generating answers from {model_name}"):
        
        if mode == 'generator':
            prompt = row['prompt'] + '\nNow, generate the corresponding clinical note: \n\n'
            print(f'\n\n### Prompt:\n\n{prompt}')
            answer = pipe(row['prompt'])[0]['generated_text']
            print(f'\n\n### Answer: \n\n{answer}')

        elif mode == 'summarizer': 

            prev_answer = {}
            valid = False
            missing = {}
            while not valid:
                if missing == {}:
                    prompt = row['prompt'] \
                        + '\nNow, generate the full patient summary: \n\n{\n'
                else: 
                    prompt = row['prompt'] \
                        + '\nNow, generate the patient summary for the following features only: \n\n' \
                        + formatting(json.dumps(missing)) \
                        + '\n\nPatient summary: \n\n{\n'
                    
                print(f'\n\n### PROMPT:\n\n{prompt}')
                partial_answer = '{\n'+pipe(prompt)[0]['generated_text']
                print(f'\n\n### PARTIAL ANSWER:\n\n{partial_answer}\n\n')
                valid, missing, prev_answer = check_summary(partial_answer, prev_answer, template_path)
                print('\n\n\nCHECKING SUMMARY')
                print(f'\n\n### VALID: {valid}')
                print(f'\n\n### MISSING {len(missing)} features:\n\n{missing} \n\n')
            answer = prev_answer
            print(f'\n\nFINAL ANSWER: \n\n{answer}')
        
        dataset.loc[i, 'pred'] = answer
        #if i % 10 == 0: 
            #save_file(dataset, output_path)
        if num_samples and i >= num_samples:
            break
    #save_file(dataset, output_path)
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
    parser.add_argument('--num_samples',
                        type=int,
                        default=None,
                        help='Number of samples to generate')
    parser.add_argument('--mode',
                        type=str,
                        default='summarizer',
                        help='Mode of inference: summarizer or generator')
    parser.add_argument('--template_path',
                        type=str,
                        default='data/template.json',
                        help='Path to the template file (.json), only for summarizer mode')
    args = parser.parse_args()
    generate(args.model_name, args.model_path, args.data_path, 
             args.output_path, args.num_samples, args.mode, args.template_path)
