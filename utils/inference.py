'''
Inference utilities. 
'''

import torch
import os
import re
import argparse
import json as json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline, StoppingCriteria, StoppingCriteriaList
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from utils.data import *

# ----------------------- Constants ----------------------- #

KV_PAIRS = {
    'summarizer': {
        'input': 'conversation',
        'output': 'pred_summary',
    },
    'generator': {
        'input': 'pred_summary',
        'output': 'pred_note',
    },
    'direct': {
        'input': 'conversation',
        'output': 'pred_direct',
    }
}

INSTRUCTIONS = {
    'summarizer': [
        'Given the provided patient-doctor dialogue, write the corresponding patient information summary in JSON format.\nMake sure to extract all the information from the dialogue into the template, but do not add any new information. \nIf a field is not mentioned, simply write \"feature\": \"None\".',
        'Now, generate the full patient summary: '
    ],
    'generator': [
        'Given the provided JSON patient information summary, generate the corresponding clinical note as written by a physician.\nMake sure to use all the information from the dialogue into the note, but do not add any new information.',
        'Now, generate the corresponding clinical note: '
    ],
    'direct': [
        'Given the provided patient-doctor conversation, generate the corresponding clinical note as written by the physician. \nMake sure to use all the information from the dialogue into the note, but do not add any new information.',
        'Now, generate the corresponding clinical note: '
    ]
}

PARAMETERS = {
    'summarizer' : {
        'max_length': 2048,
        'do_sample': True,
        'num_beams': 1,
        'top_p': 0.95,
        'num_return_sequences': 1,
        'return_full_text': False,
        'max_time': 300,
    },
    'generator' : {
        'max_length': 2048, 
        'do_sample': True,
        'num_beams': 1,
        'top_p': 0.95,
        'num_return_sequences': 1,
        'return_full_text': False
    },
    'direct' : {
        'max_length': 2048,
        'do_sample': True,
        'num_beams': 1,
        'top_p': 0.95,
        'num_return_sequences': 1,
        'return_full_text': False
    }
}


# ----------------------- Summarizer inference utilities ----------------------- #

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = []):
      StoppingCriteria.__init__(self), 

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops = []):
      self.stops = stops
      for i in range(len(stops)):
        self.stops = self.stops[i]

def load_template(template_path):
    '''
    Loads the JSON patient summary template from the path. 
    '''
    if not template_path:
        raise ValueError(f'Template path must be specified for summarizer mode.')
    elif not os.path.exists(template_path):
        raise FileNotFoundError(f'Template not found at {template_path}.')
    with open(template_path) as f:
        template = json.load(f)
    return template

def complete_json(text): 
    ''' 
    Format a (potentially partial) JSON string. 
    Removes the last character until the string is valid.
    '''
    json_string = text.replace('\n', '')
    num_tries = 0
    while True:
        num_tries += 1
        if not json_string:
            return None
        try:
            data = json.loads(json_string + '}')
        except json.decoder.JSONDecodeError:
            json_string = json_string[:-1]
            continue
        break
    print(f'JSON string completed in {num_tries} tries.')
    return data

def check_summary(answer, prev_answer, template): 
    '''
    Temporary fix for limited context length. 
    
    Loads the JSON patient summary template from the path. 
    Given the (potentially partial) model output, check whether all fields are filled. 
    Otherwise, outputs the features that are missing. 
    '''
    # Convert answer to a complete JSON dictionary
    answer = complete_json(answer)
    if answer is None:
        return False, template, prev_answer
    
    # Merge with existing answer
    for key in answer.keys():
        if key not in prev_answer.keys():
            prev_answer[key] = answer[key]

    # Check if all fields are filled
    missing = {key: template[key] for key in template.keys() if key not in prev_answer.keys()}
    valid = (len(missing) == 0)
    return valid, missing, prev_answer


# ----------------------- Inference pipeline ----------------------- #
    

def infer_summary(dialogue, 
                  pipe, 
                  template, 
                  instructions, 
                  max_tries=3, 
                  verbose=False): 
    '''
    Generates a patient summary from a patient-doctor conversation.
    If the generated summary is incomplete, query the model again with the missing fields.
    '''
    current_answer = {}
    valid = False
    missing = {}

    while not valid and max_tries > 0:
        if missing == {}:
            starter = '{\n"visit motivation": '
            prompt = instructions[0] + '\n\n' + dialogue + '\n\n' + instructions[1] + '\n\n' + starter
        else: 
            starter = '{'
            if missing != {}:
                starter += f'\n"{list(missing.keys())[0]}": '

            prompt = instructions[0] + '\n\n' + dialogue \
                + '\n\nNow, fill in the following template: \n\n' \
                + formatting(json.dumps(missing, indent=4)) \
                + '\n\n' + starter
        if verbose: print(f'\n\n### PROMPT:\n\n{prompt}')
        partial_answer = starter + pipe(prompt)[0]['generated_text'].strip()
        limiter = re.search(r'}\s*}', partial_answer)
        if limiter: partial_answer = partial_answer[:limiter.end()].strip()
        valid, missing, current_answer = check_summary(partial_answer, current_answer, template)
        max_tries -= 1
    if not valid:
        if verbose: print(f'Could not generate a valid summary in {max_tries} tries.')
        return None
    answer = json.dumps(current_answer, indent=4)
    return answer


def infer(
        model_name,
        model_path, 
        input_path=None,
        output_path=None, 
        num_samples=None, 
        mode='summarizer',
        template_path=None,
        use_gpt_summary=False,
        verbose=False):
    '''
    Loads a model and generates clinical notes. 
    Can be used for either 
    - Generator: Generate clinical notes from patient summaries
    - Direct: Generate clinical notes from patient-doctor conversations

    Arguments: 
        - model_name: Name of the model to be loaded.
        - model_path: Path to the model.
        - input_path: Path to the data file with dialog or patient summaries. 
        - output_path: Path to the output file with generated notes
        - num_samples: Number of samples to generate (default: None --> all)
        - mode: 'summarizer' or 'generator'
        - template_path: Path to the template file (.json), only for summarizer mode
        - use_gpt_summary: Whether to use GPT-4 summaries as input, only for generator mode
    '''
    print(f"\n\n# ----- INFERENCE: mode = {mode}, model = {model_name} ----- #\n\n")
    # Load model
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found at {model_path}.')
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, use_cache=True, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_cache=False)
        print(f"Model is running on {torch.cuda.device_count()} GPUs.")
    except Exception as e:
        raise ValueError(f"Error when loading model and tokenizer from {model_path}:\n{e}")
    model.eval()

    # Load parameters
    if mode not in ['summarizer', 'generator', 'direct']:
        raise ValueError(f"Invalid mode {mode}. Must be 'summarizer', 'generator' or 'direct'.")
    instructions = INSTRUCTIONS[mode]
    input_key = KV_PAIRS[mode]['input']
    output_key = KV_PAIRS[mode]['output']
    if mode == 'generator' and use_gpt_summary:
        input_key = 'summary'
        model_name += '-gpt'
    gen_parameters = PARAMETERS[mode]
    print(f"\n\n### PARAMETERS:\n\nInstruction 1: {instructions[0]}\nInstruction 2: {instructions[1]}\nInput key: {input_key}\nOutput key: {output_key}\nParameters: {gen_parameters}")

    # Load generation pipeline
    print(f"\nInitalizing pipeline...")
    stopping_criteria = None
    if mode == 'summarizer':
        template = load_template(template_path)
        stoppers = ['visit motivation']
        stops = tokenizer(stoppers, add_special_tokens=False)['input_ids']
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stops)])
    pipe = pipeline("text-generation", 
                            model=model, 
                            tokenizer=tokenizer, 
                            eos_token_id=tokenizer.eos_token_id, 
                            pad_token_id=tokenizer.eos_token_id,
                            stopping_criteria=stopping_criteria,
                            **gen_parameters)
    
    # Load data
    if input_path: 
        print(f"\nLoading input file from {input_path}...")
        data_df = load_file(input_path)
        if 'idx' not in data_df.columns:
            data_df['idx'] = data_df.index
    elif not output_path: 
        raise ValueError(f"Input path must be specified if output path is not.")
    print('Dataset columns: ', data_df.columns)

    # Load output file
    if os.path.exists(output_path):
        print(f"Loading output file from {output_path}...")
        gen_df = load_file(output_path)
    else:
        print(f"Initializing output file at {output_path}...")
        gen_df = data_df.copy()
        gen_df[output_key] = None
        gen_df['model_name'] = model_name
    
    # Check which samples to generate
    idx_done = gen_df[gen_df[output_key].notnull()]['idx'].tolist()
    idx_todo = [i for i in gen_df.index if i not in idx_done]
    if mode == 'generator' and not use_gpt_summary:
        if 'pred_summary' not in gen_df.columns:
            raise ValueError(f'No patient summaries found in {input_path}.')
        idx_todo = [i for i in idx_todo if gen_df.loc[i]['pred_summary'] is not None]
        print(f"Found {len(idx_todo)} generated summaries.")
        if len(idx_todo) == 0:
            raise ValueError(f'No patient summaries found in {input_path}.')
    if num_samples and len(idx_todo) > num_samples:
        idx_todo = idx_todo[:num_samples]

    # Generate samples
    for i in tqdm(idx_todo, desc='Generating samples'):
        row = gen_df.loc[i]

        if mode == 'generator' or mode == 'direct':
            query = instructions[0] + '\n\n' + row[input_key] + '\n\n' + instructions[1]
            if verbose: print(f'\n\n### PROMPT:\n\n{query}')
            answer = pipe(query)[0]['generated_text']

        elif mode == 'summarizer':
            answer = infer_summary(row[input_key], pipe, template, instructions, verbose=verbose)
        if verbose: print(f'\n\n### ANSWER: \n\n{answer}')
        
        gen_df.at[i, output_key] = answer
        gen_df.at[i, 'model_name'] = model_name
        save_file(gen_df, output_path)
    return gen_df
    

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
    parser.add_argument('--input_path', 
                        type=str, 
                        default=None,
                        help='Path to the data file.')
    parser.add_argument('--output_path', 
                        type=str, 
                        help='Path to the output file with generated notes. ')
    parser.add_argument('--num_samples',
                        type=int,
                        default=None,
                        help='Number of samples to generate')
    parser.add_argument('--template_path',
                        type=str,
                        default='data/template.json',
                        help='For summarizer mode only: path to the patient summary template')
    parser.add_argument('--mode',
                        type=str,
                        default='summarizer',
                        help='Mode of inference: summarizer, generator or direct')
    parser.add_argument('--use_gpt_summary', 
                        action='store_true',
                        default=False,
                        help='For generator mode only: whether to use GPT-4 summaries as input')
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Whether to print prompts and answers')
    args = parser.parse_args()
    infer(**vars(args))
