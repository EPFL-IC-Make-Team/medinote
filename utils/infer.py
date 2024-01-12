'''
Inference pipeline.

Inference modes supported: 
    - summarizer:       dialogue -> patient summary
    - generator:        patient summary (by custom model) -> clinical note
    - generator-gpt:    patient summary (by GPT-4) -> clinical note
    - direct:           dialogue -> clinical note
    - direct-gpt:       dialogue -> clinical note (with GPT-4)
'''

import torch
import os
import re
import argparse
import sys
import os
import time
import numpy as np
import vllm
import json as json
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from utils.data import *
from utils.chat import *

# ----------------------- Constants ----------------------- #

BOS_TOKEN, EOS_TOKEN = '<|im_start|>', '<|im_end|>'
TODO_VAL = -1

KEYS = {
    'meditron-7b-summarizer': {
        'input': 'conversation',
        'output': 'pred_summary',
        'combined_output' : 'pred_summary_7b',
        'gold': 'summary'
    },

    'meditron-13b-summarizer': {
        'input': 'conversation',
        'output': 'pred_summary',
        'combined_output' : 'pred_summary_13b',
        'gold': 'summary'
    },

    'generator': {
        'input': 'pred_summary',
        'output': 'pred_note',
        'combined_output' : 'pred_note',
        'gold': 'data'
    },
    'generator-gpt': {
        'input': 'summary',
        'output': 'pred_note-gpt',
        'combine_output' : 'pred_note-gpt',
        'gold': 'data'
    },
    'direct': {
        'input': 'conversation',
        'output': 'pred_direct',
        'combined_output' : 'pred_direct',
        'gold': 'data'
    },
    'direct-gpt': {
        'input': 'conversation',
        'output': 'pred_direct-gpt',
        'combined_output' : 'pred_direct-gpt',
        'gold': 'data'
    }
}

INSTRUCTIONS = {
    'summarizer': [
        """Given the provided patient-doctor dialogue, write the corresponding patient information summary in JSON format.
Make sure to extract all the information from the dialogue into the template, but do not add any new information. 
If a field is not mentioned, simply write \"feature\": \"None\".""",
        'Now, generate the full patient summary: '
    ],
    'generator': [
        """Given the provided JSON patient information summary, generate the corresponding clinical note as written by a physician.
Make sure to use all the information from the dialogue into the note, but do not add any new information.""",
        'Now, generate the corresponding clinical note: '
    ],
    'generator-gpt': [
        """Given the provided JSON patient information summary, generate the corresponding clinical note as written by a physician.
Make sure to use all the information from the dialogue into the note, but do not add any new information.""",
        'Now, generate the corresponding clinical note: '
    ],
    'generator-gpt': [
        """Given the provided JSON patient information summary, generate the corresponding clinical note as written by a physician.
        Make sure to use all the information from the dialogue into the note, but do not add any new information.""",
        'Now, generate the corresponding clinical note: '
    ],
    'direct': [
        """Given the provided patient-doctor conversation, generate the corresponding clinical note as written by the physician.
Make sure to use all the information from the dialogue into the note, but do not add any new information.""",
        'Now, generate the corresponding clinical note: '
    ]
}

# ----------------------- Inference parameters ----------------------- #

GREEDY_PARAMETERS = {
    'best_of': 1,
    'presence_penalty': 0.0,
    'frequency_penalty': 1.0,
    'top_k': -1,
    'top_p': 1.0,
    'temperature': 0.0,
    'stop': EOS_TOKEN,
    'use_beam_search': False,
    'max_tokens': 2048
}

PARAMETERS = {
    'summarizer': GREEDY_PARAMETERS,
    'generator': GREEDY_PARAMETERS,
    'generator-gpt': GREEDY_PARAMETERS,
    'direct': GREEDY_PARAMETERS
}

# ----------------------- Inference utilities ----------------------- #

class PandasDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        return self.dataframe.iloc[index]
        

def combine(input_path, output_path):
    '''
    Combine the inferred data into a single file.
    '''
    paths = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.jsonl')]
    files = list({path.split('/')[-1].split('.')[0]: load_file(path) for path in paths}.items())
 
    commom_columns = list(set.intersection(*[set(file[1].columns) for file in files]))
    
    combined_df = files[0][1].dropna()
    len_ = combined_df.shape[0]
    combined_df = combined_df.rename(columns={KEYS[files[0][0]]['output']: KEYS[files[0][0]]['combined_output']})

    for name,df in files[1:]:
        df = df.rename(columns={KEYS[name]['output']: KEYS[name]['combined_output']})
        combined_df = pd.merge(combined_df, df.dropna(), on=commom_columns, how='inner')
    
    if len(combined_df) < len_:
        raise ValueError(f'Combined dataframe has less rows than the first dataframe.')

    save_file(combined_df, output_path, mode='w')

    return combined_df

def todo_list(data_df, gen_df, input_key, output_key, num_samples=None):
    '''
    Returns the list of samples to generate.

    :param data_df: pd.DataFrame, the input data
    :param gen_df: pd.DataFrame, the generated data
    :param input_key: str, remove samples for which the input key is None
    :param output_key: str, remove samples for which the output key has already been generated in gen_df
    :param num_samples: int, keep only the first num_samples samples (default: None --> all)
    :return: list, the list of indices to generate
    '''
    if input_key not in data_df.columns:
        raise ValueError(f'Input key {input_key} not found in input file.')
    valid_data = data_df[data_df[input_key].notnull()]
    idx_todo = valid_data['idx'].tolist()
    if num_samples and len(idx_todo) > num_samples:
        idx_todo = idx_todo[:num_samples]
    idx_done = gen_df[gen_df[output_key].notnull()]['idx'].tolist()
    idx_todo = [i for i in idx_todo if i not in idx_done]
    if len(idx_todo) == 0:
        raise ValueError(f'All samples already generated.')
    return idx_todo
    
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

def format_prompt(prompt, mode, instructions):
    """
    Format prompt for inference as follows:

    <|im_start|>
    First instructions.

    Input text.

    Second instructions.
    <|im_end|>
    <|im_start|> 
    """
    if 'generator' in mode:
        prompt = '\n'.join([line for line in prompt.split('\n') if ': \"None\"' not in line])
    prompt = f"{BOS_TOKEN}question\n{instructions[0]}\n\n{prompt}\n\n{instructions[1]}{EOS_TOKEN}\n{BOS_TOKEN}answer\n"
    return prompt

def infer_vllm(client, mode, prompt):
    """
    Inference using the VLLM backend (offline mode). 
    Returns the output text.

    Reference: https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py

    :param client: vllm.LLM, the LLM offline generation engine to use for querying the VLLM backend
    :param mode: str, the mode to use for inference
    :param prompt: str, the prompt to generate from
    """
    sampling_params = vllm.SamplingParams(**PARAMETERS[mode])
    response = client.generate(prompt, sampling_params=sampling_params)
    if len(response) > 0:
        return [r.outputs[0].text for r in response]
    else:
        return response[0].outputs[0].text

# ----------------------- OpenAI inference ----------------------- #

def load_few_shot(train_path, shots=1):
    '''
    Load a few-shot example from the training data for direct-gpt inference.
    
    :param train_path: str, path to the training data file. If None --> no few-shot example.
    :param shots: int, number of few-shot examples to load
    '''
    if train_path is not None and shots > 0:
        print(f'Loading {shots}-shot exemplar from {train_path}...')
        train_df = load_file(train_path)
        sample = train_df.sample(shots)
        few_shot_prompt = f"Here are {shots} example(s) of patient-doctor conversations and their corresponding clinical notes.\n\n"
        for i in range(shots):
            dialogue = sample.iloc[i]['conversation']
            note = sample.iloc[i]['data']
            few_shot_prompt += f'Example {i+1}:\n\nConversation:\n\n{dialogue}\n\nClinical note:\n\n{note}\n\n'
    else: 
        few_shot_prompt = 'Your answer should consist in one or a few paragrpahs of text, not overstructured.'
    return few_shot_prompt


def infer_openai(input_path,
                 output_path,
                 train_path = None,
                 max_tokens = 1000000,
                 num_samples = 1000,
                 openai_model = 'gpt-3.5-turbo',
                 temperature = 1.0,
                 shots = 1):
    '''
    Generate clinical notes from conversations using an OpenAI model.

    :param input_path: str, path to the input data file
    :param output_path: str, path to the output data file
    :param train_path: str, path to the training data file. If None --> no few-shot example.
    :param max_tokens: int, maximum number of tokens to generate
    :param num_samples: int, number of samples to generate
    :param openai_model: str, name of the OpenAI model to use (gpt-3.5-turbo, gpt-4)
    :param temperature: float, temperature for generation
    :param shots: int, number of few-shot examples to load 
    '''
    input_key = KEYS['direct-gpt']['input']
    output_key = KEYS['direct-gpt']['output']
    
    print("Loading data...")
    data_df = load_file(input_path)
    if 'idx' not in data_df.columns:
        data_df['idx'] = data_df.index
    data_df = data_df.reset_index(drop=True)
    if input_key not in data_df.columns:
        raise ValueError(f'Input key {input_key} not found in output file.')

    print(f"Loading output file from {output_path}...")
    if os.path.exists(output_path):
        gen_df = load_file(output_path)
    else:
        gen_df = pd.DataFrame(columns = data_df.columns)
    if output_key not in gen_df.columns:
        gen_df[output_key] = None
    idx_todo = todo_list(data_df, gen_df, output_key, num_samples=num_samples)
    data_df = data_df[data_df['idx'].isin(idx_todo)]

    few_shot_prompt = load_few_shot(train_path, shots=shots)

    print("Loading model...")
    if openai_model == 'gpt-3.5-turbo':
        price_per_1ktokens = 0.001
        chat = chat_gpt_3
    elif openai_model == 'gpt-4':
        chat = chat_gpt_4_turbo
        price_per_1ktokens = 0.01
    else:
        raise ValueError(f'OpenAI model {openai_model} not found.')
    
    instruction, usr_prompt = INSTRUCTIONS['direct']
    prompts = [(f"{instruction}\n\n{few_shot_prompt}\n\n{dialogue}", usr_prompt) 
               for dialogue in tqdm(data_df[input_key].tolist(), desc='Building prompts')]
    data_df['messages'] = [build_messages(*prompt) for prompt in prompts]
    
    sub_batches = partition(
        dataframe = data_df,
        max_token_per_partition=max_tokens,
        model = openai_model
    )

    total_tokens = np.sum([nb_tok for _, nb_tok in sub_batches])
    print(f"Total input tokens: {total_tokens}, total input cost: {total_tokens/1000 * price_per_1ktokens}$")

    for i, (sub_batch_df, nb_tokens) in enumerate(sub_batches):
        print(f"Sub_batch {i+1}/{len(sub_batches)}: {sub_batch_df.shape[0]} calls, {nb_tokens} total tokens: {nb_tokens/1000 * price_per_1ktokens}$")
        
        start_time = time.time()
        answers = generate_answers(
            messages_list = sub_batch_df['messages'].tolist(),
            formatting = lambda x: x,
            chat = chat,
            temperature = temperature
        )

        sub_batch_df[output_key] = answers
        sub_batch_df['model_name'] = openai_model
        sub_batch_df.drop(columns = ['messages'], inplace = True)
        gen_df = pd.concat([gen_df, sub_batch_df], ignore_index = True)
        save_file(gen_df, output_path, mode='w')
        print(f'\nSub-batch {i+1} Saved (size {sub_batch_df.shape[0]})\n')
        delete_pickle_file("safety_save.pkl")
        if i == len(sub_batches) - 1:
            break
        end_time = time.time()
        time_taken = (end_time - start_time)
        breaktime = max(int(60 - time_taken) + 2, 5) 
        print(f"\nBreak for {breaktime} seconds.")
        time.sleep(breaktime)

    return gen_df

# ----------------------- Summary inference ----------------------- #

def complete_json(text): 
    ''' 
    Format a (potentially partial) JSON string. 
    Removes the last character until the string is valid.
    '''
    json_string = text.replace('\n', '')
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

def infer_summary(dialogue, 
                  client, 
                  template, 
                  instructions, 
                  max_tries=3): 
    '''
    Generates a patient summary from a patient-doctor conversation.
    If the generated summary is incomplete, query the model again with the missing fields.

    :param dialogue: str, patient-doctor conversation
    :param client: vllm.LLM, the LLM offline generation engine to use for querying the VLLM backend
    :param template: dict, the template for the patient summary
    :param instructions: list, the instructions to generate the summary
    :param max_tries: int, maximum number of tries to generate a valid summary
    :param verbose: bool, whether to print prompts and answers
    '''
    current_answer = {}
    valid = False
    missing = {}
    while not valid and max_tries > 0:
        if missing == {}:
            starter = '{\n"visit motivation":'
            prompt = f'{instructions[0]}\n\n{dialogue}\n\n{instructions[1]}\n\n{starter}'
        else: 
            starter = '{\n"' + f'{list(missing.keys())[0]}":'
            missing_dict = formatting(json.dumps(missing, indent=4))
            prompt_end = f'\n\nNow, fill in the following template: \n\n{missing_dict}\n\n{starter}'
            prompt = f'{BOS_TOKEN}\n{instructions[0]}\n\n{dialogue}{prompt_end}\n{EOS_TOKEN}\n{BOS_TOKEN} '
        partial_answer = starter + infer_vllm(client, 'summarizer', prompt)
        limiter = re.search(r'}\s*}', partial_answer)
        if limiter: partial_answer = partial_answer[:limiter.end()].strip()
        valid, missing, current_answer = check_summary(partial_answer, current_answer, template)
        max_tries -= 1
    answer = json.dumps(current_answer, indent=4)
    return answer


# ----------------------- Inference ----------------------- #

def infer(
        model_name,
        model_path, 
        input_path=None,
        output_path=None, 
        num_samples=None, 
        mode='summarizer',
        template_path=None,
        verbose=False):
    '''
    Loads a model and generates clinical notes. 
    Can be used for either 
    - Generator: Generate clinical notes from patient summaries
    - Direct: Generate clinical notes from patient-doctor conversations

    Arguments: 
        - model_name: Name of the model to be loaded.
        - model_path: Path to the model and tokenizer.
        - input_path: Path to the data file with columns {idx, data, conversation, full_note}.
        - output_path: Path to the output file with generated notes.
        - num_samples: Number of samples to generate (default: None --> all)
        - mode: 
            summarizer      -> generate summaries from conversations
            generator       -> generate notes from summarizer's summaries
            generator-gpt   -> generate notes from GPT-4's summaries
            direct          -> generate notes directly from conversations
        - template_path: Path to the template file (.json), only for summarizer mode
    '''

    print(f"\n\n# ----- INFERENCE: mode = {mode}, model = {model_name} ----- #\n\n")
    instructions = INSTRUCTIONS[mode]
    if mode == 'summarizer':
        input_key, output_key = KEYS[model_name]['input'], KEYS[model_name]['output']
    else:
        input_key, output_key = KEYS[mode]['input'], KEYS[mode]['output']

    data_df = load_file(input_path)
    print(f"\nLoaded data file with {data_df.shape[0]} samples and columns {list(data_df.columns)}...")
    if 'idx' not in data_df.columns:
        data_df['idx'] = data_df.index
    data_df = data_df.reset_index(drop=True)
    template = None if mode != 'summarizer' else load_template(template_path)

    if os.path.exists(output_path):
        gen_df = load_file(output_path)
        print(f"Loading output file with {gen_df.shape[0]} samples...")
    else:
        print(f"Creating output file at {output_path} with columns {list(data_df.columns)}...")
        gen_df = pd.DataFrame(columns = data_df.columns)
        gen_df[output_key] = TODO_VAL

    idx_todo = todo_list(data_df, gen_df, input_key, output_key, num_samples)
    print(f"{len(idx_todo)} samples to generate.")
    
    data_df = data_df[data_df['idx'].isin(idx_todo)]
    batch_size = 1 if mode == 'summarizer' else 2
    inference_data = json.loads(data_df.to_json(orient='records'))
    data_loader = DataLoader(inference_data, batch_size=batch_size, shuffle=False)
    print(f"Created data loader: {len(data_loader)} batches to generate with batch size {batch_size}.")

    print(f"Initializing vLLM client...")
    kwargs = {
        "model": model_path,
        "tokenizer": model_path,
        "trust_remote_code": True,
        "max_num_seqs": 2048,
        "tensor_parallel_size": torch.cuda.device_count(),
    }
    client = vllm.LLM(**kwargs)

    for batch in tqdm(data_loader, total=len(data_loader), position=0, leave=True):
        if mode == 'summarizer':
            prompts = batch[input_key]
            answers = [infer_summary(prompts[0], client, template, instructions, verbose=verbose)]
        else: 
            prompts = [format_prompt(prompt, mode, instructions) for prompt in batch[input_key]]
            answers = infer_vllm(client, mode, prompts)

        if verbose:
            for prompt, answer in zip(prompts, answers):
                print(f'\n\n### PROMPT:\n\n{prompt}')
                print(f'\n\n### ANSWER:\n\n{answer}')

        new_batch = pd.DataFrame(batch)
        new_batch[output_key] = answers
        gen_df = pd.concat([gen_df, new_batch], ignore_index = True)
        save_file(gen_df, output_path, mode='w')
            

# ----------------------- Main ----------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', 
                        type=str, 
                        required=True,
                        help='Model name to be loaded.')
    parser.add_argument('--model_path', 
                        type=str, 
                        default=None,
                        help='Path to the model.')
    parser.add_argument('--input_path', 
                        type=str, 
                        required=True,
                        help='Path to the data file.')
    parser.add_argument('--output_path', 
                        type=str,
                        required=True,
                        help='Path to the output file with generated notes. ')
    parser.add_argument('--num_samples',
                        type=int,
                        default=None,
                        help='Number of samples to generate')
    parser.add_argument('--template_path',
                        type=str,
                        default='data/template.json',
                        help='For summarizer mode only: path to the patient summary template.')
    parser.add_argument('--train_path',
                        type=str,
                        default=None,
                        help='Path to the training data file. Used to sample few-shot examples for direct-gpt inference.')
    parser.add_argument('--mode',
                        type=str,
                        default='summarizer',
                        choices=['summarizer', 'generator', 'generator-gpt', 'direct', 'direct-gpt'],
                        help='Mode of inference: summarizer, generator, generator-gpt, direct, direct-gpt')
    parser.add_argument('--verbose',
                        type=int,
                        default=1,
                        help='Whether to print prompts and answers')
    parser.add_argument('--combine',
                        action='store_true',
                        default=False,
                        help='Whether to combine the generated notes into a single file.')
    args = parser.parse_args()

    if args.combine: 
        combine(input_path=args.input_path, 
                output_path=args.output_path)

    elif args.mode == 'direct-gpt':     
        infer_openai(
            input_path=args.input_path,
            output_path=args.output_path,
            train_path=args.train_path,
            openai_model=args.model_name,
            num_samples=args.num_samples
        )
    else:
        infer(
            model_name=args.model_name,
            model_path=args.model_path,
            input_path=args.input_path,
            output_path=args.output_path,
            num_samples=args.num_samples,
            template_path=args.template_path,
            mode=args.mode,
            verbose=args.verbose
        )
