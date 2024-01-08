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
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from utils.data import *
from utils.chat import *

# ----------------------- Constants ----------------------- #

BOS_TOKEN = '<|im_start|>'
EOS_TOKEN = '<|im_end|>'

KV_PAIRS = {
    'meditron-7b-summarizer': {
        'input': 'conversation',
        'output': 'pred_summary',
    },
    'meditron-13b-summarizer': {
        'input': 'conversation',
        'output': 'pred_summary',
    },
    'generator': {
        'input': 'pred_summary',
        'output': 'pred_note',
    },
    'generator-gpt': {
        'input': 'summary',
        'output': 'pred_note-gpt',
    },
    'direct': {
        'input': 'conversation',
        'output': 'pred_direct',
    },
    'direct-gpt': {
        'input': 'conversation',
        'output': 'pred_direct-gpt',
    }
}

gold_note_column = 'data'
gold_summary_column = 'summary'

GP_PAIRS = {
    'meditron-7b-summarizer': {
        'gold': gold_summary_column
    },
    'meditron-13b-summarizer': {
        'gold': gold_summary_column
    },
    'generator': {
        'gold': gold_note_column
    },
    'generator-gpt': {
        'gold': gold_note_column
    },
    'direct': {
        'gold': gold_note_column
    },
    'direct-gpt': {
        'gold': gold_note_column,
    }}

for mode in KV_PAIRS.keys():
    GP_PAIRS[mode]['pred'] = KV_PAIRS[mode]['output']

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

# Default parameters for inference/evaluation. 

PARAMETERS = {
    'summarizer' : {
        'max_length': 2048,
        'do_sample': False,
        'num_return_sequences': 1,
        'return_full_text': False
    },
    'generator' : {
        'max_length': 2048, 
        'do_sample': True,
        'temperature': 0.7,
        'num_return_sequences': 1,
        'return_full_text': False
    },
    'generator-gpt' : {
        'max_length': 2048, 
        'do_sample': True,
        'temperature': 0.7,
        'num_return_sequences': 1,
        'return_full_text': False
    },
    'direct' : {
        'max_length': 2048,
        'do_sample': True,
        'temperature': 0.7,
        'num_return_sequences': 1,
        'return_full_text': False
    }
}

# ----------------------- Inference utilities ----------------------- #

'''def combine(input_path, output_path):
    
    Combine the inferred data into a single file.
    
    paths = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.jsonl')]
    files = {path.split('/')[-1].split('.')[0]: load_file(path) for path in paths}
    for source, df in files.items():
        output_key = [key for key in KV_PAIRS[source].values() if key in df.columns][0]
        df[output_key] = df[output_key].apply(lambda x: f'{source}: {x}')
        files[source] = df
    combined_df = pd.concat(files.values(), ignore_index=True)
    save_file(combined_df, output_path, mode='w')
    return combined_df'''

def combine(input_path, output_path):
    '''
    Combine the inferred data into a single file.
    '''
    paths = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.jsonl')]
    files = {path.split('/')[-1].split('.')[0]: load_file(path) for path in paths}
    dfs = list(files.values())
    commom_columns = list(set.intersection(*[set(df.columns) for df in dfs]))
    
    combined_df = dfs[0].dropna()
    len_ = combined_df.shape[0]

    for name,df in list(files.items())[1:]:
        combined_df = pd.merge(combined_df, df.dropna(), on=commom_columns, how='inner', suffixes=('',f'_{name}'))
    
    if len(combined_df) < len_:
        raise ValueError(f'Combined dataframe has less rows than the first dataframe.')

    save_file(combined_df, output_path, mode='w')

    return combined_df
    
def todo_list(data_df, gen_df, output_key, num_samples):
    '''
    Returns the list of samples to generate. 
    '''
    idx_todo = data_df['idx'].tolist()
    if num_samples and len(idx_todo) > num_samples:
        idx_todo = idx_todo[:num_samples]
    idx_done = gen_df[gen_df[output_key].notnull()]['idx'].tolist()
    idx_todo = [i for i in idx_todo if i not in idx_done]
    if len(idx_todo) == 0:
        raise ValueError(f'All samples already generated.')
    if 'input_key' in gen_df.columns:
        idx_todo = [i for i in idx_todo if gen_df[gen_df['idx'] == i]['input_key'].iloc[0] is not None]
    if len(idx_todo) == 0:
        raise ValueError(f'Samples left to generate but their input is None.')
    return idx_todo

class CustomStoppingCriteria(StoppingCriteria):

    def __init__(self, stops = []):
        self.stops = stops 

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops = []) -> bool:
        for stop in self.stops:
            last_ids = input_ids[:,-len(stop):].tolist()
            if stop in last_ids:
                return True
        return False


# ----------------------- Summarizer inference utilities ----------------------- #
    
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
    #print(f'\n\n### PARTIAL ANSWER: \n\n{json.dumps(answer, indent=4)}')
    
    # Merge with existing answer
    for key in answer.keys():
        if key not in prev_answer.keys():
            prev_answer[key] = answer[key]

    # Check if all fields are filled
    missing = {key: template[key] for key in template.keys() if key not in prev_answer.keys()}
    valid = (len(missing) == 0)
    return valid, missing, prev_answer


def load_model(model_path):
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found at {model_path}.')
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, use_cache=True, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            use_cache=False)
        print(f"Model is running on {torch.cuda.device_count()} GPUs.")
    except Exception as e:
        raise ValueError(f"Error when loading model and tokenizer from {model_path}:\n{e}")
    model.eval()

    # Check model embedding size
    if len(tokenizer) != model.config.vocab_size:
        print(f"Resizing model embedding layer from {model.config.vocab_size} to {len(tokenizer)}...")
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


# ----------------------- Inference pipeline ----------------------- #
    
def infer_openai(
        input_path,
        output_path,
        train_path = None,
        max_tokens = 1000000,
        num_samples = 1000, 
        openai_model = 'gpt-3.5-turbo',
        temperature = 0,
        shots = 1):
    '''
    Generate clinical notes from conversations using an OpenAI model. 
    '''
    input_key = KV_PAIRS['direct-gpt']['input']
    output_key = KV_PAIRS['direct-gpt']['output']
    
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
    idx_todo = todo_list(data_df, gen_df, output_key, num_samples)
    data_df = data_df[data_df['idx'].isin(idx_todo)]

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
            print("End of inference.")
            break

        end_time = time.time()
        time_taken = (end_time - start_time)
        breaktime = max(int(60 - time_taken) + 2, 5) 
        print(f"\nBreak for {breaktime} seconds.")
        time.sleep(breaktime)

    return gen_df

def infer_summary(dialogue, 
                  generate, 
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
            starter = '{\n"visit motivation":'
            prompt = f'{instructions[0]}\n\n{dialogue}\n\n{instructions[1]}\n\n{starter}'
            if verbose: print(f'\n\n### PROMPT:\n\n{prompt}')
        else: 
            starter = '{\n"' + f'{list(missing.keys())[0]}":'
            missing_dict = formatting(json.dumps(missing, indent=4))
            prompt_end = f'\n\nNow, fill in the following template: \n\n{missing_dict}\n\n{starter}'
            prompt = f'{BOS_TOKEN}\n{instructions[0]}\n\n{dialogue}{prompt_end}\n{EOS_TOKEN}\n{BOS_TOKEN} '
            #if verbose: print(f'\n\n### PROMPT:\n\n{prompt_end}')
        partial_answer = starter + generate(prompt).strip()
        limiter = re.search(r'}\s*}', partial_answer)
        if limiter: partial_answer = partial_answer[:limiter.end()].strip()
        valid, missing, current_answer = check_summary(partial_answer, current_answer, template)
        max_tries -= 1
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
    model, tokenizer = load_model(model_path)
    instructions = INSTRUCTIONS[mode]
    input_key = KV_PAIRS[mode]['input']
    output_key = KV_PAIRS[mode]['output']
    stoppers = [EOS_TOKEN, BOS_TOKEN]
    stops = [tokenizer.encode(stop)[1:] for stop in stoppers if stop in tokenizer.get_vocab().keys()]
    print(f"Stopping criteria: {stoppers}\nStopping ids: {stops}")
    pipe = pipeline("text-generation", 
                    model=model, 
                    tokenizer=tokenizer, 
                    eos_token_id=tokenizer.eos_token_id, 
                    pad_token_id=tokenizer.pad_token_id,
                    stopping_criteria=[CustomStoppingCriteria(stops=stops)],
                    **PARAMETERS[mode])
    generate = lambda x: pipe(x)[0]['generated_text']
    
    print(f"\nLoading data file from {input_path}...")
    data_df = load_file(input_path)
    if 'idx' not in data_df.columns:
        data_df['idx'] = data_df.index
    data_df = data_df.reset_index(drop=True)
    template = None if mode != 'summarizer' else load_template(template_path)

    print(f"Loading output file from {output_path}...")
    if os.path.exists(output_path):
        gen_df = load_file(output_path)
    else:
        gen_df = data_df.copy()
    if input_key not in gen_df.columns:
        raise ValueError(f'Input key {input_key} not found in output file.')
    if output_key not in gen_df.columns:
        gen_df[output_key] = None
    
    idx_todo = todo_list(data_df, gen_df, output_key, num_samples)

    # Generate samples
    for i in tqdm(idx_todo, desc='Generating samples'):
        row = gen_df[gen_df['idx'] == i].iloc[0]
        if mode == 'generator' and row[input_key] is None: 
            raise ValueError(f'Input key {input_key} not found when generating clinical note for sample {i}.')

        if mode == 'generator' or mode == 'direct' or mode == 'generator-gpt':
            input = row[input_key]
            if 'generator' in mode:
                input = '\n'.join([line for line in input.split('\n') if ': \"None\"' not in line])
            query = f"{BOS_TOKEN}\n{instructions[0]}\n\n{input}\n\n{instructions[1]}{EOS_TOKEN}\n{BOS_TOKEN} "
            if verbose: print(f'\n\n### PROMPT:\n\n{query}')
            answer = generate(query)

        elif mode == 'summarizer':
            answer = infer_summary(row[input_key], generate, template, instructions, verbose=verbose)

        if verbose: print(f'\n\n### ANSWER: \n\n{answer}')
        answer = answer.replace(BOS_TOKEN, '').replace(EOS_TOKEN, '').strip()
        gen_df.loc[gen_df['idx'] == i, output_key] = answer
        save_file(gen_df, output_path, mode='w')

    return gen_df
    

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
                        help='Mode of inference: summarizer, generator, generator-gpt, direct.')
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
            num_samples=args.num_samples,
            max_tokens=1000000,
            temperature=1.0,
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
