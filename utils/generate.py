'''
Synthetic data generation. 

From clinical notes, we generate tabular patient summaries.
'''


import time
import argparse
import shutil
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from utils.chat import *


def make_prompts(instruction, note=None, dialogue=None, template=None):
    '''
    Build prompts for chat.
    Arguments: 
        instruction: instruction to the model
        note: clinical note
        dialogue: dialogue
    '''
    sys_prompt = instruction
    if note is not None:
        sys_prompt += f"\n\nClinical Note:\n{note}"
    if dialogue is not None:
        sys_prompt += f"\n\nDialogue:\n{dialogue}"
    sys_prompt += "\n\nPatient Information Template:\n"
    if template is not None:
        sys_prompt += template
    usr_prompt = "Now fill in the patient information following the template provided.\
        \nIf a field is not mentioned in the dialogue, simply write \"feature\": None."
    return sys_prompt, usr_prompt

def load_template(template_path):
    '''
    Loads the JSON patient summary template from the path. 
    '''
    if not os.path.exists(template_path):
        raise FileNotFoundError(f'Template not found at {template_path}.')
    with open(template_path) as f:
        template = json.dumps(json.load(f), indent=4)
    return template

def extract(
        model,
        chat,
        template_path,
        instruction_path,
        data_path,
        save_path,
        max_tokens,
        use_notes=True, 
        use_dialogues=False,
        nb_to_generate = None):
    '''
    Extracts the patient summary from the clinical note and/or dialogue. 
    Arguments: 
        model: OpenAI model name
        template_path: path to the template file (.json)
        instruction_path: path to the instruction file (.txt)
        data_path: path to the dataframe containing clinical notes and dialogues
        save_path: path to save the dataframe containing extracted summaries (.jsonl)
        keys_path: path to OpenAI API keys
        dataframe: dataframe containing the clinical notes and dialogues
        use_notes: whether to use the clinical notes
        use_dialogues: whether to use the dialogues
    '''
    print("Loading Data...")
    if os.path.exists(data_path):
        notechat_batch = pd.read_json(data_path, lines=True, nrows=nb_to_generate, orient='records')
    else:
        raise ValueError(f'Data file {data_path} not found.')
    
    print("Looking for already generated summaries...")
    if os.path.exists(save_path):
        dataframe = load_file(save_path)
        ids_done = dataframe['idx'].tolist()
    else:
        dataframe = pd.DataFrame(columns=['idx', 'data', 'conversation', 'summary'])
        ids_done = []

    if len(ids_done) > 0:
        print(f'{len(ids_done)} generations already done. Skipping them.')
        notechat_batch = notechat_batch[~notechat_batch['idx'].isin(ids_done)]

    print("Loading template...")
    template = load_template(template_path)

    print("Loading instructions...")
    instruction = load_file(instruction_path)

    print("Building prompts...")
    prompts = [
            make_prompts(
                instruction=instruction,
                note=row['data'] if use_notes else None,
                dialogue=row['conversation'] if use_dialogues else None,
                template=template,
            ) for _, row in tqdm(notechat_batch.iterrows(), total=notechat_batch.shape[0])
        ]

    notechat_batch['messages'] =[build_messages(*prompt) for prompt in tqdm(
        prompts, total=len(prompts), desc="Building messages")]

    print("Creating sub-batches...")
    # Builds a partitions which have total number of tokens < max_tokens
    sub_batches = partition(dataframe = notechat_batch, max_token_per_partition=max_tokens,model = model)
    
    notechat_batch.drop(columns = ["messages"], inplace = True)

    for i ,(sub_batch_df, nb_tokens) in enumerate(sub_batches):
        print(f"Sub_batch {i+1}/{len(sub_batches)}: {sub_batch_df.shape[0]} calls, {nb_tokens} total tokens: {nb_tokens/1000 * 0.01}$")
        
        start_time = time.time()
        answers = generate_answers(
            messages_list = sub_batch_df['messages'].tolist(),
            formatting=lambda x: x,
            chat=chat,
            temperature=0.2
        )
        sub_batch_df['summary'] = answers
        sub_batch_df.drop(columns = ["messages"], inplace = True)
        ids_done.extend(sub_batch_df['idx'].tolist())
        dataframe = pd.concat([dataframe, sub_batch_df], ignore_index=True)
        shutil.copyfile(save_path, f'{save_path[:-6]}{i}.jsonl')

        with open(save_path, 'a') as f:
            f.write(sub_batch_df.to_json(orient='records', lines=True))
            print(f'\nSub-batch {i+1} Saved (size {sub_batch_df.shape[0]})\n')
            delete_pickle_file("safety_save.pkl")
        
        end_time = time.time()
        time_taken = (end_time - start_time)
        breaktime = max(int(60 - time_taken) + 2, 5) #time we wait before calling the api again
        print(f"\nBreak for {breaktime} seconds.")
        time.sleep(breaktime)
        print("End of break.")
    return dataframe


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-4-0613', 
                        help='OpenAI model name')
    parser.add_argument('--template_path', type=str, default='data/template.json', 
                        help='Path to template file')
    parser.add_argument('--instruction_path', type=str, default='data/instruction.txt', 
                        help='Path to instruction file')
    parser.add_argument('--data_path', type=str, default='data/df.jsonl', 
                        help='Path to data file')
    parser.add_argument('--save_path', type=str, default='data/df_extracted.jsonl', 
                        help='Path to save extracted data file')
    parser.add_argument('--use_notes', type=bool, default=True, 
                        help='Whether to use clinical notes')
    parser.add_argument('--use_dialogues', type=bool, default=False, 
                        help='Whether to use dialogues')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for generation')
    args = parser.parse_args()

    extract(
        model=args.model,
        template_path=args.template_path,
        instruction_path=args.instruction_path,
        data_path=args.data_path,
        save_path=args.save_path,
        use_notes=args.use_notes,
        use_dialogues=args.use_dialogues,
        batch_size=args.batch_size
    )