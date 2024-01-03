'''
Inference using gpt 3

From patient-doctor conversatio, we generate clinical notes with gpt 3.5 to have comparison baseline

'''


import time
import argparse
import shutil
import pandas as pd
import sys
import os

from utils.chat import *
from utils.generate import *
from utils.data import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

def infer_gpt3(
        data_path = './data/raw/summaries_full_test.jsonl',
        save_path = './data/raw/gpt3_full_test.jsonl',
        max_tokens = 1000000,
        nb_to_generate = 1000):
    
    print("Loading data...")
    if os.path.exists(data_path):
        dialogues_df = pd.read_json(data_path, lines=True, nrows=nb_to_generate, orient='records')
    else:
        raise FileNotFoundError(f'Data not found at {data_path}.')
    
    print("Looking for already infered dialogues ")
    if os.path.exists(save_path):
        infered_df = load_file(save_path)
        infered_df = infered_df[infered_df['gpt3 note'] != 'NA'].reset_index(drop = True)
        ids_done = infered_df['idx'].tolist()
    else:
        infered_df = pd.DataFrame(columns = dialogues_df.columns)
        ids_done = []

    if len(ids_done) > 0:
        print(f'{len(ids_done)} generations already done. Skipping them.')
        dialogues_df = dialogues_df[~dialogues_df['idx'].isin(ids_done)]
    
    instructions = 'Given the provided patient-doctor conversation, generate the corresponding clinical note as written by the physician. \nMake sure to use all the information from the dialogue into the note, but do not add any new information. Your asnwer should consist in one a or a few paragraphs of text'
    usr_prompt = 'Now, generate the corresponding clinical note: '
    
    prompts = [(f"{instructions}\n\n{dialogue}", usr_prompt) for dialogue in tqdm(dialogues_df['conversation'].tolist(), total = len(dialogues_df), desc = 'Building prompts')]
    
    dialogues_df['messages'] = [build_messages(*prompt) for prompt in tqdm(prompts, total = len(prompts), desc = 'Building messages')]

    print("Creating sub_batches...")

    sub_batches = partition(
        dataframe = dialogues_df,
        max_token_per_partition=max_tokens,
        model = 'gpt-3.5-turbo'
    )

    dialogues_df.drop(columns = ['messages'], inplace = True)


    for i, (sub_batch_df, nb_tokens) in enumerate(sub_batches):
        print(f"Sub_batch {i+1}/{len(sub_batches)}: {sub_batch_df.shape[0]} calls, {nb_tokens} total tokens: {nb_tokens/1000 * 0.001}$")
        
        start_time = time.time()
        answers = generate_answers(
            messages_list = sub_batch_df['messages'].tolist(),
            formatting = lambda x: x,
            chat = chat_gpt_3,
            temperature = 0.2
        )

        sub_batch_df['pred_noe_gpt3'] = answers

        sub_batch_df.drop(columns = ['messages'], inplace = True)
        ids_done.extend(sub_batch_df['idx'].tolist())
        infered_df = pd.concat([infered_df, sub_batch_df], ignore_index = True)
        
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

    return infered_df

        

