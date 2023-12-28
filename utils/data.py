'''
Data preparation utilities.
'''

import os
import re
import argparse
from datasets import load_dataset
import matplotlib.pyplot as plt
import tiktoken
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.chat import chat_gpt_4_turbo
from utils.generate import extract

DATA_DIR = 'data'
for folder in ['summaries', 'summarizer', 'generator', 'direct']:
    os.makedirs(os.path.join(DATA_DIR, folder), exist_ok=True)

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

def save_file(df, path, mode='w'):
    '''
    Given a dataframe, save it to a .csv or .json or .jsonl file.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if '.csv' in path:
        df.to_csv(path, index=False, mode=mode)
    elif '.jsonl' in path:
        df.to_json(path, orient='records', lines=True, mode=mode)
    elif '.json' in path:
        df.to_json(path, orient='records', mode=mode)
    else: 
        raise ValueError(f"Provided path {path} is not a .csv, .json or .jsonl file.")
    return df


def count_tokens(text: str):
    '''
    Counts the number of tokens in a string.
    '''
    encoding = tiktoken.encoding_for_model('gpt-4')
    num_tokens = len(encoding.encode(text))
    return num_tokens

def length_histogram(data_path, keys=['prompt', 'gold']): 
    '''
    Plots a histogram of the length of the notes.
    '''
    df = pd.read_json(data_path, orient='records', lines=True)
    if isinstance(keys, str):
        keys = [keys]
    encoder = tiktoken.encoding_for_model('gpt-4')
    count = lambda row: len(encoder.encode(row))
    lengths = df.apply(lambda row: sum([count(row[key]) for key in keys]), axis=1)
    plt.hist(lengths, bins=max(lengths)-min(lengths)+1)
    plt.xlim(0, 2000)
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.title('Histogram of Length')
    plt.show()

def formatting(summary): 
    '''
    Removes unnecessary spaces and formats the summary to save tokens. 
    '''
    summary = summary.replace('. ', '.\n')
    summary = summary.replace('None', '"None"')
    summary = re.sub(r'\n\s+"', '\n"', summary)
    summary = re.sub(r'\n\s+{', '\n{', summary)
    summary = re.sub(r'\n\s+}', '\n}', summary)
    summary = re.sub(r'\n\s+}', '\n]', summary)
    summary = re.sub(r'\n\s+]', '\n]', summary)
    return summary

# ----------------------- Data preparation ----------------------- #

def split(data_path, test_ratio=0.1, random_state=42): 
    '''
    Split a dataset into train and test sets. 
    '''
    train_path = data_path.replace('.jsonl', '_train.jsonl')
    test_path = data_path.replace('.jsonl', '_test.jsonl')
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        data = pd.read_json(data_path, orient='records', lines=True)
        train, test = train_test_split(data, test_size=test_ratio, random_state=random_state)
        train.to_json(train_path, orient='records', lines=True)
        test.to_json(test_path, orient='records', lines=True)
        print(f'Saved train and test sets to {train_path} and {test_path}.')
    else:
        train = pd.read_json(train_path, orient='records', lines=True)
        test = pd.read_json(test_path, orient='records', lines=True)
    return train, test

def prepare_dataset(data_path, save_path, prompt_path=None, prompt_key='conversation', gold_key='full_note'):
    '''
    Prepares the data for generation by adding the prompt and gold text to the data.
    '''

    if not os.path.exists(data_path):
        raise ValueError(f'Data file {data_path} does not exist.')
    
    prompt = ''
    if prompt_path and os.path.exists(prompt_path):
        with open(prompt_path, 'r') as f:
            prompt = f.read()

    if os.path.exists(save_path):
        print(f'Data already prepared at {save_path}.')
        return pd.read_json(save_path, orient='records', lines=True)
    
    data = pd.read_json(data_path, orient='records', lines=True)
    data['prompt'] = prompt + '\n\n' + data[prompt_key]
    data['gold'] = data[gold_key]
    data = data[['idx', 'prompt', 'gold']]
    data.to_json(save_path, orient='records', lines=True)
    return data

def prepare(data_dir): 
    '''
    Entire data preparation pipeline.

    - Load NoteChat data
    - Sort NoteChat data by decreasing length
    - Generate summaries (30K)
    - Prepare summarizer dataset
    - Prepare generator dataset 
    - Prepare direct dataset
    - Split data into train and test sets
    '''
    print(f'Data preparation initiated.\n')
    summaries_path = os.path.join(data_dir, 'summaries', 'summaries_30K.jsonl')
    if os.path.exists(summaries_path):
        print(f'Loading summaries from {summaries_path}.')
    else:
        notechat_path = os.path.join(data_dir, 'NoteChat.jsonl')
        if not os.path.exists(notechat_path):
            print(f'Loading NoteChat data from HuggingFace to {notechat_path}...')
            notechat = load_dataset("akemiH/NoteChat")['train'].to_pandas()
            notechat.to_json(notechat_path, orient='records', lines=True)
        else:
            print(f'Loading NoteChat data from {notechat_path}...')
            notechat = pd.read_json(notechat_path, orient='records', lines=True)
        notechat.rename(columns={'data': 'note'}, inplace=True)

        if 'idx' not in notechat.columns:   
            print(f'Sorting NoteChat data by decreasing length...')
            notechat['length'] = notechat['note'].apply(lambda x: count_tokens(x))
            notechat = notechat.sort_values(by=['length'], ascending=False)
            notechat = notechat.drop(columns=['length'])
            notechat['idx'] = notechat.index
            data_path = os.path.join(data_dir, 'NoteChat_sorted.jsonl')
            notechat.to_json(data_path, orient='records', lines=True)

        print(f'Generating summaries...')
        extract(
            model = 'gpt-4-1106-preview',
            chat = chat_gpt_4_turbo,
            template_path = 'generation/templates/template_definitions.json',
            instruction_path = 'generation/instructions/generate.txt',
            data_path = data_path,
            save_path = summaries_path,
            use_notes = True, 
            use_dialogues = False,
            max_tokens = 500000,
            nb_to_generate = 30000)
    
        print(f'Formatting patient summaries...')
        summaries = pd.read_json(summaries_path, lines=True)
        summaries['summary'] = summaries['summary'].apply(formatting)          
        summaries['data'] = summaries['data'].apply(lambda x: x.strip())
        summaries['conversation'] = summaries['conversation'].apply(lambda x: x.strip().replace('\n\n', '\n'))
        summaries.to_json(summaries_path, orient='records', lines=True)

    print(f'Preparing summarizer dataset...')
    summarizer_path = os.path.join(data_dir, 'summarizer', 'summarizer_30K.jsonl')
    summarizer = prepare_dataset(
        data_path=summaries_path, 
        save_path=summarizer_path,
        prompt_path='generation/instructions/summarize.txt',
        prompt_key='conversation',
        gold_key='summary')
    
    print(f'Preparing generator dataset...')
    generator_path = os.path.join(data_dir, 'generator', 'generator_30K.jsonl')
    generator = prepare_dataset(
        data_path=summaries_path, 
        save_path=generator_path,
        prompt_path='generation/instructions/generate.txt',
        prompt_key='summary',
        gold_key='data')

    print(f'Preparing direct dataset...')
    direct_path = os.path.join(data_dir, 'direct', 'direct_30K.jsonl')
    direct = prepare_dataset(
        data_path=summaries_path, 
        save_path=direct_path,
        prompt_path='generation/instructions/direct.txt',
        prompt_key='conversation',
        gold_key='data')
    
    print(f'Splitting data into train and test sets...')
    summarizer_train, summarizer_test = split(summarizer_path, test_ratio=0.1)
    generator_train, generator_test = split(generator_path, test_ratio=0.1)
    direct_train, direct_test = split(direct_path, test_ratio=0.1)

    print(f'Checking for contamination...')
    summarizer_train_idx = set(summarizer_train['idx'])
    generator_test_idx = set(generator_test['idx'])
    direct_test_idx = set(direct_test['idx'])
    assert len(summarizer_train_idx.intersection(generator_test_idx)) == 0
    assert len(summarizer_train_idx.intersection(direct_test_idx)) == 0
    print('\nData preparation completed.')


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None, help='Path to data directory')
    args = parser.parse_args()

    data_dir = DATA_DIR if args.data_dir is None else args.data_dir

    prepare(data_dir)
