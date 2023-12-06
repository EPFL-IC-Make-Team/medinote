'''
Synthetic data generation for extractor fine-tuning. 
'''

import openai
import asyncio
import nest_asyncio
from typing import Any, Callable, List, Awaitable
import time
import pickle
import tiktoken
import os
from tqdm import tqdm 
import pandas as pd
import json
import argparse


def get_openai_credentials(path='generation/keys.json'):
    if not os.path.exists(path):
        print('Please save your OpenAI API key and organization ID to keys.json')
        return
    with open(path) as f:
        keys = json.load(f)
    
    openai.api_key = keys['api_key']
    openai.organization = keys['organization']
    return keys

keys = get_openai_credentials()

openai_client = openai.AsyncOpenAI(api_key = keys['api_key'], organization = keys['organization'])

def save_to_pickle(save_path, data):
    ''' Save data to pickle file '''
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

def load_from_pickle(filename):
    ''' Load data from pickle file '''
    with open(filename, 'rb') as f:
        return pickle.load(f)

def delete_pickle_file(file_path: str):
        if os.path.exists(file_path):
            os.remove(file_path)

async def ask_chat(
        chat: Callable[[List[dict]], Awaitable[str]],   # Chat function to which messages are passed
        messages: List[dict],                           # "List" but corresponds to only one call
        formatting: Callable[[str],Any],                 # Function which converts raw (text) chat output to required format
        temperature: float
        ) -> Awaitable[Any]:
    ''' 
    Prompts chat once and returns one answer, formatted and checked. 
    Arguments:
        chat: chat function to which messages are passed
        messages: messages to send to the chat
        formatting: function which converts raw (text) chat output to required format
        temperature: temperature for the chat
    '''
    try:
        answer =  await chat(messages = messages, temperature = temperature)
        try:
            formatted_answer = formatting(answer) 
        except Exception as f:
            raise ValueError("Wasn't answered in the right format")
        return formatted_answer
    except ValueError as e:
        print(f"\nException occurred: {e}")
        print(formatted_answer)
        return f"Wrong format: {answer}"

async def openai_chat(
      messages : List[dict], 
      model_name : str, 
      temperature: float,
      max_retries: int = 5, 
      timeout: int = 500
      ) ->  Awaitable[str]:
   ''' 
   OpenAI chat function. 
   Bypass timeout by using retries. 
   '''

   for _ in range(max_retries + 1):
    try:
      ans = openai_client.chat.completions.create(
                  model=model_name,
                  messages=messages,
                  temperature=temperature
                ) #acreat for async calls
      res = (await asyncio.wait_for(ans, timeout= timeout)).choices[0].message.content
      return res
    except asyncio.TimeoutError as te:
      print(f"\nChat TimeOut")
      if _ < max_retries:
          # Retry the operation if we haven't exceeded the max number of retries
          print(f"Retrying chat (TimeOut) ({_ + 1}/{max_retries})...")
      else:
          # If we've reached the maximum number of retries, pass and continue with the code
          print("Max retries reached for chat (TimeOut): passing")
          return "NA"

    except Exception as e:
        # If an exception occurs, display the error (optional)
        print(f"\nException occurred: {e}")
        if _ < max_retries:
            # Retry the operation if we haven't exceeded the max number of retries
            print(f"Retrying chat ({_ + 1}/{max_retries})...")
        else:
            # If we've reached the maximum number of retries, pass and continue with the code
            print("Max retries reached for chat: passing")
            return "NA"

async def chat_gpt_4_turbo(
      messages : List[dict] , 
      temperature:float, 
      max_retries: int = 5
      ) ->  Awaitable[str]:
    ''' GPT4 turbo chat function '''
    return await openai_chat(messages,"gpt-4-1106-preview", temperature, max_retries)

async def chat_gpt_3(
        messages : str, 
        temperature:float, 
        max_retries: int = 5
        ) ->  Awaitable[str]:
    ''' GPT 3.5 chat function '''
    return await openai_chat(messages,"gpt-3.5-turbo", temperature, max_retries)

async def dispatch_openai_requests(
      messages_list: List[List[dict]],
      chat : Callable[[List[dict]], Awaitable[str]],
      formatting: Callable[[str],Any],
      temperature: float
      ) -> list[Any]:
    ''' 
    Multiple calls to chat. 
    Arguments: 
        messages_list: list of messages to send to the chat
        chat: chat function to which messages are passed
        formatting: function which converts raw (text) chat output to required format
    '''
    if os.path.exists("safety_save.pkl"):
        prev_res = load_from_pickle("safety_save.pkl")
        print(f"Loaded {len(prev_res)} results from safety_save.pkl \n")
        print (prev_res)
    else:
        prev_res = []
    nb_done = len(prev_res)
    res_list = prev_res.copy()
    messages_list = messages_list[nb_done:]
    async def one_call(message: str):
        ''' One async call to ask_chat. '''
        nonlocal nb_done
        res = await ask_chat(
            chat = chat,
            messages= message,
            formatting = formatting,
            temperature = temperature)
        nb_done += 1
        res_list.append(res)
        if nb_done % 20 == 0: #informative outputs
            print(nb_done)
            save_to_pickle("safety_save.pkl", res_list)
        else: 
            print('.', end = '')
        return res 
        
    async_responses = [one_call(x) for x in messages_list] #multiple calls
    new_responses = await asyncio.gather(*async_responses)

    return prev_res + new_responses


def generate_answers(messages_list: list[List[dict]],
                     formatting: Callable[[str],Any],
                     chat : Callable[[List[dict]], Awaitable[str]],
                     temperature : float
                     ) -> list[Any]:
    '''
    Generates answers from a list of messages using chat function. 
    Arguments: 
        messages_list: list of messages to send to the chat
        max_tokens: maximum number of tokens the chat can handle per minute
        formatting: function which converts raw (text) chat output to required format
        chat: chat function to which messages are passed
        model: model name
    '''
    loop = asyncio.get_event_loop()
    nest_asyncio.apply()
    answers = loop.run_until_complete(
        dispatch_openai_requests(
            messages_list, 
            formatting=formatting, 
            chat=chat, 
            temperature=temperature))
    delete_pickle_file("safety_save.pkl")
    return(answers)

def num_tokens_from_messages(messages, model):
    '''
    Return the number of tokens used by a list of messages.
    (from OpenAI's documentation) 
    '''
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    models = ["gpt-3.5-turbo-0613", 
              "gpt-3.5-turbo-16k-0613", 
              "gpt-4-1106-preview", 
              "gpt-4-0314", 
              "gpt-4-32k-0314", 
              "gpt-4-0613", 
              "gpt-4-32k-0613"]
    if model in models:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        #print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        #print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. \
                See https://github.com/openai/openai-python/blob/main/chatml.md \
                    for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def build_messages(system_prompt: str, built_user_prompt: str): 
    ''' Build messages in the right format given system prompt and user prompt. '''
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": built_user_prompt}
    ]

def new_sub_batch(row, msg_len): 
  return {"sub_batch": pd.DataFrame([row]) , "Total_nb_token": msg_len}

def partition(dataframe, max_token_per_partition, model): 
  ''' Build most optimal partitions given max number of tokens, model and messages. '''
  sub_batches = []
  for _, row in dataframe.iterrows():
      msg_len = num_tokens_from_messages(row['messages'], model)
      batch_len = len(sub_batches)
      if batch_len == 0 : 
          sub_batches.append(new_sub_batch(row, msg_len))
      else : 
        current_partion = sub_batches[-1]
        if current_partion["Total_nb_token"] + msg_len <= max_token_per_partition : 
            current_partion["sub_batch"] = pd.concat([current_partion["sub_batch"], pd.DataFrame([row])], ignore_index=True)
            current_partion["Total_nb_token"] += msg_len
        else: sub_batches.append(new_sub_batch(row, msg_len))

  print(f"Created {len(sub_batches)} sub-batches with token number:" +\
         f"{[(sub_batch['sub_batch'].shape[0],sub_batch['Total_nb_token']) for sub_batch in sub_batches]}")
  
  return [(sb['sub_batch'], sb['Total_nb_token']) for sb in sub_batches]


"""def ask(sys_prompt, user_prompt, model="gpt-4", max_tokens=2000):
    ''' 
    One-time chat with OpenAI model.
    Prompt OpenAI model with system prompt + user prompt.
    Default model: gpt-4 (8K context length)
    '''
    message=[{"role": "system", "content": sys_prompt},
             {"role": "user", "content": user_prompt}]
    response = openai.chat.completions.create(
        model=model,
        messages=message,
        temperature=0.2,
        max_tokens=max_tokens,
        frequency_penalty=0.0
    )
    answer = response['choices'][0]['message']['content']
    return answer"""

def make_prompts(
        instruction, 
        note=None,
        dialogue=None,
        template=None,
        nshot=1):
    '''
    Build prompts for chat.
    Arguments: 
        instruction: instruction to the model
        note: clinical note
        dialogue: dialogue
        nshot: number of examples to build prompts for
    TODO: Support few-shot prompting
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
    # Load data (resume if save_path exists)
    if os.path.exists(data_path):
        if nb_to_generate is not None:
            notechat_batch = pd.read_json(data_path, lines=True, nrows = nb_to_generate ,orient='records')
        else :
            notechat_batch = pd.read_json(data_path, lines=True, orient='records')
    else:
        raise ValueError(f'Data file {data_path} not found.')
    
    # Create dataframe to save extracted summaries
    if os.path.exists(save_path):
        dataframe = pd.read_json(save_path, lines=True, orient='records')
        ids_done = dataframe['idx'].tolist()
    else:
        dataframe = pd.DataFrame(columns=['idx', 'data', 'conversation', 'summary'])
        ids_done = []

    if len(ids_done) > 0:
        print(f'{len(ids_done)} generations already done. Skipping them.')
        notechat_batch = notechat_batch[~notechat_batch['idx'].isin(ids_done)]

    # Load template
    if not os.path.exists(template_path):
        raise ValueError(f'Template file {template_path} not found.')
    with open(template_path) as f:
        template = json.dumps(json.load(f), indent=4)

    # Load instructions
    if not os.path.exists(instruction_path):
        raise ValueError(f'Instruction file {instruction_path} not found.')
    with open(instruction_path) as f:
        instruction = f.read()

    prompts = [
            make_prompts(
                instruction=instruction,
                note=row['data'] if use_notes else None,
                dialogue=row['conversation'] if use_dialogues else None,
                template=template,
            ) for _, row in notechat_batch.iterrows()
        ]
    
    # Build messages
    notechat_batch['messages'] =[build_messages(*prompt) for prompt in prompts]

    #Creating sub_batches:
    print(max_tokens)
    sub_batches = partition(dataframe = notechat_batch, max_token_per_partition=max_tokens,model = model) #builds a partitions which have total number of tokens < max_tokens
    
    notechat_batch.drop(columns = ["messages"], inplace = True)

    for i ,(sub_batch_df, nb_tokens) in enumerate(sub_batches):
        print(f"Sub_batch {i+1}/{len(sub_batches)}: {sub_batch_df.shape[0]} calls, {nb_tokens} tokens")
        
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
        with open(save_path, 'a') as f:
            f.write(sub_batch_df.to_json(orient='records', lines=True))
            print(f'\nSub-batch {i+1} Saved (size {sub_batch_df.shape[0]})\n')
        
        end_time = time.time()
        time_taken = (end_time - start_time)
        breaktime = max(int(60 - time_taken) + 2, 5) #time we wait before calling the api again
        print(f"\nBreak for {breaktime} seconds.")
        time.sleep(breaktime)
        print("End of break.")
    return dataframe


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-4-0613', help='OpenAI model name')
    parser.add_argument('--template_path', type=str, default='data/template.json', help='Path to template file')
    parser.add_argument('--instruction_path', type=str, default='data/instruction.txt', help='Path to instruction file')
    parser.add_argument('--data_path', type=str, default='data/df.jsonl', help='Path to data file')
    parser.add_argument('--save_path', type=str, default='data/df_extracted.jsonl', help='Path to save extracted data file')
    parser.add_argument('--use_notes', type=bool, default=True, help='Whether to use clinical notes')
    parser.add_argument('--use_dialogues', type=bool, default=False, help='Whether to use dialogues')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
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