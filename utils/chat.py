'''
OpenAI API chat functions.
'''

import openai
import asyncio
import nest_asyncio
from typing import Any, Callable, List, Awaitable
import pickle
import tiktoken
import os
from tqdm import tqdm 
import pandas as pd
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

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
        return answer
    except ValueError as e:
        print(f"\nException occurred: {e}")
        return None

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
            temperature = temperature)
        nb_done += 1
        print(".", end = "")
        res_list.append(res)
        if nb_done % 20 == 0: #informative outputs
            print(nb_done)
            save_to_pickle("safety_save.pkl", res_list)  
        return res 
        
    async_responses = [one_call(x) for x in messages_list] #multiple calls
    new_responses = await asyncio.gather(*async_responses)
    try:
        prev_res = [formatting(x) for x in prev_res]
        new_responses = [formatting(x) for x in new_responses]
    except Exception as e:
        print(f"Formatting failed: {e}")

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
    print()
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
    # Create a new DataFrame with the original index
    return {"sub_batch": pd.DataFrame([row], index=[row.name]), "Total_nb_token": msg_len}

def partition(dataframe, max_token_per_partition, model):
    ''' Build most optimal partitions given max number of tokens, model and messages. '''
    sub_batches = []
    for _, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0], desc="Building sub-batches"):
        msg_len = num_tokens_from_messages(row['messages'], model)
        batch_len = len(sub_batches)

        if batch_len == 0:
            sub_batches.append(new_sub_batch(row, msg_len))
        else:
            current_partition = sub_batches[-1]
            if current_partition["Total_nb_token"] + msg_len <= max_token_per_partition:
                # Concatenating while keeping the original index
                current_partition["sub_batch"] = pd.concat([current_partition["sub_batch"], pd.DataFrame([row], index=[row.name])], ignore_index=False)
                current_partition["Total_nb_token"] += msg_len
            else:
                sub_batches.append(new_sub_batch(row, msg_len))
    
    return [d.values() for d in sub_batches]

