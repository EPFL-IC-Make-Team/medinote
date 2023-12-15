'''
Evaluation utilities. 

1 - Patient Summary evaluation
Given our model’s patient summary and GPT-4's summary generated from note, 
compute the BLEU/ROUGE/BERT scores for all matching fields in the template. 
We use BERT score to match list elements (e.g. symptoms). 

2 - Clinical note evaluation
Here we want to evaluate the quality of our generated clinical notes using GPT-4. Two methods:
--> Given 2 models’ answers (randomly mixed), ask GPT-4 to pick which one is the best and compute an ELO score. 
--> Given a model’s answer and GPT-4' answer (silver label), ask GPT-4 with CoT to compute the similarity 
between the two answers on a scale of 1 to 10. The higher the number the closer the model’s quality to GPT-4's.

'''

from utils.chat import *

import asyncio
import numpy as np
import re
import argparse
from evaluate import load
from multielo import MultiElo
import matplotlib.pyplot as plt

BERT_SCORER = load("bertscore")
ROUGE_SCORER = load("rouge")
BLEU_SCORER = load("bleu")

none_match  = "None_Match"
no_such_key = "no_such_key"
none_field = "None"

# ----------------------- Generic utils ----------------------- #

def load_file(path): 
    '''
    Given a .csv or .json or .jsonl file, load it into a dataframe.
    '''
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist.")
    if '.csv' in path:
        df = pd.read_csv(path)
    elif '.jsonl' in path:
        df = pd.read_json(path, lines=True)
    elif '.json' in path:
        df = pd.read_json(path)
    else: 
        raise ValueError(f"Provided path {path} is not a .csv, .json or .jsonl file.")
    return df

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

# ----------------------- Scoring functions (BLEU/ROUGE/BERT/GPT-4) ----------------------- #

class Scorer(): 
    def __init__(self, score_types='all', cot=False):
        ''' 
        Initializes a scorer with a list of score types to be used.
        Argument: 
            - score_types (str or list): list of scoring functions. Default: 'all' (all scoring functions)
            - cot (bool): whether to use Chain-of-Thought or not for GPT-4 evaluation (default: True)
        '''
        self.cot = cot
        self.score_types = ['bleu', 'rouge', 'bert', 'gpt_rank', 'gpt_score'] if score_types == 'all' else score_types
        print('Initialized scorer with score types: ', list(self.score_types))

    async def __call__(self, gold, pred): 
        '''
        Given a gold and predicted string, returns a dictionary of all scores. 

        Example usage: 
            scorer = Scorer(['bleu', 'rouge', 'bert'])
            scores = await scorer(gold, pred)
        '''
        scores = {}
        if 'bleu' in self.score_types:
            scores['bleu'] = self.BLEU_score(gold, pred)
        if 'rouge' in self.score_types:
            scores['rouge'] = self.ROUGE_score(gold, pred)
        if 'bert' in self.score_types:
            scores['bert'] = self.BERT_score(gold, pred)
        if 'gpt_rank' in self.score_types:
            scores['gpt_rank'] = await self.GPT4_rank(gold, pred)
        if 'gpt_score' in self.score_types:
            scores['gpt_score'] = await self.GPT4_score(gold, pred)
        return scores
    
    def evaluate(self, path):
        '''
        Given a path to a file, load it into a dataframe 
        and compute scores for each pair of gold and pred strings.
        '''
        df = load_file(path)
        df = df.sample(frac=1).reset_index(drop=True)
        for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Scoring..."):
            scores = self(row['gold'], row['pred'])
            for score_type, score in scores.items():
                df.loc[i, score_type] = score
            if i % 10 == 0:
                save_file(df, path)
        save_file(df, path)
        return df
    
    
    def BLEU_score(self, gold, pred):
        ''' BLEU score for summary evaluation (precision-oriented)'''
        results = BLEU_SCORER.compute(predictions=[pred], references=[[gold]])
        return results['precisions'][0]
    
    def ROUGE_score(self, gold, pred):
        ''' ROUGE score for summary evaluation (recall-oriented)'''
        results = ROUGE_SCORER.compute(predictions=[pred], references=[[gold]])
        return results

    def BERT_score(self, gold, pred):
        results = BERT_SCORER.compute(predictions=[pred], references=[gold], lang="en")
        return results['f1'][0]
    
    async def GPT4_rank(self, gold, pred, 
                        model_name='gpt-4-1106-preview', 
                        temperature=0.0, 
                        max_retries=5
                        ):
        ''' 
        Given 2 models’ answers (randomly mixed), ask GPT-4 to pick which one is the best. 
        NOTE: we randomly mix gold and pred to avoid bias.

        Arguments:
            - gold (str): gold answer
            - pred (str): predicted answer
            - model_name (str): name of the GPT-4 model to use (default: gpt-4-1106-preview)
            - temperature (float): temperature for GPT-4 sampling
            - max_retries (int): max number of retries for GPT-4 sampling
            - cot (bool): whether to use Chain-of-Thought or not (default: True)
        Returns: 
            - dictionary with the winner and the explanation.
        '''
        switch = np.random.choice([0, 1])
        option1 = gold if switch == 0 else pred
        option2 = gold if switch == 1 else pred
        sys_prompt = f"Compare the two clinical notes and rank which one is of higher quality. \
            \n\nAnswer 1: {option1}\n\nAnswer 2: {option2}\n\nAnswer 3: They are equally good."
        if self.cot: 
            usr_prompt = "In one sentence, explain your reasoning in comparing the two clinical notes, then select your final answer. \
                Format your response as follows: \n\nExplanation: <your explanation>\n\nAnswer: <your answer>"
        else: 
            usr_prompt = "Directly select your final answer as follows: \n\nAnswer: <your answer>"
        messages = build_messages(sys_prompt, usr_prompt)
        try: 
            response = await openai_chat(messages,model_name, temperature, max_retries)
            answer = response.split('Answer: ')[1]
            explanation = None if not self.cot else response.split('Explanation: ')[1].split('Answer: ')[0].strip()
            if '1' in answer: 
                winner = 'gold' if switch == 0 else 'pred'
            elif '2' in answer:
                winner = 'gold' if switch == 1 else 'pred'
            elif '3' in answer:
                winner = 'tie'
            else:
                raise ValueError(f"Invalid answer {answer}.")
            return {'winner': winner, 'explanation': explanation}
        except:
            return None
    
    async def GPT4_score(self, gold, pred, 
                         model_name='gpt-4-1106-preview',
                         temperature=0.0,
                         max_retries=5):
        ''' 
        Given a model’s answer and GPT-4' answer (silver label), 
        ask GPT-4 with CoT to compute the similarity between the two answers on a scale of 1 to 10. 
        The higher the number the closer the model’s quality to GPT-4's. 
        NOTE: we randomly mix gold and pred to avoid bias.

        A
        '''
        switch = np.random.choice([0, 1])
        option1 = gold if switch == 0 else pred
        option2 = gold if switch == 1 else pred
        sys_prompt = f"Compare the two clinical notes and rate how similar they are to each other on a scale of 1 to 10. \
            \n\nAnswer 1: {option1}\n\nAnswer 2: {option2}\n\nAnswer 3: They are equally good."
        if self.cot: 
            usr_prompt = "In one sentence, explain your reasoning in comparing the two clinical notes, then select your final answer. \
                Format your response as follows: \n\nExplanation: <your explanation>\n\nAnswer: <your answer>"
        else: 
            usr_prompt = "Directly select your final answer as follows: \n\nAnswer: <your answer>"
        messages = build_messages(sys_prompt, usr_prompt)
        try:
            response = await openai_chat(messages, model_name, temperature, max_retries)
            answer = response.split('Answer: ')[1]
            explanation = None if not self.cot else response.split('Explanation: ')[1].split('Answer: ')[0].strip()
            similarity = int(re.findall(r'\d+', answer)[0])
            return {'similarity': similarity, 'explanation': explanation}
        except:
            return None

    
    
# ----------------------- 1 - Patient Summary evaluation ----------------------- #
    
def match_pred_list(gold_list, pred_list, match_score):
    '''
    Given two lists of dictionaries, match corresponding dictionaries by value
    and compute matching score between their values.
    '''
    if len(gold_list) == 1 and len(pred_list) == 1:
        return gold_list, pred_list
    
    if len(gold_list) == 0:
        matched_gold = [{}] * len(pred_list)
    if len(pred_list) == 0:
        matched_pred = [{}] * len(gold_list)

    pred_list_ = pred_list.copy()
    matched_pred = []
    matched_gold = gold_list.copy()

    # Iterate through each element in gold_list
    for gold_item in gold_list:
        max_score = None
        best_match_index = -1

        # Find the best match in pred_list
        if len(pred_list_) > 0:
            for i, pred_item in enumerate(pred_list_):
                gold_string = ", ".join(str(value) for value in gold_item.values() if value is not None)
                pred_string = ", ".join(str(value) for value in pred_item.values() if value is not None)
                score = match_score(gold_string, pred_string)
                if max_score is None or score > max_score:
                    max_score = score
                    best_match_index = i

            matched_pred.append(pred_list[best_match_index])
            pred_list_.pop(best_match_index)

    if len(pred_list_) > 0:
        matched_gold += [{}] * len(pred_list)
        matched_pred += pred_list_
    else:
        matched_pred.extend([{}] * (len(matched_gold) - len(matched_pred)))

    return matched_gold, matched_pred


def flatten_and_match_dicts(gold_dict, pred_dict, match_score, parent_key=''):
    '''
    Given two dictionaries, match corresponding keys 
    and compute matching score between their values. 

    Arguments: 
        - gold_dict (dict): dictionary of gold values
        - pred_dict (dict): dictionary of predicted values
        - match_score (function): function ((str, str) --> float) to compute matching score
        - parent_key (str): key of parent dictionary, used for recursion
    '''
    scorer = Scorer(['bleu', 'rouge', 'bert'])
    flattened_dict_items = []
    for gold_key, gold_value in gold_dict.items():
        if gold_key in pred_dict.keys():
            if isinstance(gold_value, str):
                    flattened_dict_items.append((f"{parent_key}{gold_key}",
                                                (gold_value, pred_dict[gold_key])))               
            if isinstance(gold_value, dict):
                flattened_dict_items.extend(flatten_and_match_dicts(gold_value,
                                                            pred_dict[gold_key],
                                                            match_score,
                                                            parent_key=f"{parent_key}{gold_key}/"))
            if isinstance(gold_value, list):
                matched_gold, matched_pred = match_pred_list(gold_list = gold_value,
                                                pred_list = pred_dict[gold_key],
                                                match_score = match_score)
                for i, gold_list_val in enumerate(matched_gold):
                    if isinstance(gold_list_val, str):
                        flattened_dict_items.append(f"{parent_key}{gold_key}/{i}" ,
                                                (gold_list_val, matched_pred[i]))
                    if isinstance(gold_list_val, dict):
                        flattened_dict_items.extend(flatten_and_match_dicts(gold_list_val,
                                                                matched_pred[i],
                                                                match_score,
                                                                parent_key=f"{parent_key}{gold_key}/{i}/"))
        else :
            if isinstance(gold_value, str):
                flattened_dict_items.append(f"{parent_key}{gold_key}" ,
                                                (gold_value, no_such_key))
            if isinstance(gold_value, dict):
                flattened_dict_items.extend(flatten_and_match_dicts(gold_value,
                                                            {},
                                                            match_score,
                                                            parent_key=f"{parent_key}{gold_key}/"))
            if isinstance(gold_value, list):
                for i, val in enumerate(gold_value):
                    if isinstance(val, str):
                        flattened_dict_items.append(f"{parent_key}{gold_key}/{i}" ,
                                                (gold_value, no_such_key))
                    if isinstance(val, dict):
                        flattened_dict_items.extend(flatten_and_match_dicts(val,
                                                                {},
                                                                match_score,
                                                                parent_key=f"{parent_key}{gold_key}/{i}/"))

    
    for pred_key, pred_value in pred_dict.items():
        if pred_key not in gold_dict.keys():
            if isinstance(pred_value, str):
                flattened_dict_items.append(f"{parent_key}{pred_key}" ,
                                                (no_such_key, pred_value))
            if isinstance(pred_value, dict):
                flattened_dict_items.extend(flatten_and_match_dicts({},
                                                            pred_value,
                                                            match_score,
                                                            parent_key=f"{parent_key}{pred_key}/"))
            if isinstance(pred_value, list):
                for i, val in enumerate(pred_value):
                    if isinstance(val, str):
                        flattened_dict_items.append(f"{parent_key}{pred_key}/{i}" ,
                                                (no_such_key, pred_value))
                    if isinstance(val, dict):
                        flattened_dict_items.extend(flatten_and_match_dicts({},
                                                                val,
                                                                match_score,
                                                                parent_key=f"{parent_key}{pred_key}/{i}"))
    

    if parent_key == '':
        for k in flattened_dict_items:
            print(k)
        return dict(flattened_dict_items)
    else:
        return flattened_dict_items

def get_evaluation_dict(flattened_matched_dict, evaluation_scores):
    evaluation_dict = {}
    for key in ['missing_keys', 'additional_keys', 'accurate_nones', 
                'accurate_not_nones', 'non_accurate_nones', 'non_accurate_none_nones']:
        evaluation_dict[key] = 0
    for key, (gold, pred) in flattened_matched_dict.items():
        if gold == no_such_key:
            evaluation_dict['additional_keys'] += 1
            evaluation_dict[key] = (gold, pred)
        if pred == no_such_key:
            evaluation_dict['missing_keys'] += 1
            evaluation_dict[key] = (gold, pred)
        if gold == none_field:
            if pred == none_field:
                evaluation_dict['accurate_nones'] += 1
                evaluation_dict[key] = none_match
            else:
                evaluation_dict[key] = (gold, pred)
                evaluation_dict['non_accurate_none_nones'] += 1
        else:
            if pred == none_field:
                evaluation_dict[key] = (gold, pred)
                evaluation_dict['non_accurate_nones'] += 1
            else:
                evaluation_dict[key] = evaluation_scores(gold, pred)
                accurate_not_nones += 1
    
    return evaluation_dict

def build_gpt_4_scoring_dataset(evaluation_dicts_df):
    '''
    Given a df of evaluation dictionaries,
    build a df of evaluation dictionaries for GPT-4 scoring.
    '''
    gpt_4_scoring_dataset = evaluation_dicts_df[['idxs', 'eval_dict']]
    gpt_4_scoring_dataset['eval_dict'] = build_gpt_4_scoring_dataset['eval_dict'].apply(lambda x: {k: v for k, v in x.items() if isinstance(v, dict)})
    gpt_4_scoring_dataset['eval_dict'] = build_gpt_4_scoring_dataset['eval_dict'].apply(lambda x: {k: v['gpt_4'] for k, v in x.items()})
    gpt_4_scoring_dataset = build_gpt_4_scoring_dataset.explode('eval_dict')
    gpt_4_scoring_dataset['key'] = build_gpt_4_scoring_dataset['eval_dict'].apply(lambda x: x[0])
    gpt_4_scoring_dataset['gold'] = build_gpt_4_scoring_dataset['eval_dict'].apply(lambda x: x[1][0])
    gpt_4_scoring_dataset['pred'] = build_gpt_4_scoring_dataset['eval_dict'].apply(lambda x: x[1][1])
    gpt_4_scoring_dataset.drop(columns=['eval_dict'])

    return build_gpt_4_scoring_dataset

def input_gpt_4_scores(evaluation_dicts_df, gpt_4_scoring_dataset):
    '''
    Given a df of evaluation dictionaries and a df of GPT-4 scoring dictionaries,
    input GPT-4 scores into the evaluation dictionaries df.
    '''
    gpt_4_scoring_dataset = gpt_4_scoring_dataset.groupby('idxs').agg({'key': list, 'score': list})
    evaluation_dicts_df = evaluation_dicts_df.merge(gpt_4_scoring_dataset, on='idxs')
    evaluation_dicts_df['eval_dict'] = evaluation_dicts_df[['eval_dict']].apply(
        lambda x: {k: v for k, v in x['eval_dict'].items() if k in x['key']}, axis=1)
    return evaluation_dicts_df

def summary_evaluation(path, score_types='all'): 
    '''
    1 - Patient summary evaluation
    Run evaluation on a dataframe with 'gold' and 'pred' patient summaries. 

    Arguments: 
        - path (str): path to dataframe with 'gold' and 'pred' patient summaries
        - score_types (str or list): list of scoring functions to be used. Default: 'all' (all scoring functions)

    NOTE: Need to implement the inference and creation of this dataframe in inference.py
    '''
    # Load dataframe with inference results
    df = load_file(path)
    df = df.sample(frac=1).reset_index(drop=True)

    # TODO: Compute scores for each pair of gold and pred patient summaries
    raise NotImplementedError



# ----------------------- 2 - Clinical note evaluation ----------------------- #
    
    
def clinical_note_evaluation(model_name, path, score_types='all'): 
    '''
    2 - Clinical note evaluation
    Given 2 models’ answers (randomly mixed), ask GPT-4 to pick which one is the best and compute an ELO score. 
    Run evaluation on a dataframe with 'gold' and 'pred' clinical notes. 

    Arguments: 
        - model_name (str): name of the model used to generate the clinical notes
        - path (str): path to dataframe with 'gold' and 'pred' clinical notes
        - score_types (str or list): list of scoring functions to be used. Default: 'all' (all scoring functions)

    NOTE: Need to implement inference and creation of this dataframe in inference.py
    '''
    # Load dataframe with inference results
    df = load_file(path)
    df = df.sample(frac=1).reset_index(drop=True)
    df['model_name'] = model_name

    # Compute scores for each pair of gold and pred clinical notes
    scorer = Scorer(score_types)
    scores_path = path.replace('.jsonl', '_scores.jsonl')
    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Evaluating pairs of clinical notes"):
        scores = scorer.evaluate(row['gold'], row['pred'])
        for score_type, score in scores.items():
            df.loc[i, score_type] = score
        if i % 10 == 0:
            save_file(df, scores_path)
    save_file(df, scores_path)

    # Compute ELO ranking from GPT-4 scores
    if 'gpt_4_rank' in score_types:
        rankings_path = path.replace('.jsonl', '_rankings.jsonl')
        rankings = elo_ranking(df)
        print(f'ELO rankings: {rankings}')
        with open(rankings_path, 'w') as f:
            json.dump(rankings, f)
            print(f'Saved ELO rankings to {rankings_path}')
    return df

def elo_ranking(df):
    ''' 
    Elo ranking for clinical note evaluation with GPT-4.
    Taken from https://portkey.ai/blog/comparing-llm-outputs-with-elo-ratings/

    Argument:
        - df (pd.DataFrame): dataframe with GPT-4 scores for each model
    Returns: 
        - rankings (dict of str: float): dictionary of Elo rankings for each model

    All models start at 1500 Elo rating.
    '''
    model_names = list(df['model_name'].unique()) + ['gpt-4']
    elo_history = {model: np.array([1500]) for model in model_names}
    elo = MultiElo()

    # For each score given in df, compute Elo ranking
    score_dict = {model: [] for model in model_names}
    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Computing ELO rankings"):
        model = row['model_name']
        rank_score = row['gpt_4_rank']
        score_dict[model].append(rank_score)
        new_ratings = elo.get_new_ratings(elo_history[model], [rank_score])
        elo_history[model] = np.append(elo_history[model], new_ratings[0])

    # Show Elo ranking history as a plot
    for model in model_names:
        plt.plot(elo_history[model], label=model)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Elo Rating")
    plt.title("Elo Rating Changes")
    plt.legend()
    plt.show()

    # Compute Elo ranking for each model
    rankings = {}
    for model in model_names:
        rankings[model] = elo_history[model][-1]
    return rankings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', 
                        type=str, 
                        default='summary', 
                        help='summary or clinical_note')
    parser.add_argument('--path', 
                        type=str, 
                        default='data/evaluation/summary_evaluation.jsonl', 
                        help='path to dataframe with evaluation results')
    parser.add_argument('--score_types', 
                        type=str,
                        default='all', 
                        help='List of scoring functions to be used (choices: bleu, rouge, bert, gpt_4_rank, gpt_4_sim). \
                            \nDefault: all (all scoring functions). Format example: "bleu, rouge, bert"')
    args = parser.parse_args()
    score_types = args.score_types.split(', ')

    print(f'Running evaluation on {args.mode} with score types: {score_types}')

    if args.mode == 'summary':
        summary_evaluation(args.path, score_types)

    elif args.mode == 'clinical_note':
        clinical_note_evaluation(args.path, score_types)

    else:
        raise ValueError(f"Mode {args.mode} is not valid. Please choose between 'summary' and 'clinical_note'.")
    
