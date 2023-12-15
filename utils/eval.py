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
from utils.inference import *

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

NONE_MATCH  = "None_Match"
NO_SUCH_KEY = "no_such_key"
NONE_FIELD = "None"

# ----------------------- Scoring functions (BLEU/ROUGE/BERT/GPT-4) ----------------------- #

class Scorer(): 
    def __init__(self, score_types='all', cot=False):
        ''' 
        Initializes a scorer with a list of scoring modes to be used.
        Argument: 
            - score_types (str or list): list of scoring functions. Default: 'all' (all scoring functions)
            - cot (bool): whether to use Chain-of-Thought or not for GPT-4 evaluation (default: True)
        '''
        self.cot = cot
        self.score_types = ['bleu', 'rouge', 'bert', 'gpt_rank', 'gpt_score'] if score_types == 'all' else score_types
        #print('Initialized scorer with modes: ', list(self.score_types))

    def __call__(self, pairs): 
        '''
        Given a list of dictionaries with gold and predicted pairs, 
        returns a dictionary of lists with all metrics computed for each pair.

        Example usage: 
            pairs = [{'gold': 'x', 'pred': 'y'}]
            scorer = Scorer(['bleu', 'rouge', 'bert'])
            scores = await scorer(pairs)
            --> scores = [{'bleu': 0.5, 'rouge': 0.3, 'bert': 0.7}]
        '''
        scores = {}
        if 'bleu' in self.score_types:
            scores.update(self.BLEU_scorer(pairs))
        if 'rouge' in self.score_types:
            scores.update(self.ROUGE_scorer(pairs))
        if 'bert' in self.score_types:
            scores.update(self.BERT_scorer(pairs))
        if 'gpt_rank' in self.score_types:
            scores['gpt_rank'] = self.GPT_ranker(pairs)
        if 'gpt_score' in self.score_types:
            scores['gpt_score'] = self.GPT_scorer(pairs)
        return scores
    
    def BLEU_scorer(self, pairs):
        ''' BLEU score for summary evaluation (precision-oriented)'''
        scores = {'bleu': []}
        for pair in pairs:
            score = BLEU_SCORER.compute(
                predictions=[pair['pred']], references=[pair['gold']])['bleu']
            scores['bleu'].append(score)
        return scores
    
    def ROUGE_scorer(self, pairs):
        ''' ROUGE score for summary evaluation (recall-oriented)'''
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': []}
        for pair in pairs:
            rouge = ROUGE_SCORER.compute(
                predictions=[pair['pred']], references=[pair['gold']])
            for metric in rouge.keys():
                scores[metric].append(rouge[metric])
        return scores

    def BERT_scorer(self, pairs):
        scores = {'bert': []}
        for pair in pairs:
            score = BERT_SCORER.compute(
                predictions=[pair['pred']], references=[pair['gold']], lang='en')['f1'][0]
            scores['bert'].append(score)
        return scores
    
    def GPT_ranker(self, 
                   pairs,
                   model_name='gpt-4-1106-preview', 
                   batch_size=10, 
                   temperature=0.0):
        ''' 
        For each pair of gold and pred strings, ask GPT-4 to pick which one is the best.
        NOTE: we randomly mix answer order to avoid bias.

        Arguments:
            - pairs (list of dict {'gold': str, 'pred': str}): list of gold and pred strings
            - model_name (str): name of the GPT-4 model to use (default: gpt-4-1106-preview)
            - batch_size (int): batch size for GPT-4 parallelization (default: 10)
            - temperature (float): temperature for GPT-4 sampling (default: 0.0)
        Returns: 
            - dataframe with the winner and the explanation

        TODO: Save the explanations and scores in a separate file.
        '''

        # Build prompts for GPT-4 from gold/pred pairs
        dataset = pd.DataFrame(pairs)
        dataset[['winner', 'explanation', 'switch', 'model_name']] = [None, None, None, model_name]
        messages = []
        for i, pair in tqdm(dataset.iterrows(), total=len(pairs), desc="Building rank prompts"): 
            gold = pair['gold']
            pred = pair['pred']
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
            messages.append(build_messages(sys_prompt, usr_prompt))
            dataset.iloc[i]['switch'] = switch
        
        # Generate answers by batches
        for i in tqdm(range(0, dataset.shape[0], batch_size), desc="Ranking pairs of clinical notes"):
            batch_df = dataset.iloc[i:i+batch_size]
            try:
                answers = generate_answers(
                    messages_list = messages[i:i+batch_size],
                    formatting=lambda x: x,
                    chat=chat_gpt_4_turbo,
                    temperature=temperature
                )
                for j, response in enumerate(answers):
                    answer = response.split('Answer: ')[1]
                    explanation = None if not self.cot else response.split('Explanation: ')[1].split('Answer: ')[0].strip()
                    switch = batch_df.iloc[i+j]['switch']
                    if '1' in answer: 
                        winner = 'gold' if switch == 0 else 'pred'
                    elif '2' in answer:
                        winner = 'gold' if switch == 1 else 'pred'
                    elif '3' in answer:
                        winner = 'tie'
                    else:
                        raise ValueError(f"Invalid answer {answer}.")
                    dataset.iloc[i+j]['winner'] = winner
                    dataset.iloc[i+j]['explanation'] = explanation
            except: 
                answers = [None] * batch_df.shape[0]
        return list(dataset['winner'])
    
    
    def GPT_scorer(self, 
                   pairs,
                   model_name='gpt-4-1106-preview',
                   temperature=0.0,
                   batch_size=10):
        ''' 
        Given a model’s answer and GPT-4' answer (silver label), 
        ask GPT-4 with CoT to compute the similarity between the two answers on a scale of 1 to 10. 
        The higher the number the closer the model’s quality to GPT-4's. 
        NOTE: we randomly mix gold and pred to avoid bias.

        Arguments:
            - pairs (list of dict {'gold': str, 'pred': str}): list of gold and pred strings
            - model_name (str): name of the GPT-4 model to use (default: gpt-4-1106-preview)
            - temperature (float): temperature for GPT-4 sampling (default: 0.0)
            - batch_size (int): batch size for GPT-4 parallelization (default: 10)
        Returns: 
            - dataframe with the similarity score and the explanation

        TODO: Save the explanations and scores in a separate file.
        '''

        # Build prompts for GPT-4 from gold/pred pairs
        dataset = pd.DataFrame(pairs)
        messages = []
        dataset[['similarity', 'explanation', 'model_name']] = [None, None, model_name]
        for i, pair in tqdm(dataset.iterrows(), total=len(pairs), desc="Building rank prompts"): 
            gold = pair['gold']
            pred = pair['pred']
            switch = np.random.choice([0, 1])
            option1 = gold if switch == 0 else pred
            option2 = gold if switch == 1 else pred
            sys_prompt = f"Compare the two clinical notes and rate how similar they are to each other on a scale of 1 to 10.\
                \n\nNote 1:\n\n{option1}\n\nNote 2:\n\n{option2}"
            if self.cot: 
                usr_prompt = "In one sentence, explain your reasoning in comparing the two clinical notes, then select your final answer. \
                    Format your response as follows: \n\nExplanation: <your explanation>\n\nAnswer: <your answer>"
            else: 
                usr_prompt = "Directly respond with your final answer as follows: \n\nAnswer: <your answer>"
            messages.append(build_messages(sys_prompt, usr_prompt))

        # Generate answers by batches
        for i in tqdm(range(0, dataset.shape[0], batch_size), desc="Scoring pairs of clinical notes"):
            batch_df = dataset.iloc[i:i+batch_size]
            try:
                answers = generate_answers(
                    messages_list = messages[i:i+batch_size],
                    formatting=lambda x: x,
                    chat=chat_gpt_4_turbo,
                    temperature=temperature
                )
                for j, response in enumerate(answers):
                    answer = response.split('Answer: ')[1]
                    explanation = None if not self.cot else response.split('Explanation: ')[1].split('Answer: ')[0].strip()
                    similarity = int(re.findall(r'\d+', answer)[0])
                    dataset.iloc[i+j]['similarity'] = similarity
                    dataset.iloc[i+j]['explanation'] = explanation
            except: 
                answers = [None] * batch_df.shape[0]
        return list(dataset['similarity'])
        

# ----------------------- 1 - Patient Summary evaluation ----------------------- #
    

def match_list(gold_list, pred_list, scorer_type='bert'):
    '''
    Given two lists of dictionaries, match corresponding dictionaries by score.
    Arguments: 
        - gold_list (list of dict): list of gold dictionaries
        - pred_list (list of dict): list of predicted dictionaries
        - scorer_type (str): type of scorer to use for matching (default: 'bert') 
    '''
    scorer = Scorer([scorer_type])

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
                score = scorer(gold_string, pred_string)[scorer_type]
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


def flatten_dict(gold, pred, parent_key=''):
    '''
    Flattening utility function; 
    Given two dictionaries, match corresponding keys and compute matching score between their values. 

    Arguments: 
        - gold (dict): dictionary of gold values
        - pred (dict): dictionary of predicted values
        - parent_key (str): key of parent dictionary, used for recursion
    Returns:   
        - flat (dict of str: (str, str)): flattened dictionary of matched (gold, pred) values
    '''
    flat = []
    for gold_key, gold_value in gold.items():
        if gold_key in pred.keys(): # Matched pred & gold keys
            if isinstance(gold_value, str):
                    flat.append((f"{parent_key}{gold_key}", (gold_value, pred[gold_key])))               
            if isinstance(gold_value, dict):
                flat.extend(flatten_dict(gold_value, pred[gold_key], parent_key=f"{parent_key}{gold_key}/"))
            if isinstance(gold_value, list):
                matched_gold, matched_pred = match_list(
                    gold_list = gold_value, pred_list = pred[gold_key])
                for i, gold_list_val in enumerate(matched_gold):
                    if isinstance(gold_list_val, str):
                        flat.append(f"{parent_key}{gold_key}/{i}", (gold_list_val, matched_pred[i]))
                    if isinstance(gold_list_val, dict):
                        flat.extend(flatten_dict(gold_list_val, matched_pred[i], 
                                                 parent_key=f"{parent_key}{gold_key}/{i}/"))
        else: # No match for gold key
            if isinstance(gold_value, str):
                flat.append(f"{parent_key}{gold_key}", (gold_value, NO_SUCH_KEY))
            if isinstance(gold_value, dict):
                flat.extend(flatten_dict(gold_value, {}, parent_key=f"{parent_key}{gold_key}/"))
            if isinstance(gold_value, list):
                for i, val in enumerate(gold_value):
                    if isinstance(val, str):
                        flat.append(f"{parent_key}{gold_key}/{i}", (gold_value, NO_SUCH_KEY))
                    if isinstance(val, dict):
                        flat.extend(flatten_dict(
                            val, {}, parent_key=f"{parent_key}{gold_key}/{i}/"))
    # No match for pred key
    for pred_key, pred_value in pred.items():
        if pred_key not in gold.keys():
            if isinstance(pred_value, str):
                flat.append(f"{parent_key}{pred_key}", (NO_SUCH_KEY, pred_value))
            if isinstance(pred_value, dict):
                flat.extend(flatten_dict({}, pred_value, parent_key=f"{parent_key}{pred_key}/"))
            if isinstance(pred_value, list):
                for i, val in enumerate(pred_value):
                    if isinstance(val, str):
                        flat.append(f"{parent_key}{pred_key}/{i}", (NO_SUCH_KEY, pred_value))
                    if isinstance(val, dict):
                        flat.extend(flatten_dict({}, val, parent_key=f"{parent_key}{pred_key}/{i}"))
                        
    return flat if parent_key != '' else dict(flat)

def summary_statistics(gold, pred, score_types=['rouge', 'bleu', 'bert']):
    '''
    Given two patient summaries, flatten and match keys, 
    then compute matching scores & statistics.
    '''
    flat = flatten_dict(gold, pred)
    scorer = Scorer(score_types)
    scores = {}
    stats = {'total': len(flat)}
    for key in ['missing_keys', 'extra_keys', 'common', 'common_none', 'gold_none', 'pred_none']:
        stats[key] = 0
    for key, (gold, pred) in flat.items():
        if gold == NO_SUCH_KEY:
            stats['extra_keys'] += 1
        if pred == NO_SUCH_KEY:
            stats['missing_keys'] += 1
        if gold == NONE_FIELD:
            if pred == NONE_FIELD:
                stats['common_none'] += 1
            else:
                stats['gold_none'] += 1
        else:
            if pred == NONE_FIELD:
                stats['pred_none'] += 1
            else:
                stats['common'] += 1
                scores[key] = scorer([{'gold': gold, 'pred': pred}])
    return scores, stats

def summary_evaluation(path, score_types=['bleu', 'rouge', 'bert']): 
    '''
    1 - Patient summary evaluation
    Run evaluation on a dataframe with 'gold' and 'pred' patient summaries. 

    Arguments: 
        - path (str): path to dataframe with 'gold' and 'pred' patient summaries
        - score_types (str or list): list of scoring functions to be used. (Default: BLEU + ROUGE + BERT)
    '''
    dataset = load_file(path)
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    scores_path = path.replace('.jsonl', '_scores.jsonl')
    for i, row in tqdm(dataset.iterrows(), total=dataset.shape[0], desc="Evaluating pairs of patient summaries"):
        # Compute summary statistics
        scores, stats = summary_statistics(row['gold'], row['pred'], score_types)
        for stat_type, stat in stats.items():
            dataset.loc[i, stat_type] = stat
        
        # Compute average matching score for each metric
        metrics = list(scores[list(scores.keys())[0]].keys())
        for metric in metrics:
            avg_score = np.mean([score[metric] for score in scores.values()])
            dataset.iloc[i][metric] = avg_score
        if i % 10 == 0:
            save_file(dataset, scores_path)
    save_file(dataset, scores_path)
    return dataset

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
    if 'gpt_rank' in score_types:
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
    
    TODO: CHECK THIS RUNS. 
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
        summary_evaluation(args.path)

    elif args.mode == 'clinical_note':
        clinical_note_evaluation(args.path, score_types)

    else:
        raise ValueError(f"Mode {args.mode} is not valid. Please choose between 'summary' and 'clinical_note'.")
    
