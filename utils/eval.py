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

COUNTS_TYPES = ['missing_keys', 'extra_keys', 'common_none',  'gold_none', 'pred_none', 'common', 'total']

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
        returns a dataframe with the different computed scores.

        Example usage: 
            pairs = [{'gold': 'x', 'pred': 'y'}]
            scorer = Scorer(['bleu', 'rouge', 'bert'])
            scores = await scorer(pairs)
            --> scores = [{'bleu': 0.5, 'rouge': 0.3, 'bert': 0.7}]
        '''
        pairs_df = pd.DataFrame(pairs)
        if 'bleu' in self.score_types:
            pairs_df['bleu'] = self.BLEU_scorer(pairs)['bleu']
        if 'rouge' in self.score_types:
            rouges = self.ROUGE_scorer(pairs)
            for metric in rouges.keys():
                pairs_df[metric] = rouges[metric]
        if 'bert' in self.score_types:
            pairs_df['bert'] = self.BERT_scorer(pairs)['bert']
        if 'gpt_rank' in self.score_types:
            pairs_df['gpt_rank'] = self.GPT_ranker(pairs)
        if 'gpt_score' in self.score_types:
            pairs_df['gpt_score'] = self.GPT_scorer(pairs)

        return pairs_df
    
    def BLEU_scorer(self, pairs):
        ''' BLEU score for summary evaluation (precision-oriented)'''
        bleu_scores = {'bleu': [BLEU_SCORER.compute(predictions=[pair['pred']],
                                                    references=[pair['gold']])['bleu'] 
                                                for pair in tqdm(pairs, total = len(pairs) ,desc="Computing BLEU scores")]}
        return bleu_scores
    
    def ROUGE_scorer(self, pairs):
        ''' ROUGE score for summary evaluation (recall-oriented)'''
        rouges = [ROUGE_SCORER.compute(
                predictions=[pair['pred']], references=[pair['gold']])
                for pair in tqdm(pairs, total = len(pairs) ,desc="Computing ROUGE scores")]
        
        metrics = rouges[0].keys()
        scores = {metric: [rouge[metric] for rouge in rouges] for metric in metrics} 
        return scores

    def BERT_scorer(self, pairs):
        print("Computing BERT scores...")
        scores = {'bert': 
            BERT_SCORER.compute(
                predictions=[pair['pred'] for pair in pairs],
                references=[pair['gold'] for pair in pairs],
                lang='en')['f1']
        }
        print('BERTscores computed.')
        return scores
    
    def GPT_ranker(self, 
                   pairs,
                   model_name='gpt-4-1106-preview', 
                   max_tokens = 300000,
                   one_call_batch_size = 20, 
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
        dataset = pd.DataFrame({'pairs_list': [pairs[i: i+one_call_batch_size] for i in range(0, len(pairs), one_call_batch_size)]})

        dataset[['winner', 'explanation', 'switch', 'model_name']] = [None, None, None, model_name]

        messages = []
        for i, pair_list in tqdm(dataset.iterrows(), total=dataset.shape[0], desc="Building score prompts"): 
            golds = [pair['gold'] for pair in pair_list['pairs_list']]
            preds = [pair['pred'] for pair in pair_list['pairs_list']]
            switches = [np.random.choice([0, 1]) for _ in range(len(golds))]
            optionsA = [gold if switch == 0 else pred for gold, pred, switch in zip(golds, preds, switches)]
            optionsB = [gold if switch == 1 else pred for gold, pred, switch in zip(golds, preds, switches)]
            
            sys_prompt = f"For each pair of clinical notes (pair i, NoteA, NoteB) you are given, compare them and rank which one is of higher quality. "
            if self.cot:
                sys_prompt += "Explain in one sentence your reasoning in comparing the two clinical notes, then select your final answer.\n \
                    Format your response as follows: a list of dictionnaries [{pair_number : i, explanation: <your explanation>, higher_quality_note: <your answer>}]."
            else:
                sys_prompt += "Directly respond with your final answers as follows: a list of dictionnaries [{pair_number : i, higher_quality_note: <your answer>}]"
            
            sys_prompt += "<your answer should be 'NoteA' if NoteA has better quality, 'NoteB' if NoteB has better quality, or 'tie' if they have the same quality."
            usr_prompt = '\n'.join([f"(pair {i}, {optionA}, {optionB})" for i, (optionA, optionB) in enumerate(zip(optionsA, optionsB))])
            
            messages.append(build_messages(sys_prompt, usr_prompt))
            dataset.at[i,'switch'] = switches
        
        print("Creating sub-batches...")
        # Builds a partitions which have total number of tokens < max_tokens
        sub_batches = partition(dataframe = pd.concat([pd.DataFrame({'messages': messages}),dataset['switch']], axis =1), max_token_per_partition=max_tokens,model = model_name)

        # Generate answers by batches
        for i, (sub_batch, nb_tokens) in enumerate(sub_batches):
            print(f"Sub_batch {i+1}/{len(sub_batches)}: {sub_batch.shape[0]} calls, {nb_tokens} total tokens: {nb_tokens/1000 * 0.01}$")
            try:
                answers = generate_answers(
                    messages_list = sub_batch['messages'].tolist(),
                    formatting=self.ranker_formatting,
                    chat=chat_gpt_3,#chat_gpt_4_turbo,
                    temperature=temperature
                ) # answer is list of list
                explanations =  [None] * sub_batch.shape[0] if not self.cot else [answer[1] for answer in answers]
                winners = [['tie' if (answer == 'tie') else
                        'gold' if (answer == 'NoteA' and switch == 0) or (answer == 'NoteB' and switch == 1) else
                        'pred' for answer,switch in zip(winner_list, switches)]
                        for winner_list, switches in zip(answers, sub_batch['switch'].tolist())]
                
                dataset.loc[sub_batch.index,'winner'] = pd.Series(winners, index = sub_batch.index)
                dataset.loc[sub_batch.index,'explanations'] = pd.Series(explanations, index = sub_batch.index)
            except Exception as e:
                print(e)
        return sum(dataset['winner'], []) #concateanate the similarities list of list into one list
    
    def ranker_formatting(self, answer):
        """Format the ranking answer from GPT-4
        to get the winner and explanaion if cot as list of int/string""" 
        winner_pattern = r"higher_quality_note: '([^']+)'"
        winners = re.findall(winner_pattern, answer)
        if len(winners) == 0:
            raise ValueError(f"Invalid format {answer}.")
        if self.cot:
            explanation_pattern = r"explanation: '([^']+)'"
            explanations = re.findall(explanation_pattern, answer)
            if len(explanations) != len(winners):
                raise ValueError(f"Invalid format {answer}.")
            return winners, explanations
        else:
            return winners

    def scorer_formatting(self, answer):
        """Format the scoring answer from GPT-4 
        to get the similarity scores and explanaion if cot as list of int/string"""
        similarity_score_pattern = r"similarity_score:\s*(\d+)"
        similarity_scores = re.findall(similarity_score_pattern, answer)
        int_answers = [int(score) for score in similarity_scores]
        if len(int_answers) == 0:
            raise ValueError(f"Invalid format {answer}.")
        if self.cot:
            explanation_pattern = r"explanation: '([^']+)'"
            explanations = re.findall(explanation_pattern, answer)
            if len(explanations) != len(int_answers):
                raise ValueError(f"Invalid format {answer}.")
            return int_answers, explanations
        else:
            return int_answers
  

    
    def GPT_scorer(self, 
                   pairs,
                   model_name='gpt-4-1106-preview',
                   temperature=0.0,
                   max_tokens = 700,
                   one_call_batch_size=10):
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
        dataset = pd.DataFrame({'pairs_list': [pairs[i: i+one_call_batch_size] for i in range(0, len(pairs), one_call_batch_size)]})

        dataset[['similarity', 'explanation', 'model_name']] = [None , None, model_name]
        messages = []
        for _, pair_list in tqdm(dataset.iterrows(), total=dataset.shape[0], desc="Building score prompts"): 
            golds = [pair['gold'] for pair in pair_list['pairs_list']]
            preds = [pair['pred'] for pair in pair_list['pairs_list']]
            switches = [np.random.choice([0, 1]) for _ in range(len(golds))]
            optionsA = [gold if switch == 0 else pred for gold, pred, switch in zip(golds, preds, switches)]
            optionsB = [gold if switch == 1 else pred for gold, pred, switch in zip(golds, preds, switches)]
            sys_prompt = f"For each pair of clinical notes (pair i, NoteA, NoteB) you are given, compare them and rate how similar they are to each other on a scale of 1 to 10. "
            
            if self.cot:
                sys_prompt += "Explain in one sentence your reasoning in comparing the two clinical notes, then select your final answer.\n \
                    Format your response as follows: a list of dictionnaries [{pair_number : i, explanation: <your explanation>, similarity_score: <your answer>}]"
            else:
                sys_prompt += "Directly respond with your final answers as follows: a list of dictionnaries [{pair_number : i, similarity_score: <your answer>}]"
            
            usr_prompt = '\n'.join([f"(pair {i}, {optionA}, {optionB})" for i, (optionA, optionB) in enumerate(zip(optionsA, optionsB))])
            messages.append(build_messages(sys_prompt, usr_prompt))
        
        sub_batches = partition(dataframe = pd.DataFrame({'messages': messages}), max_token_per_partition=max_tokens,model = model_name)
        # Generate answers by batches
        for i, (sub_batch, nb_tokens) in enumerate(sub_batches):
            print(f"Sub_batch {i+1}/{len(sub_batches)}: {sub_batch.shape[0]} calls, {nb_tokens} total tokens: {nb_tokens/1000 * 0.01}$")
            try:
                answers = generate_answers(
                    messages_list = sub_batch['messages'].tolist(),
                    formatting=self.scorer_formatting,
                    chat=chat_gpt_3,#chat_gpt_4_turbo,
                    temperature=temperature
                ) # answer is list of list
                explanations =  [None] * sub_batch.shape[0] if not self.cot else [answer[1] for answer in answers]
                similarities = answers if not self.cot else [answer[0] for answer in answers]
                dataset.loc[sub_batch.index,'similarity'] = pd.Series(similarities, index = sub_batch.index)
                dataset.loc[sub_batch.index,'explanation'] = pd.Series(explanations, index = sub_batch.index)
            except Exception as e:
                print(e)
        return sum(dataset['similarity'], []) #concateanate the similarities list of list into one list



        

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
                gold_string = ", ".join(str(value) for value in gold_item.values() if value is not NONE_FIELD)
                pred_string = ", ".join(str(value) for value in pred_item.values() if value is not NONE_FIELD)
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
    Given two dictionaries, match corresponding keys and flatten them.

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
                        flat.append((f"{parent_key}{gold_key}/{i}", (gold_list_val, matched_pred[i])))
                    if isinstance(gold_list_val, dict):
                        flat.extend(flatten_dict(gold_list_val, matched_pred[i], 
                                                 parent_key=f"{parent_key}{gold_key}/{i}/"))
        else: # No match for gold key
            if isinstance(gold_value, str):
                flat.append((f"{parent_key}{gold_key}", (gold_value, NO_SUCH_KEY)))
            if isinstance(gold_value, dict):
                flat.extend(flatten_dict(gold_value, {}, parent_key=f"{parent_key}{gold_key}/"))
            if isinstance(gold_value, list):
                for i, val in enumerate(gold_value):
                    if isinstance(val, str):
                        flat.append((f"{parent_key}{gold_key}/{i}", (gold_value, NO_SUCH_KEY)))
                    if isinstance(val, dict):
                        flat.extend(flatten_dict(
                            val, {}, parent_key=f"{parent_key}{gold_key}/{i}/"))
    # No match for pred key
    for pred_key, pred_value in pred.items():
        if pred_key not in gold.keys():
            if isinstance(pred_value, str):
                flat.append((f"{parent_key}{pred_key}", (NO_SUCH_KEY, pred_value)))
            if isinstance(pred_value, dict):
                flat.extend(flatten_dict({}, pred_value, parent_key=f"{parent_key}{pred_key}/"))
            if isinstance(pred_value, list):
                for i, val in enumerate(pred_value):
                    if isinstance(val, str):
                        flat.append((f"{parent_key}{pred_key}/{i}", (NO_SUCH_KEY, pred_value)))
                    if isinstance(val, dict):
                        flat.extend(flatten_dict({}, val, parent_key=f"{parent_key}{pred_key}/{i}"))
                        
    return flat if parent_key != '' else dict(flat)

def get_counts_and_clean_dict(flattened_dict):
    '''
    Given a flattened dictionary, compute the number of missing keys, extra keys, 
    common keys with None value, common keys with None value in gold, common keys with None value in pred.
    also returns remaining common keys (with no nones) for further evaluation.
    '''
    counts = {key: 0 for key in COUNTS_TYPES}
    counts['total']= len(flattened_dict)
    clean_flat_dict = {}
    for key, (gold, pred) in flattened_dict.items():
        if gold == NO_SUCH_KEY:
            counts['extra_keys'] += 1
        if pred == NO_SUCH_KEY:
            counts['missing_keys'] += 1
        if gold == NONE_FIELD:
            if pred == NONE_FIELD:
                counts['common_none'] += 1
            else:
                counts['gold_none'] += 1
        else:
            if pred == NONE_FIELD:
                counts['pred_none'] += 1
            else:
                counts['common'] += 1
                clean_flat_dict[key] = (gold, pred)

    return counts, clean_flat_dict

def summary_statistics(golds, preds, score_types=['rouge', 'bleu', 'bert']):
    '''
    Given several gold,pred patient summaries, flatten and match keys, 
    then compute matching scores & counts.
    Retunrs a pandas dataframe with:
        - a row for each gold,pred dictionary pair,
        - a column scores containing a dictionary of scores for each key
    '''
    if golds.shape[0] != preds.shape[0]:
        raise ValueError("Gold and pred lists must be of same length.")
    
    stats_df = pd.concat([golds, preds], axis=1)
    stats_df['flat_dicts'] = stats_df.apply(lambda row: flatten_dict(row['gold'], row['pred']), axis=1)
    stats_df.drop(['gold', 'pred'], axis=1, inplace=True)

    stats_df[['counts','cleaned_flat_dicts']] = pd.DataFrame(
                        stats_df['flat_dicts'].apply(get_counts_and_clean_dict).tolist(),
                        columns=['counts', 'cleaned_flat_dicts'])
    
    stats_df.drop(['flat_dicts'], axis=1, inplace=True)

    for count_type in COUNTS_TYPES:
        stats_df[count_type] = stats_df['counts'].apply(lambda x: x[count_type])
    stats_df.drop(['counts'], axis=1, inplace=True)
    
    scorer = Scorer(score_types)

    stats_df['keys'] = stats_df['cleaned_flat_dicts'].apply(lambda x: list(x.keys()))
    stats_df['pairs'] = stats_df['cleaned_flat_dicts'].apply(lambda x: list(x.values()))
    stats_df.drop(['cleaned_flat_dicts'], axis=1, inplace=True)


    pairs_df = stats_df[['keys', 'pairs']].explode(['keys', 'pairs'])
    pairs_df['pairs'] = pairs_df['pairs'].apply(lambda x: {'gold': x[0], 'pred': x[1]})
    scores = scorer(pairs_df['pairs'])
    for metric in scores.columns:
        pairs_df[metric] = scores[metric]

    grouped = pairs_df.groupby(pairs_df.index).agg(lambda x: list(x))

    for metric in scores.columns:
        stats_df[metric] = grouped[metric]
    
    display(stats_df)

    return stats_df

def summary_evaluation(path, score_types='all'):#['bleu', 'rouge', 'bert']): 
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

    #save_file(dataset, scores_path)
    stats = summary_statistics(dataset['gold'], dataset['pred'], score_types)

    # Compute average matching scores accrosss all field for each metric for each patient summary
    dataset['scores'] = stats['scores']
    for metric in score_types:
        dataset[metric] = stats['scores'].apply(lambda x: np.mean([scores[metric] for scores in x.values()]))
    
    for count_type in COUNTS_TYPES:
        dataset[count_type] = stats[count_type]
    #Compute average matching scores accross all patient summaries for each metric for each field
    scores_by_keys = stats.explode('scores').groupby('keys').aggregate(lambda x: np.mean(x.tolist()))
    
    
    return dataset, scores_by_keys

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
    
