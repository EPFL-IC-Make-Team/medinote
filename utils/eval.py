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
from utils.scorer import *
from utils.data import *
from utils.saving_manager import *

import numpy as np
import argparse
from evaluate import load
from multielo import MultiElo
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

BERT_SCORER = load("bertscore")
ROUGE_SCORER = load("rouge")
BLEU_SCORER = load("bleu")

NONE_MATCH  = "None_Match"
NO_SUCH_KEY = "no_such_key"
NONE_FIELD = "None"
ALL_SCORE_TYPES = ['bleu', 'rouge', 'bert', 'gpt_rank', 'gpt_score']
ROUGE_SUB_SCORES = ['rouge1', 'rouge2',	'rougeL', 'rougeLsum']
COUNTS_TYPES = ['missing_keys_count', 'extra_keys_count', 'common_none_count',  
                'gold_none_count', 'pred_none_count', 'common', 'total']
KEY_MISMATCH_TYPE = ['gold_none_keys', 'pred_none_keys', 'missing_keys']



EVAL_DIR = '../evaluation'
os.makedirs(EVAL_DIR, exist_ok=True)

# ----------------------- 1 - Patient Summary evaluation ----------------------- #
    

def match_list(gold_list, pred_list, scorer_type='bert'):
    '''
    Given two lists of (sub)-dictionaries, matches corresponding dictionaries by maximum score.
    Arguments: 
        - gold_list (list of dict): list of gold (sub)-dictionaries
        - pred_list (list of dict): list of predicted (sub)-dictionaries
        - scorer_type (str): type of scorer to use for matching (default: 'bert')

    Output:
        - matched_gold (list of dict): same as gold_list,
                        extended with empty dictionnaries if pred_list is longer
        - matched_pred (list of dict): pred_list reordered to match gold_list,
                        extended with empty dictionnaries if gold_list is longer 
    '''
    scorer = Scorer([scorer_type])

    if len(gold_list) == 1 and len(pred_list) == 1:
        return gold_list, pred_list
    if len(gold_list) == 0:
        matched_gold = [{}] * len(pred_list)
        return matched_gold, pred_list
    if len(pred_list) == 0:
        matched_pred = [{}] * len(gold_list)
        return gold_list, matched_pred

    max_items = max(len(gold_list), len(pred_list))
    matched_gold = gold_list.copy() + [{}] * (max_items - len(gold_list))
    matched_pred = [{}] * max_items

    gold_strings = [", ".join(str(value) for value in gold_item.values() if value is not NONE_FIELD) for gold_item in gold_list]
    pred_strings = [", ".join(str(value) for value in pred_item.values() if value is not NONE_FIELD) for pred_item in pred_list]
    pairs = [{'gold' : gold_string, 'pred' : pred_string} for gold_string in gold_strings for pred_string in pred_strings]
    #print(f"total pairs: {len(pairs)}, all pairs: {pairs}")
    scores = scorer(pairs)[scorer_type]
    scores = np.array(scores).reshape(len(gold_list), len(pred_list))
    #print(f"scores: {scores}")
    min_items = min(len(gold_list), len(pred_list))
    for _ in range(min_items):
        best_match_index = np.unravel_index(np.argmax(scores, axis=None), scores.shape) # get argmax in scores dimensions (2D)
        matched_pred[best_match_index[0]] = pred_list[best_match_index[1]] # add best match pred at right index (matching gold)
        scores[best_match_index[0], :] = 0 # remove scores for that match in gold items
        scores[:, best_match_index[1]] = 0 # remove scores for that match in pred items

    #print(f"new_matched_gold: {matched_gold}")
    #print(f"new_matched_pred: {matched_pred}")

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
            if isinstance(gold_value, list): #We have to match most similar lists
                matched_gold, matched_pred = match_list(gold_list = gold_value, 
                                                        pred_list = pred[gold_key])
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
    also returns remaining common keys (with no nones) amd missing ones for further evaluation.
    Outputs:
        - counts (dict of str: int): dictionaries of counts mentionned above
        - clean_flat_dict (dict of str: (str, str)): flattened dictionary of    
                                    matched (gold, pred) values that had no None values
        - key_mismatches (dict of str: list of str): dictionary of lists of mismatched keys
                                    (either missing keys, or with unexpected None values)
    '''

    counts = {key: 0 for key in COUNTS_TYPES}
    counts['total']= len(flattened_dict)
    clean_flat_dict = {}
    key_mismatches = {key : [] for key in KEY_MISMATCH_TYPE}
    for key, (gold, pred) in flattened_dict.items():
        if gold == NO_SUCH_KEY:
            counts['extra_keys_count'] += 1
        if pred == NO_SUCH_KEY:
            key_mismatches['missing_keys'].append(key)
        if gold == NONE_FIELD:
            if pred == NONE_FIELD:
                counts['common_none_count'] += 1
            else:
                key_mismatches['gold_none_keys'].append(key)
        else:
            if pred == NONE_FIELD:
                key_mismatches['pred_none_keys'].append(key)
            else:
                counts['common'] += 1
                clean_flat_dict[key] = (gold, pred)
    counts['gold_none_count'] = len(key_mismatches['gold_none_keys'])
    counts['pred_none_count'] = len(key_mismatches['pred_none_keys'])
    counts['missing_keys_count'] = len(key_mismatches['missing_keys'])

    print(clean_flat_dict)
    return counts, clean_flat_dict, key_mismatches

def summary_statistics(golds, preds, saving_manager, score_types=['rouge', 'bleu', 'bert', 'gpt_score']):
    '''
    Given several (gold,pred) patient summaries pairs, flatten and match keys, 
    then compute matching scores & counts.
    Returns a pandas dataframe with:
        - a row for each gold,pred dictionary pair
        - a column for the (list of) keys which were matched and value were scored
        - a column for each (list of) value pairs of the matched keys adn which were scored
        - a column for each (list of) unmatcehd keys
        - a column for each counts type that is computed
        - a column for each (list of) scores type that is computed
    
    NOTE: raw output not meant to be used but function meant to be used in summary_evaluation
    '''
    if golds.shape[0] != preds.shape[0]:
        raise ValueError("Gold and pred lists must be of same length.")
    
    if score_types == 'all' or 'gpt_rank' in score_types:
        raise ValueError("GPT-4 ranking makes no sense for summary evaluation. \
                         Please choose in 'bleu', 'rouge', 'bert' and 'gpt_score'.")
    
    # Flatten and match keys of each dict pair

    if saving_manager.get_progress_dict()['flatten and match dicts'] != 'done':

        stats_df = pd.concat([golds, preds], axis=1)

        print(f"Flattening summary dictionnaries and matching keys...")
        stats_df['flat_dicts'] = stats_df.apply(lambda row: flatten_dict(row['gold'], row['pred']), axis=1)
        stats_df.drop(['gold', 'pred'], axis=1, inplace=True)

        saving_manager.flatten_and_match_dicts_update(stats_df)
    
    elif saving_manager.get_progress_dict()['clean dicts and counts'] != 'done':
        stats_df = saving_manager.load_flatten_and_match_dicts()

    # Compute counts and clean each flattened dict

    if saving_manager.get_progress_dict()['clean dicts and counts'] != 'done':
        print(f"Computing counts and cleaning flattened dictionaries...")
        stats_df[['counts','cleaned_flat_dicts','key_mismatches']] = pd.DataFrame(
                            stats_df['flat_dicts'].apply(get_counts_and_clean_dict).tolist(),
                            columns=['counts', 'cleaned_flat_dicts', 'key_mismatches'])
        
        stats_df.drop(['flat_dicts'], axis=1, inplace=True)
        # Unpack counts, key_mismatches and matched (key, (gold,pred)) pairs
        for key_mismatch in KEY_MISMATCH_TYPE:
            stats_df[key_mismatch] = stats_df['key_mismatches'].apply(lambda x: x[key_mismatch])
        
        stats_df.drop(['key_mismatches'], axis=1, inplace=True)

        for count_type in COUNTS_TYPES:
            stats_df[count_type] = stats_df['counts'].apply(lambda x: x[count_type])
        stats_df.drop(['counts'], axis=1, inplace=True)

        saving_manager.clean_dicts_and_counts_update(stats_df)

    elif saving_manager.get_progress_dict()['summary_statistics'] != 'done': 
        stats_df = saving_manager.load_clean_dicts_and_counts()

    scorer = Scorer(score_types)

    print(f"Initialized scorer with modes: {list(scorer.score_types)}.")

    stats_df['keys'] = stats_df['cleaned_flat_dicts'].apply(lambda x: list(x.keys()))
    stats_df['pairs'] = stats_df['cleaned_flat_dicts'].apply(lambda x: list(x.values()))
    stats_df.drop(['cleaned_flat_dicts'], axis=1, inplace=True)
    
    #Prepare df to pass to scorer (mainly flattening the list of list of keys and pairs)
    pairs_df = stats_df[['keys', 'pairs']].explode(['keys', 'pairs'])
    pairs_df['pairs'] = pairs_df['pairs'].apply(lambda x: {'gold': x[0], 'pred': x[1]})
    
    #Compute scores for each gold,pred pair
    scores = scorer(pairs_df['pairs'])

    #Unpack scores
    for metric in scores.columns:
        pairs_df[metric] = scores[metric]

    #Group scores by dictionaries as list of scores (mainly unlfattening the list of scores
        #as a list of list of scores)
    grouped_scores = pairs_df.groupby(pairs_df.index).agg(lambda x: list(x))

    #Unpack grouped scores
    for metric in grouped_scores.columns:
        stats_df[metric] = grouped_scores[metric]
        
    return stats_df

def summary_evaluation(path, save_path = None ,score_types=['bleu', 'rouge', 'bert', 'gpt_score']): 
    '''
    1 - Patient summary evaluation
    Run evaluation on a dataframe with 'gold' and 'pred' patient summaries. 

    Arguments: 
        - path (str): path to dataframe with 'gold' and 'pred' patient summaries
        - score_types (str or list): list of scoring functions to be used.
            (Default: BLEU + ROUGE + BERT + GPT-4 score). Cannot be 'gpt_rank'
    
    Returns two pandas dataframes:
        - dataset (pd.DataFrame): origninal dataframe with evaluation results:
            - a row for each gold,pred dictionary pair
            - a column for each (mean) score computed accros keys who were matched
            - a column for each count about key matching, nones etc.
        - score_by_keys (pd.DataFrame): dataframe with evaluation results by key
            - a row for each (gold) key
            - a column for each (mean) score
                    computed accros all patient summaries for this key
            - a column for each prorpotion about key matching, nones etc.
                    computed accros all patient summaries for this key 
                    e.g. "proprtion of pred summaries where this key is mssing"
    '''

    if score_types == 'all' or 'gpt_rank' in score_types:
        raise ValueError("GPT-4 ranking makes no sense for summary evaluation. \
                         Please choose in 'bleu', 'rouge', 'bert' and 'gpt_score'.")

    print(f"Running summary evaluation with score types: {score_types}")
    print(f"Loading data...")
    dataset = load_file(path)
    eval_saving = EvalSaving(SUMMARY_EVALUATION_STEPS, path, save_path)

    if eval_saving.get_progress_dict()['summary_statistics'] != 'done':    
        stats = summary_statistics(golds=dataset['gold'],
                                   preds=dataset['pred'],
                                   saving_manager=eval_saving,
                                   score_types=score_types)
    else:
        stats = eval_saving.load_summary_statistics()
    
    if 'rouge' in score_types:
        score_types.remove('rouge')
        score_types.extend(ROUGE_SUB_SCORES)

    # Compute average matching scores accrosss all keys for each metric for each patient summary
    if eval_saving.get_progress_dict()['eval_by_sample'] != 'done':

        for metric in score_types:
            dataset[f"mean_{metric}"] = stats[metric].apply(lambda x: np.mean(x))
        
        for count_type in COUNTS_TYPES:
            dataset[count_type] = stats[count_type]
        eval_saving.save_eval_by_sample(dataset)
    else:
        dataset = eval_saving.load_eval_by_sample()
    
    # Compute average matching scores accross all patient summaries for each metric for each field
    # As well as average counts of missing, none_pred, none_gold
    
    if eval_saving.get_progress_dict()['eval_by_key'] != 'done':
        exploding1 = ['keys'] + score_types
        score_by_keys = stats[exploding1].explode(exploding1)
        score_by_keys = score_by_keys.groupby(['keys']).agg('mean')
        N = dataset.shape[0]
        score_by_keys = score_by_keys.merge(
            stats['gold_none_keys'].explode('gold_none_keys').value_counts().rename('gold_none_prop')/N,
            how='left',
            left_on='keys', 
            right_index=True).fillna(0)
        score_by_keys = score_by_keys.merge(
            stats['pred_none_keys'].explode('pred_none_keys').value_counts().rename('pred_none_prop')/N,
            how='left',
            left_on='keys',
            right_index=True).fillna(0)
        score_by_keys = score_by_keys.merge(
            stats['missing_keys'].explode('missing_keys').value_counts().rename('missing_keys_prop')/N,
            how='left',
            left_on='keys',
            right_index=True).fillna(0)
        
        eval_saving.save_eval_by_key(score_by_keys)
    else:
        score_by_keys = eval_saving.load_eval_by_key()
    
    return dataset, score_by_keys

# ----------------------- 2 - Clinical note evaluation ----------------------- #
    
    
def note_evaluation(model_name, path, score_types='all'): 
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

    pairs = pd.Series([{'gold': gold, 'pred': pred} for gold, pred in zip(df['gold'], df['pred'])])

    # Compute scores for each pair of gold and pred clinical notes
    scorer = Scorer(score_types)
    scores = scorer(pairs)

    if score_types == 'all':
        score_types = ALL_SCORE_TYPES
    elif score_types == 'rouge':
        score_types = ROUGE_SUB_SCORES
    else:   
        if 'rouge' in score_types:
            score_types.remove('rouge')
            score_types.extend(ROUGE_SUB_SCORES)

    for metric in score_types:
        df[metric] = scores[metric]

    # Compute ELO ranking from GPT-4 scores
    '''if 'gpt_rank' in score_types:
        rankings_path = path.replace('.jsonl', '_rankings.jsonl')
        rankings = elo_ranking(df)
        print(f'ELO rankings: {rankings}')
        with open(rankings_path, 'w') as f:
            json.dump(rankings, f)
            print(f'Saved ELO rankings to {rankings_path}')'''
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
        rank_score = row['gpt_rank']
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
                        help='summary or note')
    parser.add_argument('--path', 
                        type=str, 
                        default='data/evaluation/summary_evaluation.jsonl', 
                        help='path to dataframe with evaluation results')
    parser.add_argument('--score_types', 
                        type=str,
                        default='all', 
                        help='List of scoring functions to be used (choices: bleu, rouge, bert, gpt_rank, gpt_score). \
                            \nDefault: all (all scoring functions). Format example: "bleu, rouge, bert"')
    args = parser.parse_args()
    score_types = args.score_types.split(', ')

    print(f'Running evaluation on {args.mode} with score types: {score_types}')

    if args.mode == 'summary':
        summary_evaluation(args.path)

    elif args.mode == 'note':
        note_evaluation(args.path, score_types)

    else:
        raise ValueError(f"Mode {args.mode} is not valid. Please choose between 'summary' and 'note'.")
    
