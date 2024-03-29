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
from utils.infer import *
from utils.scorer import *
from utils.data import *
from utils.saving_manager import *

import numpy as np
import argparse
from evaluate import load
from multielo import MultiElo, Player
import sys
import os
import json
from tqdm import tqdm
from itertools import combinations
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

BERT_SCORER = load("bertscore")
ROUGE_SCORER = load("rouge")
BLEU_SCORER = load("bleu")

NONE_MATCH  = "None_Match"
NO_SUCH_KEY = "no_such_key"
NONE_FIELD = "None"

SUMMARY_SCORE_TYPES = ['bleu', 'rouge', 'bert', 'gpt_similarity_score']

NOTE_SPECIFIC_SCORE_TYPES = [f"gpt_{score_type}_score" for score_type in ["clarity", "coherence", "factuality"]]
NOTE_ALL_SCORE_TYPES = SUMMARY_SCORE_TYPES + NOTE_SPECIFIC_SCORE_TYPES

ROUGE_SUB_SCORES = ['rouge1', 'rouge2',	'rougeL', 'rougeLsum']

ALL_NOTES_SCORE_TYPES_OUTPUT = [score for score in NOTE_ALL_SCORE_TYPES if score != 'rouge'] + ROUGE_SUB_SCORES
ALL_SUMMARIES_SCORE_TYPES_OUTPUT = [score for score in SUMMARY_SCORE_TYPES if score != 'rouge'] + ROUGE_SUB_SCORES

ALL_SUMMARIES_SCORE_TYPES_OUTPUT_MEAN = [f"mean_{score}" for score in ALL_SUMMARIES_SCORE_TYPES_OUTPUT]

COUNTS_TYPES = ['missing_keys_count', 'extra_keys_count', 'common_none_count',  
                'gold_none_count', 'pred_none_count', 'common_non_none', 'all_keys_count']
KEY_MISMATCH_TYPE = ['gold_none_keys', 'pred_none_keys', 'missing_keys']

EVAL_DIR = 'evaluation'

os.makedirs(EVAL_DIR, exist_ok=True)

NOTE_MODELS = [model for model in MODELS_TO_MODE.keys() if 'summarizer' not in MODELS_TO_MODE[model] ]

SUMMARIZER_MODELS = [model for model in MODELS_TO_MODE.keys() if 'summarizer' in MODELS_TO_MODE[model]]

ELO_MODELS = NOTE_MODELS.copy()

ELO_MODELS.remove('meditron-7b-generator-gold')
ELO_MODELS.remove('meditron-13b-generator-gold')
ELO_MODELS.remove('gpt4-direct')
ELO_MODELS.remove('gpt3-generator-gpt')
ELO_MODELS.remove('gpt3-generator-13b')
ELO_MODELS.remove('gpt3-generator-7b')

# ----------------------- 0 - Prepare Evaluation inputs ----------------------- #

def save_evaluation_input(eval_input_filename, inference_df, pred_data, gold_data):
    eval_input = inference_df[['idx', gold_data, pred_data, 'conversation']].dropna()
    eval_input = eval_input.rename(columns={gold_data: 'gold', pred_data: 'pred'})
    if gold_data == 'summary': #post processing of summaries
        for key in ['gold', 'pred']:
            eval_input[key] = eval_input[key].apply(lambda x: json.loads((x.replace('\n', ' ').replace(
            '""None""', '"None"').replace(
                '""None', '').replace('''", },''','''" },'''))))
    save_file(eval_input, eval_input_filename)

def build_evaluation_inputs(combined_inferences_path, eval_input_dir_path = EVAL_DIR ,models = MODELS_TO_MODE.keys()):
    combined_inference = load_file(combined_inferences_path)
    for model in models:
        if model not in MODELS_TO_MODE.keys():
            raise ValueError(f"Model {model} is not valid. Please choose between {MODELS_TO_MODE.keys()}.")
        save_evaluation_input(f'{eval_input_dir_path}/{model}_evaluation_input.jsonl', combined_inference, MODELS_TO_OUTPUT[model], KEYS[MODELS_TO_MODE[model]]['gold'])
    
    save_elo_inputs('elo_inputs.jsonl', combined_inference)

def save_elo_inputs(output_filename, inference_sample, models_to_compare = ELO_MODELS):
    #Selecting columns to keep
    for model in models_to_compare:
        if 'summarizer' in model:
            raise ValueError("Summarizer cannot be used for Elo ranking.")

    if len(models_to_compare) >=2:
        outputs_notes = inference_sample[['idx'] + [MODELS_TO_OUTPUT[model] for model in models_to_compare]].copy()
        outputs_notes.replace({None: np.nan, '': np.nan}, inplace=True)
        
        #renaming columns
        outputs_notes = outputs_notes.rename(columns={MODELS_TO_OUTPUT[model]: model for model in models_to_compare})
        outputs_notes.index = outputs_notes['idx']
        #Creating a dataframe with all possible pairs
        nb_comparison = 4
        np.random.seed(42)

        outputs_notes['possible_models'] = outputs_notes[models_to_compare].apply(lambda row: row.dropna().index.tolist(), axis=1)

        outputs_notes['random_comparisons'] = outputs_notes['possible_models'].apply(lambda x: np.random.choice(x, size=min(nb_comparison*2, len(x)//2 * 2), replace=False))
        
        outputs_notes['random_comparisons'] = outputs_notes['random_comparisons'].apply(lambda x: [(x[i], x[i+1]) for i in range(0, len(x), 2)])
    
        outputs_notes['notes'] = outputs_notes.apply(lambda row: [(row[modelA], row[modelB]) for (modelA,modelB) in row['random_comparisons']] , axis=1)

        all_pairs_df = outputs_notes[['idx', 'random_comparisons', 'notes']].copy()

        all_pairs_df = all_pairs_df.explode(['random_comparisons', 'notes'])

        all_pairs_df['modelA'] = all_pairs_df['random_comparisons'].apply(lambda x: x[0])
        all_pairs_df['modelB'] = all_pairs_df['random_comparisons'].apply(lambda x: x[1])
        all_pairs_df['noteA'] = all_pairs_df['notes'].apply(lambda x: x[0])
        all_pairs_df['noteB'] = all_pairs_df['notes'].apply(lambda x: x[1])

        all_pairs_df.drop(['random_comparisons', 'notes'], axis=1, inplace=True)
        all_pairs_df = all_pairs_df.reset_index(drop=True)
        #Saving dataframe
        save_file(all_pairs_df, os.path.join(EVAL_DIR, output_filename))

def aggregate_scores(dfs, score_types):
    normalized_scores_dfs = [pd.DataFrame() for _ in dfs]
    normalized_scores = [f"normalized_{metric}" for metric in score_types]
    
    for normalized_metric, metric in zip(normalized_scores, score_types):
        if 'gpt' in metric:
            for norm_df,df in zip(normalized_scores_dfs, dfs):
                norm_df[normalized_metric] = df[metric].apply(lambda x: (x - 1) / 9)
        else:
            for norm_df,df in zip(normalized_scores_dfs, dfs):
                norm_df[normalized_metric] = df[metric]

    concatenated_norm_df = pd.concat(normalized_scores_dfs, axis=0)

    means = concatenated_norm_df.mean(axis=0)
    stds = concatenated_norm_df.std(axis=0)
    rouge_deweight = means.index.to_series().apply(lambda x: 1/len(ROUGE_SUB_SCORES) if 'rouge' in x else 1)

    weights = (1/(means * rouge_deweight))
    
    for norm_df in normalized_scores_dfs:
        norm_df['aggregated_score'] = norm_df.multiply(weights, axis=1).sum(axis=1) / weights.sum()
    
    if 'missing_keys_prop' in dfs[0].columns:
        for norm_df,df in zip(normalized_scores_dfs, dfs):
            norm_df['aggregated_score'] = norm_df['aggregated_score'] * (1 - df['missing_keys_prop'] - df['pred_none_prop'] - df['gold_none_prop'])

    return [norm_df['aggregated_score'].tolist() for norm_df in normalized_scores_dfs]
    
# ----------------------- 1 - Patient Summary evaluation ----------------------- #

def matching_bert_scrorer(pairs):
        '''BERT score for matching lists in the template'''
        '''BERT score for summary evaluation'''
        bert_scores = BERT_SCORER.compute(
                        predictions=[pair['pred'] for pair in pairs],
                        references=[pair['gold'] for pair in pairs],
                        model_type= 'distilbert-base-uncased',
                        verbose = False)['f1']
              
        return bert_scores

def match_list(gold_list, pred_list):
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
    scores = matching_bert_scrorer(pairs)
    scores = np.array(scores).reshape(len(gold_list), len(pred_list))
    #print(f"scores: {scores}")
    min_items = min(len(gold_list), len(pred_list))
    for _ in range(min_items):
        best_match_index = np.unravel_index(np.argmax(scores, axis=None), scores.shape) # get argmax in scores dimensions (2D)
        matched_pred[best_match_index[0]] = pred_list[best_match_index[1]] # add best match pred at right index (matching gold)
        scores[best_match_index[0], :] = 0 # remove scores for that match in gold items
        scores[:, best_match_index[1]] = 0 # remove scores for that match in pred items

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
                if not isinstance(pred[gold_key], dict):
                    if pred[gold_key] == NONE_FIELD:
                        pred[gold_key] = {}
                    
                    else:
                        pred[gold_key] = {pred[gold_key]: NONE_FIELD}
                flat.extend(flatten_dict(gold_value, pred[gold_key], parent_key=f"{parent_key}{gold_key}/"))
            if isinstance(gold_value, list): #We have to match most similar lists
                if not isinstance(pred[gold_key], list):
                    if isinstance(pred[gold_key], dict):
                        pred[gold_key] = [pred[gold_key]]
                    elif pred[gold_key] == NONE_FIELD:
                        pred[gold_key] = []
                    else: 
                        pred[gold_key] = [{pred[gold_key]: NONE_FIELD}]
                matched_gold, matched_pred = match_list(gold_list = gold_value, 
                                                        pred_list = pred[gold_key])
                for i, gold_list_val in enumerate(matched_gold):
                    if isinstance(gold_list_val, str):
                        flat.append((f"{parent_key}{gold_key}/{i}", (gold_list_val, matched_pred[i])))
                    if isinstance(gold_list_val, dict):
                        if matched_pred[i] == NONE_FIELD:
                            matched_pred[i] = {}
                        if not isinstance(matched_pred[i], dict):
                            matched_pred[i] = {matched_pred[i]: NONE_FIELD}
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
    counts['all_keys_count']= len(flattened_dict)
    clean_flat_dict = {}
    key_mismatches = {key : [] for key in KEY_MISMATCH_TYPE}
    for key, (gold, pred) in flattened_dict.items():
        if gold == NO_SUCH_KEY:
            counts['extra_keys_count'] += 1
        elif pred == NO_SUCH_KEY:
            key_mismatches['missing_keys'].append(key)
        elif gold == NONE_FIELD or gold == "":
            if pred == NONE_FIELD or pred == "":
                counts['common_none_count'] += 1
            else:
                key_mismatches['gold_none_keys'].append(key)
        else:
            if pred == NONE_FIELD or pred == "":
                key_mismatches['pred_none_keys'].append(key)
            else:
                clean_flat_dict[key] = (str(gold), str(pred))
    counts['gold_none_count'] = len(key_mismatches['gold_none_keys'])
    counts['pred_none_count'] = len(key_mismatches['pred_none_keys'])
    counts['missing_keys_count'] = len(key_mismatches['missing_keys'])
    counts['common_non_none'] = len(clean_flat_dict)

    return counts, clean_flat_dict, key_mismatches

def summary_statistics(golds, preds, saving_manager, score_types):
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
        
    # Flatten and match keys of each dict pair

    if saving_manager.get_progress_dict()['flatten and match dicts'] != 'done':

        stats_df = pd.concat([golds, preds], axis=1)

        print(f"Flattening summary dictionnaries and matching keys...")
        tqdm.pandas()
        stats_df['flat_dicts'] = stats_df.progress_apply(lambda row: flatten_dict(row['gold'], row['pred']), axis=1)
        stats_df.drop(['gold', 'pred'], axis=1, inplace=True)

        saving_manager.flatten_and_match_dicts_update(stats_df)
    
    else:
        stats_df = saving_manager.load_flatten_and_match_dicts()

    # Compute counts and clean each flattened dict

    if saving_manager.get_progress_dict()['clean dicts and counts'] != 'done':
        print(f"Computing counts and cleaning flattened dictionaries...")
        tqdm.pandas()
        stats_df[['counts','cleaned_flat_dicts','key_mismatches']] = pd.DataFrame(
                            stats_df['flat_dicts'].progress_apply(get_counts_and_clean_dict).tolist(),
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

    else: 
        stats_df = saving_manager.load_clean_dicts_and_counts()

    scorer = Scorer(saving_manager,score_types)
    stats_df['keys'] = stats_df['cleaned_flat_dicts'].apply(lambda x: list(x.keys()))
    stats_df['pairs'] = stats_df['cleaned_flat_dicts'].apply(lambda x: list(x.values()))
    stats_df.drop(['cleaned_flat_dicts'], axis=1, inplace=True)
    
    if saving_manager.get_progress_dict()['pairs_idx'] != 'done':
        #Prepare df to pass to scorer (mainly flattening the list of list of keys and pairs)
        pairs_df = stats_df[['keys', 'pairs']].explode(['keys', 'pairs'])
        pairs_df = pairs_df.dropna()
        pairs_df['pairs'] = pairs_df['pairs'].apply(lambda x: {'gold': x[0], 'pred': x[1]})
        pairs_df = pairs_df['pairs'].to_frame()
        pairs_df['sample_idx'] = pairs_df.index
        pairs_df['idxs'] = range(pairs_df.shape[0])
        saving_manager.save_pairs_idx(pairs_df)
    else:
        pairs_df = saving_manager.load_pairs_idx()
    
    #Compute scores for each gold,pred pair
    if saving_manager.get_progress_dict()['scores'] != 'done':
        scores = scorer(pairs_df)
    else:
        scores = saving_manager.load_all_scores()

    #Unpack scores
    for metric in scores.columns:
        pairs_df[metric] = scores[metric]

    #Group scores by dictionaries as list of scores (mainly unlfattening the list of scores
        #as a list of list of scores)
    grouped_scores = pairs_df.groupby(pairs_df['sample_idx']).agg(lambda x: list(x))

    #Unpack grouped scores
    for metric in grouped_scores.columns:
        stats_df[metric] = grouped_scores[metric]

    saving_manager.save_summary_statistics(stats_df)  
    return stats_df

def summary_evaluation(model_name,save_path = None ,score_types= SUMMARY_SCORE_TYPES): 
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
    path = f'evaluation/{model_name}_evaluation_input.jsonl'

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


    new_score_types = score_types.copy()
    if score_types == 'rouge':
        new_score_types = ROUGE_SUB_SCORES.copy()
    elif 'rouge' in score_types:
        new_score_types.remove('rouge')
        new_score_types.extend(ROUGE_SUB_SCORES)

    # Compute average matching scores accrosss all keys for each metric for each patient summary
    if eval_saving.get_progress_dict()['eval_by_sample'] != 'done':
        mean_scores = [f"mean_{metric}" for metric in new_score_types]
        for mean_metric, metric in zip(mean_scores, new_score_types):
            dataset[mean_metric] = stats[metric].apply(lambda x: np.mean(x))

        for count_type in COUNTS_TYPES:
            dataset[count_type] = stats[count_type]
        
        eval_saving.save_eval_by_sample(dataset)
    else:
        dataset = eval_saving.load_eval_by_sample()

    # Compute average matching scores accross all patient summaries for each metric for each field
    # As well as average counts of missing, none_pred, none_gold
    if eval_saving.get_progress_dict()['eval_by_key'] != 'done':
        exploding1 = ['keys'] + new_score_types
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
        
        score_by_keys = score_by_keys.reset_index()

        eval_saving.save_eval_by_key(score_by_keys)
    else:
        score_by_keys = eval_saving.load_eval_by_key()
    
    return dataset, score_by_keys

def merge_summary_evaluation(model_names):
    '''
    Merge summary evaluations for several models into one dataframe.
    Arguments:
        - model_names (list of str): list of models to merge
    Returns:
        - merged_df (pd.DataFrame): merged dataframe with all models' evaluations means
    '''

    if any([model_name for model_name in model_names if 'summarizer' not in MODELS_TO_MODE[model_name]]):
        raise ValueError("All models must be summarizers.")

    eval_res_paths = [f'evaluation/{model_name}_eval_res/summary_eval_by_sample.jsonl' for model_name in model_names]
    dfs = []
    done_model_names = []
    for path, name in zip(eval_res_paths, model_names):
        try:
            df = load_file(path).reset_index(drop=True)
            dfs.append(df)
            done_model_names.append(name)
        except ValueError:
            print(f"{path} not existing yet")
    
    for agg_score, df, eval_res_path in zip(aggregate_scores(dfs, ALL_SUMMARIES_SCORE_TYPES_OUTPUT_MEAN), dfs, eval_res_paths):
        df['aggregated_score'] = agg_score
        df.to_json(eval_res_path, orient='records', lines=True)

    dfs_score_means = [df[ALL_SUMMARIES_SCORE_TYPES_OUTPUT_MEAN  + ['aggregated_score']].multiply(df['common_non_none'], axis=0).sum() / df['common_non_none'].sum() for df in dfs]

    dfs_count_means = [(df[COUNTS_TYPES].divide(df['all_keys_count'], axis = 0)).mean() for df in dfs]

    dfs_mean_key_count = [df['all_keys_count'].mean() for df in dfs]

    result_df = pd.DataFrame({'model': done_model_names})
    for score in ALL_SUMMARIES_SCORE_TYPES_OUTPUT_MEAN +  ['aggregated_score']:
        result_df[score] = [df_mean.loc[score] for df_mean in dfs_score_means]
    
    for count in COUNTS_TYPES:
        if count == 'all_keys_count':
            result_df["mean_number_of_keys"] = dfs_mean_key_count
        else:
            result_df[f"{count}_prop"] = [df_mean.loc[count] for df_mean in dfs_count_means]
    
    eval_res_paths = [f'evaluation/{model_name}_eval_res/summary_eval_by_key.jsonl' for model_name in model_names]
    dfs = []
    done_model_names = []
    for path, name in zip(eval_res_paths, model_names):
        try:
            df = load_file(path).reset_index(drop=True)
            dfs.append(df)
            done_model_names.append(name)
        except ValueError:
            print(f"{path} not existing yet")
    
    for agg_score, df, eval_res_path in zip(aggregate_scores(dfs, ALL_SUMMARIES_SCORE_TYPES_OUTPUT), dfs, eval_res_paths):
        df['aggregated_score'] = agg_score
        df.to_json(eval_res_path, orient='records', lines=True)

    return result_df


# ----------------------- 2 - Clinical note evaluation ----------------------- #

def note_evaluation(model_name, save_path = None ,score_types= NOTE_ALL_SCORE_TYPES): 
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
    path = f'evaluation/{model_name}_evaluation_input.jsonl'

    eval_saving = EvalSaving(NOTE_EVALUATION_STEPS, path, save_path)

    df = load_file(path)
    df = df.sample(frac=1).reset_index(drop=True)
    df['model_name'] = model_name
    # Compute scores for each pair of gold and pred clinical notes
    scorer = Scorer(eval_saving,score_types)
    
    df['pairs'] = df.apply(lambda row: {'gold': row['gold'], 'pred': row['pred']}, axis=1)
    df['idxs'] = df.index

    if eval_saving.get_progress_dict()['scores'] != 'done':
        scores = scorer(df, remove_stopwords = True)
    else:
        scores = eval_saving.load_all_scores()

    if score_types == 'rouge':
        new_score_types = ROUGE_SUB_SCORES.copy()
    elif 'rouge' in score_types:
            new_score_types = score_types.copy()
            new_score_types.remove('rouge')
            new_score_types.extend(ROUGE_SUB_SCORES)
    else : new_score_types = score_types.copy()

    for metric in new_score_types:
        df[metric] = scores[metric]

    df.drop(['idxs', 'pairs'], axis=1, inplace=True)

    return df

def elo_ranking(path ,save_path = None):
    ''' 
    Elo ranking for clinical note evaluation with GPT-4.
    Taken from https://portkey.ai/blog/comparing-llm-outputs-with-elo-ratings/

    Argument:
        - df (pd.DataFrame): dataframe with GPT-4 scores for each model
    Returns: 
        - rankings (dict of str: float): dictionary of Elo rankings for each model

    '''
    df = load_file(path)

    if save_path is None:
        save_path = path.replace('_inputs.jsonl', '_ranking_res')

    saving_manager = EvalSaving(ELO_RANKING_STEPS, path, save_path)

    if saving_manager.get_progress_dict()['compute elos'] == 'done':
        return saving_manager.load_elos()

    df['pairs'] = df.apply(lambda row: {'noteA': row['noteA'], 'noteB': row['noteB']}, axis=1)

    model_names = list(set(list(df['modelA'].unique()) + list(df['modelB'].unique())))

    if saving_manager.get_progress_dict()['pairs_idx'] != 'done':
        df = df.sample(frac=1).reset_index(drop=True)
        df['idxs'] = df.index
        saving_manager.save_pairs_idx(df)
    else:
        df = saving_manager.load_pairs_idx()

    
    ranker = Scorer(saving_manager,['gpt_rank'])

    ranks = ranker(df)
    ranks.drop(['pairs'], axis=1, inplace=True)

    players = {model: Player(model) for model in model_names}
    elo = MultiElo(k_value = 1)

    # For each score given in df, compute Elo ranking
    for _, row in tqdm(ranks.iterrows(), total=ranks.shape[0], desc="Computing ELO rankings"):
        
        winner = row['gpt_rank']
        if winner == "tie":
            result_order = [1, 1]
        else :
            result_order = [1,2]
        
        if winner != 'pred':
            winner = row['modelA']
            loser = row['modelB']
        else:
            winner = row['modelB']
            loser = row['modelA']
        
        new_ratings = elo.get_new_ratings([players[winner].rating, players[loser].rating],
                            result_order = result_order)
        players[winner].update_rating(new_ratings[0], row['idx'])
        players[loser].update_rating(new_ratings[1], row['idx'])
        
    elo_histories = pd.DataFrame({'model' : model_names,
                                  'elo_history' : [players[model].rating_history for model in model_names]})
    

    elo_histories['score_histories'] = elo_histories['elo_history'].apply(lambda x: [score[1] for score in x])
    elo_histories['final_score'] = elo_histories['elo_history'].apply(lambda x: x[-1][1])

    saving_manager.save_elos(elo_histories)

    return elo_histories

def merge_note_evaluations(model_names):
    ''' 
    Merge note evaluations for several models into one dataframe.
    Arguments:
        - model_names (list of str): list of models to merge
    Returns:
        - merged_df (pd.DataFrame): merged dataframe with all models' evaluations means
    '''

    eval_res_paths = [f'evaluation/{model_name}_eval_res/all_scores.jsonl' for model_name in model_names]
    dfs = []
    done_model_names = []
    for path, name in zip(eval_res_paths, model_names):
        try:
            df = load_file(path).reset_index(drop=True)
            dfs.append(df)
            done_model_names.append(name)
        except ValueError:
            print(f"{path} not existing yet")

    for agg_score, df, eval_res_path in zip(aggregate_scores(dfs, ALL_NOTES_SCORE_TYPES_OUTPUT), dfs, eval_res_paths):
        df['aggregated_score'] = agg_score
        df.to_json(eval_res_path, orient='records', lines=True)

    dfs_means = [df[ALL_NOTES_SCORE_TYPES_OUTPUT + ['aggregated_score']].mean() for df in dfs]
    

    result_df = pd.DataFrame({'model': done_model_names})
    for score in ALL_NOTES_SCORE_TYPES_OUTPUT + ['aggregated_score']:
        result_df[score] = [df_mean.loc[score] for df_mean in dfs_means]

    return result_df   
       
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
    
