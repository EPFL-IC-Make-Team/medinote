'''
Evaluation utilities. 

1- Patient Summary evaluation
Given our model’s patient summary and GPT-4's summary generated from note, 
compute the BLEU/ROUGE/BERT scores for all matching fields in the template. 
We use BERT score to match list elements (e.g. symptoms). 

2- Clinical note evaluation
Here we want to evaluate the quality of our generated clinical notes using GPT-4. Two methods:
Given 2 models’ answers (randomly mixed),  ask GPT-4 to pick which one is the best and compute an ELO score
Given a model’s answer and GPT-4' answer (silver label), ask GPT-4 with CoT to compute the similarity 
between the two answers on a scale of 1 to 10. The higher the number the closer the model’s quality to GPT-4's.

'''

from chat import *
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from bert_score import BERTScorer

none_match  = "None_Match"
no_such_key = "no_such_key"
none_field = "None"

# ----------------------- Generic scorer (BLEU/ROUGE/BERT/GPT-4) ----------------------- #
class Scorer(): 
    def __init__(self, score_types='all'):
        ''' 
        Initializes a scorer with a list of score types to be used.
        If score_types is 'all', all score types will be used.
        Otherwise, use a list of score types to be used. 
        '''
        self.scoring_functions = {
            'bleu': self.BLEU_score,
            'rouge': self.ROUGE_score,
            'bert': self.BERT_score,
            'random': self.random_score,
            'gpt_4_rank': self.GPT4_score_rank,
            'gpt_4_sim': self.GPT4_score_sim
        }
        self.score_types = self.scoring_functions.keys() if score_types == 'all' else score_types

    def evaluate(self, gold, pred):
        '''
        Given a gold and predicted string, returns a dictionary of all scores. 
        '''
        scores = {}
        for score_type in self.score_types:
            scores[score_type] = self.scoring_functions[score_type](gold, pred)
        return scores

    def BLEU_score(self, gold, pred):
        ''' BLEU score for summary evaluation (precision focused)'''
        return sentence_bleu([gold], pred)
    
    def ROUGE_score(self, gold, pred):
        ''' ROUGE score for summary evaluation (recall focused)'''
        return Rouge().get_scores(gold, pred)[0]

    def BERT_score(self, gold, pred):
        scorer = BERTScorer(model_type='bert-base-uncased')
        precision, recall, F1 = scorer.score([pred], [gold])
        return F1.item()
    
    def random_score(self, gold, pred):
        return np.random.random()
    
    def GPT4_score_rank(self, gold, pred):
        ''' 
        Given 2 models’ answers (randomly mixed),  
        ask GPT-4 to pick which one is the best and compute an ELO score
        '''
        raise NotImplementedError
    
    def GPT4_score_sim(self, gold, pred):
        ''' 
        Given a model’s answer and GPT-4' answer (silver label), 
        ask GPT-4 with CoT to compute the similarity between the two answers 
        on a scale of 1 to 10. The higher the number the closer the model’s quality to GPT-4's.
        '''
        raise NotImplementedError
    
    
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

# ----------------------- 2 - Clinical note evaluation ----------------------- #
    
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

def only_non_none_eval_dict(evaluation_dict):
    return {k: v for k, v in evaluation_dict.items() if isinstance(v, dict)} #keeping only the non-none values

def build_gpt_4_scoring_dataset(evaluation_dicts_df):
    gpt_4_scoring_dataset = evaluation_dicts_df[['idxs', 'eval_dict']]
    gpt_4_scoring_dataset['eval_dict'] = build_gpt_4_scoring_dataset['eval_dict'].apply(lambda x: only_non_none_eval_dict(x))
    gpt_4_scoring_dataset['eval_dict'] = build_gpt_4_scoring_dataset['eval_dict'].apply(lambda x: {k: v['gpt_4'] for k, v in x.items()})
    gpt_4_scoring_dataset = build_gpt_4_scoring_dataset.explode('eval_dict')
    gpt_4_scoring_dataset['key'] = build_gpt_4_scoring_dataset['eval_dict'].apply(lambda x: x[0])
    gpt_4_scoring_dataset['gold'] = build_gpt_4_scoring_dataset['eval_dict'].apply(lambda x: x[1][0])
    gpt_4_scoring_dataset['pred'] = build_gpt_4_scoring_dataset['eval_dict'].apply(lambda x: x[1][1])
    gpt_4_scoring_dataset.drop(columns=['eval_dict'])

    return build_gpt_4_scoring_dataset

def input_gpt_4_scores(evaluation_dicts_df, gpt_4_scoring_dataset):
    gpt_4_scoring_dataset = gpt_4_scoring_dataset.groupby('idxs').agg({'key': list, 'score': list})
    evaluation_dicts_df = evaluation_dicts_df.merge(gpt_4_scoring_dataset, on='idxs')
    evaluation_dicts_df['eval_dict'] = evaluation_dicts_df[['eval_dict']].apply(
        lambda x: {k: v for k, v in x['eval_dict'].items() if k in x['key']}, axis=1)
    


    