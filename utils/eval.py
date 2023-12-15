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

from chat import *
import numpy as np
import re
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from bert_score import BERTScorer
from multielo import MultiElo
import matplotlib.pyplot as plt

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
            #'random': self.random_score,
            'gpt_4_rank': self.GPT4_score_rank,
            'gpt_4_sim': self.GPT4_score_sim
        }
        self.score_types = self.scoring_functions.keys() if score_types == 'all' else score_types
        print('Initialized scorer with score types: ', self.score_types)

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
        ask GPT-4 to pick which one is the best. 
        Returns a dictionary with the winner and the explanation.
        NOTE: we randomly mix gold and pred to avoid bias.
        ELO ranking is computed during the evaluation.
        '''
        switch = np.random.choice([0, 1])
        option1 = gold if switch == 0 else pred
        option2 = gold if switch == 1 else pred
        sys_prompt = f"Compare the two clinical notes and rank which one is of higher quality. \n\nAnswer 1: {option1}\n\nAnswer 2: {option2}\n\nAnswer 3: They are equally good."
        usr_prompt = "Please concisely explain your reasoning in comparing the two clinical notes, then select your final answer. \
            Format your response as follows: \n\nExplanation: <your explanation>\n\nAnswer: <your answer>"
        chat = Chat(model_name='gpt-4-turbo')
        try: 
            response = chat(sys_prompt, usr_prompt, temperature=0.0) # TEMPERATURE 0?
            answer = response.split('\nAnswer: ')[1]
            explanation = response.split('Explanation: ')[1].split('\n\nAnswer: ')[0]
            if '1' in answer: 
                winner = 'gold' if switch == 0 else 'pred'
            elif '2' in answer:
                winner = 'gold' if switch == 1 else 'pred'
            elif '3' in answer:
                winner = 'tie'
            else:
                return None
            return {'winner': winner, 'explanation': explanation}
        except:
            return None
    
    
    def GPT4_score_sim(self, gold, pred):
        ''' 
        Given a model’s answer and GPT-4' answer (silver label), 
        ask GPT-4 with CoT to compute the similarity between the two answers 
        on a scale of 1 to 10. The higher the number the closer the model’s quality to GPT-4's.
        NOTE: we randomly mix gold and pred to avoid bias.
        '''
        switch = np.random.choice([0, 1])
        option1 = gold if switch == 0 else pred
        option2 = gold if switch == 1 else pred
        sys_prompt = f"Compare the two clinical notes and rate how similar they are to each other on a scale of 1 to 10. \n\nAnswer 1: {option1}\n\nAnswer 2: {option2}\n\nAnswer 3: They are equally good."
        usr_prompt = "Please concisely explain your reasoning in comparing the two clinical notes, then select your final answer. \
            Format your response as follows: \n\nExplanation: <your explanation>\n\nAnswer: <your answer>"
        chat = Chat(model_name='gpt-4-turbo')
        try:
            response = chat(sys_prompt, usr_prompt, temperature=0.0) # TEMPERATURE 0?
            answer = response.split('\nAnswer: ')[1]
            explanation = response.split('Explanation: ')[1].split('\n\nAnswer: ')[0]
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

    NOTE: Need to implement the inference and creation of this dataframe in inference.py
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
        save_file(rankings, rankings_path)
    return df

def elo_ranking(df):
    ''' 
    Elo ranking for clinical note evaluation with GPT-4.
    Taken from https://portkey.ai/blog/comparing-llm-outputs-with-elo-ratings/

    Arguments:
        - score_dict (dict of str: List[float]): dictionary of scores for each model
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