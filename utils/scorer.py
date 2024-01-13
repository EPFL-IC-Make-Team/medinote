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
import re
import numpy as np
from evaluate import load
import sys
import os
import time
from concurrent.futures import ProcessPoolExecutor
import nltk
from nltk.corpus import stopwords


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from utils.chat import *
from utils.eval import *
from utils.infer import *

BERT_SCORER = load("bertscore")
ROUGE_SCORER = load("rouge")
BLEU_SCORER = load("bleu")

NONE_MATCH  = "None_Match"
NO_SUCH_KEY = "no_such_key"
NONE_FIELD = "None"
ALL_SCORE_TYPES = ['bleu', 'rouge', 'bert', 'gpt_rank', 'gpt_score']
ROUGE_SUB_SCORES = ['rouge1', 'rouge2',	'rougeL', 'rougeLsum']
COUNTS_TYPES = ['missing_keys_count', 'extra_keys_count', 'common_none_count',  'gold_none_count', 'pred_none_count', 'common', 'total']
KEY_MISMATCH_TYPE = ['gold_none_keys', 'pred_none_keys', 'missing_keys']

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = nltk.word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

def remove_stopwords_from_pair(pair):
    return {'gold': remove_stopwords(pair['gold']), 'pred': remove_stopwords(pair['pred'])}

# ----------------------- Scoring functions (BLEU/ROUGE/BERT/GPT-4) ----------------------- #

class Scorer(): 
    def __init__(self, saving_manager, score_types='all', cot=False):
        ''' 
        Initializes a scorer with a list of scoring modes to be used.
        Argument: 
            - score_types (str or list): list of scoring functions. Default: 'all' (all scoring functions)
            - cot (bool): whether to use Chain-of-Thought or not for GPT-4 evaluation (default: True)
        '''
        self.saving_manager = saving_manager
        self.cot = cot
        self.score_types = ALL_SCORE_TYPES if score_types == 'all' else score_types
        for score_type in self.score_types:
            saving_manager.get_progress_dict()[score_type] = self.saving_manager.get_progress_dict().get(score_type, 'tbd')
        
        #print('Initialized scorer with modes: ', list(self.score_types))

    def __call__(self, pairs_df, remove_stopwords = False): 
        '''
        Given a pandas dataframe with gold and predicted pairs and idxs, 
        returns a dataframe with the different computed scores.

        Example usage: 
            pairs = [{'gold': 'x', 'pred': 'y'}]
            scorer = Scorer(['bleu', 'bert'])
            scores = scorer(pairs)
            --> scores = dataframe(
                {
                'pairs':[{'gold': 'x', 'pred': 'y'}],
                'bleu': [0.5],  
                'bert': [0.7]
                    })
        '''

        if remove_stopwords and ('bleu' in self.score_types or 'rouge' in self.score_types):
            print("Removing stopwords...")
            tqdm.pandas()
            pairs_df['pairs_prep'] = pairs_df['pairs'].progress_apply(lambda x: remove_stopwords_from_pair(x))
            print("Stopwords removed.")
        else:
            pairs_df['pairs_prep'] = pairs_df['pairs']

        if 'bleu' in self.score_types:
            pairs_df['bleu'] = self.BLEU_scorer(pairs_df['pairs_prep'])['bleu']

        if 'rouge' in self.score_types:
            rouges = self.ROUGE_scorer(pairs_df[['idxs','pairs_prep']])
            for metric in ROUGE_SUB_SCORES: #different rouge scores
                pairs_df[metric] = rouges[metric]
        if 'bert' in self.score_types:
            pairs_df['bert'] = self.BERT_scorer(pairs_df[['idxs','pairs']])

        if 'gpt_rank' in self.score_types:
            pairs_df['gpt_rank'] = self.GPT_ranker(pairs_df[['idxs','pairs']])
        if 'gpt_score' in self.score_types:
            pairs_df['gpt_score'] = self.GPT_scorer(pairs_df[['idxs','pairs']])

        if remove_stopwords and ('bleu' in self.score_types or 'rouge' in self.score_types):
            pairs_df.drop(columns=['pairs_prep'], inplace=True)
        self.saving_manager.save_all_scores(pairs_df)

        return pairs_df
    
    def BLEU_scorer(self, pairs):
        ''' BLEU score for summary evaluation (precision-oriented)'''
        status = self.saving_manager.get_progress_dict()['bleu']
        if status == 'done':
                print("BLEU scores already computed.")
                return self.saving_manager.load_one_score('bleu')
        
        else :
            if status == 'in progress':
                print("Bleu score computation already in progress, resuming")
                pairs_to_compute = self.saving_manager.get_one_score_to_compute('bleu', pairs)
            else :
                pairs_to_compute = pairs

            bleu_scores = {'bleu': [BLEU_SCORER.compute(
                predictions=[pair['pred']],
                references=[pair['gold']])['bleu'] if len(pair['gold']) > 0 and len(pair['pred']) > 0 else 0
                for pair in tqdm(pairs_to_compute, total = len(pairs_to_compute) ,desc="Computing BLEU scores")]}
            
            self.saving_manager.save_one_score(pd.DataFrame(bleu_scores),'bleu',done = True)
            
            return bleu_scores
    
    def ROUGE_scorer(self, pairs):
        ''' ROUGE score for summary evaluation (recall-oriented)'''''

        status = self.saving_manager.get_progress_dict()['rouge']
        if status == 'done':
                print("ROUGE scores already computed.")
                return self.saving_manager.load_one_score('rouge')
        
        else :
            if status == 'in progress':
                print("ROUGE score computation already in progress, resuming")
                computed = self.saving_manager.load_one_score('rouge')
                pairs_to_compute = pairs[~pairs['idxs'].isin(computed['idxs'].tolist())]
                res = pairs[pairs['idxs'].isin(computed['idxs'].tolist())]
                res[ROUGE_SUB_SCORES] = computed[ROUGE_SUB_SCORES]
            else :
                pairs_to_compute = pairs
                res = pd.DataFrame()

            batch_size = 100
            batches = [pairs_to_compute.iloc[i:i + batch_size].copy() for i in range(0, len(pairs_to_compute), batch_size)]
            for i, batch in tqdm(enumerate(batches), total = len(batches) ,desc="Computing Rouge scores"):
                rouges = [ROUGE_SCORER.compute(
                    predictions=[pair['pred']], references=[pair['gold']]) if len(pair['gold']) > 0 and len(pair['pred']) > 0 else {metric: 0 for metric in ROUGE_SUB_SCORES}
                    for pair in batch['pairs_prep']]
            
                for metric in ROUGE_SUB_SCORES:
                    batch[metric] = [rouge[metric] for rouge in rouges]

                self.saving_manager.save_one_score(batch[['idxs']+ ROUGE_SUB_SCORES],'rouge', done = (i == (len(batches) - 1)))
                
                res = pd.concat([res, batch[['idxs'] + ROUGE_SUB_SCORES]], ignore_index=True)

            self.saving_manager.save_one_score(res,'rouge', done = True)

            return res[ROUGE_SUB_SCORES]

    def BERT_scorer(self, pairs):
        '''BERT score for summary evaluation'''
        print("Computing BERTscores...")
        status = self.saving_manager.get_progress_dict()['bert']
        if status == 'done':
            print("BERTscores already computed.")
            return self.saving_manager.load_one_score('bert')['bert']
        else :
            if status == 'in progress':
                print("BERTscores computation already in progress, resuming")
                computed = self.saving_manager.load_one_score('bert')
                pairs_to_compute = pairs[~pairs['idxs'].isin(computed['idxs'].tolist())]
                res = pairs[pairs['idxs'].isin(computed['idxs'].tolist())]
                res['bert'] = computed['bert']

            else :
                pairs_to_compute = pairs
                res = pd.DataFrame()
            
            #display(pairs_to_compute)

            #batches of 100 pairs
            batch_size = 100    
            batches = [pairs_to_compute.iloc[i:i + batch_size].copy() for i in range(0, len(pairs_to_compute), batch_size)]
           
            for i, batch in tqdm(enumerate(batches), total = len(batches) ,desc="Computing BERT scores"):
                bert_scores = {'bert': 
                    BERT_SCORER.compute(
                        predictions=[pair['pred'] for pair in batch['pairs']],
                        references=[pair['gold'] for pair in batch['pairs']],
                        model_type= 'distilbert-base-uncased',
                        verbose = False)['f1']
                    }
                
                batch['bert'] = bert_scores['bert']
                self.saving_manager.save_one_score(batch[['idxs', 'bert']], 'bert', done = (i == (len(batches) - 1)))
                res = pd.concat([res, batch[['idxs', 'bert']]], ignore_index=True)
            
            return res['bert']
    
    def ranker_formatting(self, answer):
        """
        Format the ranking answer from GPT-4
        to get the winners and (explanations if cot) as list of strings
        """
        winner_pattern = r"'higher_quality_note': '([^']+)'"      
        winners = re.findall(winner_pattern, answer)

        pair_numbers = [int(x) for x in re.findall(r"'pair_number': (\d+)", answer)]
        max_pair_number = max(pair_numbers) if pair_numbers else None
        if len(winners) != max_pair_number + 1:
            raise ValueError(f"Invalid format:\n{answer}.")
        winners = [x if x in ['NoteA', 'NoteB', 'tie'] else None for x in winners]
        
        if self.cot:
            explanation_pattern = r"'explanation': '([^']+)'"
            explanations = re.findall(explanation_pattern, answer)
            if len(explanations) != max_pair_number + 1:
                raise ValueError(f"Invalid format:\n{answer}.")
            
            return winners, explanations
        
        else:
            return winners

    def scorer_formatting(self, answer):
        """
        Format the scoring answer from GPT-4 
        to get the similarity scores (and explanaion if cot) as list of int/string
        """
        similarity_score_pattern = r"'similarity_score':\s*(\d+)"
        similarity_scores = re.findall(similarity_score_pattern, answer)
        int_answers = [int(score) for score in similarity_scores]
        
        pair_numbers = [int(x) for x in re.findall(r"'pair_number': (\d+)", answer)]
        max_pair_number = max(pair_numbers) if pair_numbers else None
        
        if len(int_answers) != max_pair_number + 1:
            raise ValueError(f"Invalid format {answer}.")
        
        int_answers = [x if x in range(1,11) else None for x in int_answers]
        
        if self.cot:
            explanation_pattern = r"'explanation':'([^']+)'"
            explanations = re.findall(explanation_pattern, answer)
            if len(explanations) != max_pair_number + 1:
                raise ValueError(f"Invalid format {answer}.")
            
            return int_answers, explanations
        
        else:
            return int_answers

    def GPT_ranker(self, 
                   pairs_df,
                   model_name='gpt-4-1106-preview',
                   chat=chat_gpt_4_turbo,
                   max_tokens = 300,
                   one_call_batch_size = 3, 
                   temperature=0.0):
        ''' 
        For each pair of gold and pred strings, ask GPT-4 to pick which one is the best.
        NOTE: we randomly mix answer order to avoid bias.

        Arguments:
            - pairs (list of dict {'gold': str, 'pred': str}): list of gold and pred strings
            - model_name (str): name of the GPT-4 model to use for tokan counts (default: gpt-4-1106-preview)
            - chat (function): chat function to use for requests (default:  gpt-4-1106-preview)
            - one_call_batch_size (int): numer of pairs to evaluate in one call to GPT-4 (default: 20) 
                    (nothing related to parallelization, it's really one call)
            - temperature (float): temperature for GPT-4 sampling (default: 0.0)
        Returns: 
            - Pandas series of the winners (in the same order as the pairs)

        TODO: Save the explanations and scores in a separate file as backup, possibility of resume
        '''
        print("GPT-4 ranking...")

        res = pd.DataFrame({'idxs': [], 'similarity': [], 'explanations': []})
        if self.saving_manager.get_progress_dict()['gpt_rank'] == 'done':
            print("GPT-4 ranks already computed.")
            return self.saving_manager.load_one_score('gpt_rank')['winner']
        else:
            if self.saving_manager.get_progress_dict()['gpt_rank'] == 'in progress':
                print("GPT-4 score computation already in progress, resuming")
                computed = self.saving_manager.load_one_score('gpt_rank')
                pairs_to_compute = pairs_df[~pairs_df['idxs'].isin(computed['idxs'].tolist())]
                res = pd.concat([res, computed], ignore_index=True)
            else:
                pairs_to_compute = pairs_df

        #Builds the batch of pairs to evaluate as one single call
        dataset = pairs_to_compute.groupby(pairs_to_compute['idxs'] // one_call_batch_size).agg(lambda x: x.tolist())
        dataset['switches'] = None
        messages = []
        idxs_grouped = []

        for i, pair_list in tqdm(dataset.iterrows(), total=dataset.shape[0], desc="Building ranking prompts"): 
            idxs = pair_list['idxs']
            noteAs = [pair['noteA'] for pair in pair_list['pairs']]
            noteBs = [pair['noteB'] for pair in pair_list['pairs']]
            switches = [np.random.choice([0, 1]) for _ in range(len(noteAs))]
            #We randomly mix gold and pred to avoid bias
            optionsA = [gold if switch == 0 else pred for gold, pred, switch in zip(noteAs, noteBs, switches)]
            optionsB = [gold if switch == 1 else pred for gold, pred, switch in zip(noteAs, noteBs, switches)]
            
            sys_prompt = f"For each pair of clinical notes (pair i, NoteA, NoteB) you are given, compare them and rank which one is of higher quality. "
            if self.cot:
                sys_prompt += "Explain in one sentence your reasoning in comparing the two clinical notes, then select your final answer.\n \
                    Format your response as follows: a list of dictionnaries [{'pair_number': i, 'explanation': <your explanation>, 'higher_quality_note': <your answer>}].\n"
            else:
                sys_prompt += "Directly respond with your final answers as follows: a list of dictionnaries [{'pair_number': i, 'higher_quality_note': <your answer>}\n"
            
            sys_prompt += "Your answer should be 'NoteA' if NoteA has better quality, 'NoteB' if NoteB has better quality, or 'tie' if they have the same quality."
            usr_prompt = '\n'.join([f"(pair {j}, {optionA}, {optionB})" for j, (optionA, optionB) in enumerate(zip(optionsA, optionsB))])
            idxs_grouped.append(idxs)
            messages.append(build_messages(sys_prompt, usr_prompt))
            dataset.at[i, 'switches'] = switches

        # Builds batch of calls so that each call has a total number of tokens less than max_tokens
        sub_batches = partition(dataframe = pd.concat([pd.DataFrame({'idxs': idxs_grouped, 'messages': messages}),dataset['switches']], axis =1),
                                max_token_per_partition=max_tokens,
                                model = model_name)
        
        # Generate answers by batches
        for i, (sub_batch, nb_tokens) in enumerate(sub_batches):
            print(f"Sub_batch {i+1}/{len(sub_batches)}: {sub_batch.shape[0]} calls, {nb_tokens} total tokens: {nb_tokens/1000 * 0.01}$")
            if os.path.exists("safety_save.pkl"):
                delete_pickle_file("safety_save.pkl") #in-batch resume not possible due to swtiches
            
            start_time = time.time()
            answers = generate_answers(
                messages_list = sub_batch['messages'].tolist(),
                formatting= self.ranker_formatting,
                chat=chat,
                temperature=temperature
            ) # answer is list of list

            if self.cot:
                explanations = [answer[1] for answer in answers]
                winners = [['tie' if (answer == 'tie') else
                    'gold' if (answer == 'NoteA' and switch == 0) or (answer == 'NoteB' and switch == 1) else
                    'pred' 
                    for answer,switch in zip(answer_list[0], switches)]
                    for answer_list, switches in zip(answers, sub_batch['switches'].tolist())]

            else:
                #print(sub_batch['switches'].tolist())
                #print(answers)
                #print(zip(answers, sub_batch['switches'].tolist()))
                explanations =  [[None] * len(answer) for answer in answers]
                winners = [['tie' if (answer == 'tie') else
                        'NoteA' if (answer == 'NoteA' and switch == 0) or (answer == 'NoteB' and switch == 1) else
                        'NoteB' 
                        for answer,switch in zip(answer_list, switches)]
                        for answer_list, switches in zip(answers, sub_batch['switches'].tolist())]
            sub_batch_res = pd.DataFrame({'idxs': sub_batch['idxs'],
                                            'winner': winners, 
                                            'explanation': explanations}).explode(['idxs','winner','explanation'])
            res = pd.concat([res, sub_batch_res], ignore_index=True)

            self.saving_manager.save_one_score(sub_batch_res, 'gpt_rank', done = False)
            delete_pickle_file("safety_save.pkl")
            
            end_time = time.time()
            time_taken = (end_time - start_time)
            breaktime = max(int(20 - time_taken) + 2, 5) #time we wait before calling the api again
            print(f"\nBreak for {breaktime} seconds.")
            time.sleep(breaktime)
            print("End of break.")

        self.saving_manager.save_one_score(res, 'gpt_rank', done = True)
        return res['winner']

    def GPT_scorer(self, 
                   pairs_df,
                   model_name='gpt-4-1106-preview',
                   chat=chat_gpt_4_turbo,
                   temperature=0.0,
                   max_tokens = 500000,
                   one_call_batch_size=15):
        ''' 
        Given a model’s answer and GPT-4' answer (silver label), 
        ask GPT-4 with CoT to compute the similarity between the two answers on a scale of 1 to 10. 
        The higher the number the closer the model’s quality to GPT-4's. 
        NOTE: we randomly mix gold and pred to avoid bias.

        Arguments:
            - pairs (list of dict {'gold': str, 'pred': str}): list of gold and pred strings
            - model_name (str): name of the GPT-4 model to use for tokan counts (default: gpt-4-1106-preview)
            - chat (function): chat function to use for requests (default:  gpt-4-1106-preview)
            - one_call_batch_size (int): numer of pairs to evaluate in one call to GPT-4 (default: 20) 
                    (nothing related to parallelization, it's really one call)
            - temperature (float): temperature for GPT-4 sampling (default: 0.0)
        Returns: 
            - pandas series of similarity scores (in the same order as the pairs given)

        TODO: Save the explanations and scores in a separate file.
        '''
        print("GPT-4 scoring...")
        
        res = pd.DataFrame({'idxs': [], 'similarity': [], 'explanations': []})
        if self.saving_manager.get_progress_dict()['gpt_score'] == 'done':
            print("GPT-4 scores already computed.")
            return self.saving_manager.load_one_score('gpt_score')['similarity']
        else:
            if self.saving_manager.get_progress_dict()['gpt_score'] == 'in progress':
                print("GPT-4 score computation already in progress, resuming")
                computed = self.saving_manager.load_one_score('gpt_score')
                pairs_to_compute = pairs_df[~pairs_df['idxs'].isin(computed['idxs'].tolist())]
                res = pd.concat([res, computed], ignore_index=True)
            else:
                pairs_to_compute = pairs_df

        dataset = pairs_to_compute.groupby(pairs_to_compute['idxs'] // one_call_batch_size).agg(lambda x: x.tolist())
        # Build prompts for GPT-4 from gold/pred pairs
        messages = []
        idxs_grouped = []
        for _, pair_list in tqdm(dataset.iterrows(), total=dataset.shape[0], desc="Building score prompts"): 
            idxs = pair_list['idxs']
            golds = [pair['gold'] for pair in pair_list['pairs']]
            preds = [pair['pred'] for pair in pair_list['pairs']]
            #We randomly mix gold and pred to avoid bias
            switches = [np.random.choice([0, 1]) for _ in range(len(golds))]
            optionsA = [gold if switch == 0 else pred for gold, pred, switch in zip(golds, preds, switches)]
            optionsB = [gold if switch == 1 else pred for gold, pred, switch in zip(golds, preds, switches)]
            sys_prompt = f"For each pair of notes (pair i, NoteA, NoteB) you are given, rate how similar they are from 1 to 10. Focus on content and pay attention to details. Be very strict with any content differences"
            
            if self.cot:
                sys_prompt += "Explain in one sentence your reasoning in comparing the two notes, then select your final answer.\n \
                    Format your response as follows: a list of dictionnaries [{'pair_number': i, 'explanation': <your explanation>, 'similarity_score': <your answer>}]"
            else:
                sys_prompt += "Directly respond with your final answers as follows: a list of dictionnaries [{'pair_number': i, 'similarity_score': <your answer>}]"
            
            usr_prompt = '\n'.join([f"(pair {i}, {optionA}, {optionB})" for i, (optionA, optionB) in enumerate(zip(optionsA, optionsB))])
            idxs_grouped.append(idxs)
            messages.append(build_messages(sys_prompt, usr_prompt))
        
        

        

        sub_batches = partition(dataframe = pd.DataFrame({'idxs': idxs_grouped, 'messages': messages}), max_token_per_partition=max_tokens,model = model_name)

        total_tokens = np.sum([nb_tok for _, nb_tok in sub_batches])
        print(f"Total input tokens: {total_tokens}, total input cost: {total_tokens/1000 * 0.01}$")

        # Generate answers by batches
        
        for i, (sub_batch, nb_tokens) in enumerate(sub_batches):
            print(f"Sub_batch {i+1}/{len(sub_batches)}: {sub_batch.shape[0]} calls, {nb_tokens} total tokens: {nb_tokens/1000 * 0.01}$")
            
            start_time = time.time()
            answers = generate_answers(
                messages_list = sub_batch['messages'].tolist(),
                formatting=self.scorer_formatting,
                chat=chat,
                temperature=temperature
            ) # answer is list of list

            

            explanations =  [[None]* len(answer) if ~self.cot else answer[1] for answer in answers]
            similarities = answers if not self.cot else [answer[0] for answer in answers]
            sub_batch_res = pd.DataFrame({'idxs': sub_batch['idxs'],
                                            'similarity': similarities, 
                                            'explanations': explanations}).explode(['idxs','similarity','explanations'])
            
            res = pd.concat([res, sub_batch_res], ignore_index=True)

            self.saving_manager.save_one_score(sub_batch_res, 'gpt_score', done = (i == len(sub_batches) - 1))
            delete_pickle_file("safety_save.pkl")
            
            end_time = time.time()
            time_taken = (end_time - start_time)
            breaktime = max(int(20 - time_taken) + 2, 5) #time we wait before calling the api again
            print(f"\nBreak for {breaktime} seconds.")
            time.sleep(breaktime)
            print("End of break.")
            

        self.saving_manager.save_one_score(res, 'gpt_score', done = True)
        return res['similarity']
