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
from inference import *

import re
import numpy as np
from evaluate import load

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
        self.score_types = ALL_SCORE_TYPES if score_types == 'all' else score_types
        #print('Initialized scorer with modes: ', list(self.score_types))

    def __call__(self, pairs): 
        '''
        Given a list of dictionaries with gold and predicted pairs, 
        returns a dataframe with the different computed scores.

        Example usage: 
            pairs = [{'gold': 'x', 'pred': 'y'}]
            scorer = Scorer(['bleu', 'rouge', 'bert'])
            scores = scorer(pairs)
            --> scores = dataframe(
                {
                'pairs':[{'gold': 'x', 'pred': 'y'}],
                'bleu': [0.5], 
                'rouge': [0.3], 
                'bert': [0.7]
                    })
        '''
        pairs_df = pd.DataFrame(pairs)
        if 'bleu' in self.score_types:
            pairs_df['bleu'] = self.BLEU_scorer(pairs)['bleu']
        if 'rouge' in self.score_types:
            rouges = self.ROUGE_scorer(pairs)
            for metric in rouges.keys(): #different rouge scores
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
        '''BERT score for summary evaluation'''
        print("Computing BERT scores...")
        scores = {'bert': 
            BERT_SCORER.compute(
                predictions=[pair['pred'] for pair in pairs],
                references=[pair['gold'] for pair in pairs],
                lang='en')['f1']
        }
        print('BERTscores computed.')
        return scores
    
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
                   pairs,
                   model_name='gpt-4-1106-preview',
                   chat=chat_gpt_4_turbo,
                   max_tokens = 300,
                   one_call_batch_size = 5, 
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

        #Builds the batch of pairs to evaluate as one single call
        dataset = pd.DataFrame({'pairs_list': [pairs[i: i+one_call_batch_size] for i in range(0, len(pairs), one_call_batch_size)]})

        dataset[['winner', 'explanation', 'switch', 'model_name']] = [None, None, None, model_name]

        messages = []
        for i, pair_list in tqdm(dataset.iterrows(), total=dataset.shape[0], desc="Building ranking prompts"): 
            golds = [pair['gold'] for pair in pair_list['pairs_list']]
            preds = [pair['pred'] for pair in pair_list['pairs_list']]
            switches = [np.random.choice([0, 1]) for _ in range(len(golds))]
            #We randomly mix gold and pred to avoid bias
            optionsA = [gold if switch == 0 else pred for gold, pred, switch in zip(golds, preds, switches)]
            optionsB = [gold if switch == 1 else pred for gold, pred, switch in zip(golds, preds, switches)]
            
            sys_prompt = f"For each pair of clinical notes (pair i, NoteA, NoteB) you are given, compare them and rank which one is of higher quality. "
            if self.cot:
                sys_prompt += "Explain in one sentence your reasoning in comparing the two clinical notes, then select your final answer.\n \
                    Format your response as follows: a list of dictionnaries [{'pair_number': i, 'explanation': <your explanation>, 'higher_quality_note': <your answer>}].\n"
            else:
                sys_prompt += "Directly respond with your final answers as follows: a list of dictionnaries [{'pair_number': i, 'higher_quality_note': <your answer>}\n"
            
            sys_prompt += "Your answer should be 'NoteA' if NoteA has better quality, 'NoteB' if NoteB has better quality, or 'tie' if they have the same quality."
            usr_prompt = '\n'.join([f"(pair {i}, {optionA}, {optionB})" for i, (optionA, optionB) in enumerate(zip(optionsA, optionsB))])
            
            messages.append(build_messages(sys_prompt, usr_prompt))
            dataset.at[i,'switch'] = switches

        # Builds batch of calls so that each call has a total number of tokens less than max_tokens
        sub_batches = partition(dataframe = pd.concat([pd.DataFrame({'messages': messages}),dataset['switch']], axis =1), max_token_per_partition=max_tokens,model = model_name)

        # Generate answers by batches
        for i, (sub_batch, nb_tokens) in enumerate(sub_batches):
            print(f"Sub_batch {i+1}/{len(sub_batches)}: {sub_batch.shape[0]} calls, {nb_tokens} total tokens: {nb_tokens/1000 * 0.01}$")
            try:
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
                        for answer_list, switches in zip(answers, sub_batch['switch'].tolist())]

                else:
                    explanations =  [None] * sub_batch.shape[0]
                    winners = [['tie' if (answer == 'tie') else
                            'gold' if (answer == 'NoteA' and switch == 0) or (answer == 'NoteB' and switch == 1) else
                            'pred' 
                            for answer,switch in zip(answer_list, switches)]
                            for answer_list, switches in zip(answers, sub_batch['switch'].tolist())]
                dataset.loc[sub_batch.index,'winner'] = pd.Series(winners, index = sub_batch.index)
                dataset.loc[sub_batch.index,'explanation'] = pd.Series(explanations, index = sub_batch.index)
            except Exception as e:
                print(e)
        return dataset['winner'].explode().to_list()

    def GPT_scorer(self, 
                   pairs,
                   model_name='gpt-4-1106-preview',
                   chat=chat_gpt_4_turbo,
                   temperature=0.0,
                   max_tokens = 300,
                   one_call_batch_size=5):
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
            sys_prompt = f"For each pair of notes (pair i, NoteA, NoteB) you are given, compare them and rate how similar they are to each other on a scale of 1 to 10. "
            
            if self.cot:
                sys_prompt += "Explain in one sentence your reasoning in comparing the two notes, then select your final answer.\n \
                    Format your response as follows: a list of dictionnaries [{'pair_number': i, 'explanation': <your explanation>, 'similarity_score': <your answer>}]"
            else:
                sys_prompt += "Directly respond with your final answers as follows: a list of dictionnaries [{'pair_number': i, 'similarity_score': <your answer>}]"
            
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
                    chat=chat,
                    temperature=temperature
                ) # answer is list of list
                explanations =  [None] * sub_batch.shape[0] if not self.cot else [answer[1] for answer in answers]
                similarities = answers if not self.cot else [answer[0] for answer in answers]
                dataset.loc[sub_batch.index,'similarity'] = pd.Series(similarities, index = sub_batch.index)
                dataset.loc[sub_batch.index,'explanation'] = pd.Series(explanations, index = sub_batch.index)
            except Exception as e:
                print(e)
        return dataset['similarity'].explode().to_list() #concateanate the similarities list of list into one list
