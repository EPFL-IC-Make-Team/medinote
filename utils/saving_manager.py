import os
import json
import pandas as pd
import numpy as np

SUMMARY_EVALUATION_STEPS = {'create eval directory' : 'tbd', 'flatten and match dicts' : 'tbd', 'clean dicts and counts' : 'tbd', 'summary_statistics' : 'tbd', 'scores' : 'tbd', 'pairs_idx' : 'tbd' ,'eval_by_sample' : 'tbd', 'eval_by_key' : 'tbd'}
NOTE_EVALUATION_STEPS = {'create eval directory' : 'tbd', 'scores' : 'tbd'}
ELO_RANKING_STEPS = {'create eval directory' : 'tbd', 'build_sub_batch' : 'tbd', 'scores' : 'tbd', 'compute elos' : 'tbd' }

class EvalSaving():
    def __init__(self, steps ,path, save_path = None):
        if save_path is None:
            save_path = path.replace('evaluation_input.jsonl', 'eval_res')
            self.save_path = save_path
        else:
            self.save_path = save_path

        if os.path.exists(save_path):
            print("Evaluation directory exists, loading...")
            self.progress_dict = json.load(open(f"{save_path}/progress"))
            print(f"Progress: {self.progress_dict}")
        else:
            os.makedirs(save_path)
            print("Creating Evaluation directory and progress monitoring")
            self.progress_dict = steps.copy()
            self.progress_dict['create eval directory'] = 'done'
            with open(f"{save_path}/progress", 'w') as f:
                json.dump(self.progress_dict, f)
    
    def get_progress_dict(self):
        return self.progress_dict

    def save_progress_dict(self):
        with open(f"{self.save_path}/progress", 'w') as f:
            json.dump(self.progress_dict, f)

    def flatten_and_match_dicts_update(self, df):
        df.to_json(f"{self.save_path}/flatten_and_match_dicts.jsonl", orient='records', lines=True)
        self.progress_dict['flatten and match dicts'] = 'done'
        self.progress_dict['summary_statistics'] = 'in progress'
        self.save_progress_dict()
    
    def load_flatten_and_match_dicts(self):
        return pd.read_json(f"{self.save_path}/flatten_and_match_dicts.jsonl", orient='records', lines=True)

    def clean_dicts_and_counts_update(self, df, delete_flat = False):
        df.to_json(f"{self.save_path}/clean_dicts_and_counts.jsonl", orient='records', lines=True)
        self.progress_dict['clean dicts and counts'] = 'done'
        if delete_flat:
            os.remove(f"{self.save_path}/flatten_and_match_dicts.jsonl")
        self.save_progress_dict()
    
    def load_clean_dicts_and_counts(self):
        return pd.read_json(f"{self.save_path}/clean_dicts_and_counts.jsonl", orient='records', lines=True)

    def save_summary_statistics(self, df, delete_clean = False):
        df.to_json(f"{self.save_path}/summary_statistics.jsonl", orient='records', lines=True)
        self.progress_dict['summary_statistics'] = 'done'
        if delete_clean:
            os.remove(f"{self.save_path}/clean_dicts_and_counts.jsonl")
        self.save_progress_dict()

    def load_summary_statistics(self):
        return pd.read_json(f"{self.save_path}/summary_statistics.jsonl", orient='records', lines=True).fillna(value=np.nan)
    
    def save_eval_by_sample(self, df, delete_summary_statistics = False):
        df.to_json(f"{self.save_path}/summary_eval_by_sample.jsonl", orient='records', lines=True)
        self.progress_dict['eval_by_sample'] = 'done'
        self.save_progress_dict()

    def load_eval_by_sample(self):
        return pd.read_json(f"{self.save_path}/summary_eval_by_sample.jsonl", orient='records', lines=True).fillna(value=np.nan)
    
    def save_eval_by_key(self, df):
        df.to_json(f"{self.save_path}/summary_eval_by_key.jsonl", orient='records', lines=True)
        self.progress_dict['eval_by_key'] = 'done'
        self.save_progress_dict()

    def load_eval_by_key(self):
        return pd.read_json(f"{self.save_path}/summary_eval_by_key.jsonl", orient='records', lines=True).fillna(value=np.nan)
    
    def save_pairs_idx(self, df):
        df.to_json(f"{self.save_path}/pairs_to_score.jsonl", orient='records', lines=True)
        self.progress_dict['pairs_idx'] = 'done'
        self.save_progress_dict()
    
    def load_pairs_idx(self):
        return pd.read_json(f"{self.save_path}/pairs_to_score.jsonl", orient='records', lines=True)

    def save_one_score(self, batch_df, score_name, done = False):
        with open(f"{self.save_path}/{score_name}.jsonl", 'a') as f:
            f.write(batch_df.to_json(orient='records', lines=True))
        if done:
            self.progress_dict[score_name] = 'done'
        else:
            self.progress_dict[score_name] = 'in progress'
        self.save_progress_dict()
    
    def load_one_score(self, score_name):
        return pd.read_json(f"{self.save_path}/{score_name}.jsonl", orient='records', lines=True)
            
    def save_all_scores(self, df):
        df.to_json(f"{self.save_path}/all_scores.jsonl", orient='records', lines=True)
        self.progress_dict['scores'] = 'done'
        self.save_progress_dict()
    
    def load_all_scores(self):
        return pd.read_json(f"{self.save_path}/all_scores.jsonl", orient='records', lines=True)

    def save_sub_batch(self, df):
        df.to_json(f"{self.save_path}/sub_batch_for_elo.jsonl", orient='records', lines=True)
        self.progress_dict['build_sub_batch'] = 'done'
        self.save_progress_dict()
    
    def load_sub_batch(self):
        return pd.read_json(f"{self.save_path}/sub_batch_for_elo.jsonl", orient='records', lines=True)
    
    def save_elos(self, df):
        df.to_json(f"{self.save_path}/computed_elos.jsonl", orient='records', lines=True)
        self.progress_dict['compute elos'] = 'done'
        self.save_progress_dict()
    
    def load_elos(self):
        return pd.read_json(f"{self.save_path}/computed_elos.jsonl", orient='records', lines=True)