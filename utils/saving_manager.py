import os
import json
import pandas as pd

SUMMARY_EVALUATION_STEPS = {'create eval directory' : 'tbd', 'flatten and match dicts' : 'tbd', 'clean dicts and counts' : 'tbd', 'summary_statistics' : 'tbd', 'eval_by_sample' : 'tbd', 'eval_by_key' : 'tbd'}
NOTE_EVALUATION_STEPS = {'create eval directory' : 'tbd'}

class EvalSaving():
    def __init__(self, steps ,path, save_path = None):
        if save_path is None:
            save_path = path.replace('.jsonl', '_evaluation')
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
            self.progress_dict = steps
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
        self.save_progress_dict()
    
    def load_flatten_and_match_dicts(self):
        return pd.read_json(f"{self.save_path}/flatten_and_match_dicts.jsonl", orient='records', lines=True)

    def clean_dicts_and_counts_update(self, df, delete_flat = True):
        df.to_json(f"{self.save_path}/clean_dicts_and_counts.jsonl", orient='records', lines=True)
        self.progress_dict['clean dicts and counts'] = 'done'
        if delete_flat:
            os.remove(f"{self.save_path}/flatten_and_match_dicts.jsonl")
        self.save_progress_dict()
    
    def load_clean_dicts_and_counts(self):
        return pd.read_json(f"{self.save_path}/clean_dicts_and_counts.jsonl", orient='records', lines=True)

    def save_summary_statistics(self, df, delete_clean = True):
        df.to_json(f"{self.save_path}/summary_statistics.jsonl", orient='records', lines=True)
        self.progress_dict['summary_statistics'] = 'done'
        if delete_clean:
            os.remove(f"{self.save_path}/clean_dicts_and_counts.jsonl")
        self.save_progress_dict()

    def load_summary_statistics(self):
        return pd.read_json(f"{self.save_path}/summary_statistics.jsonl", orient='records', lines=True)
    
    def save_eval_by_sample(self, df, delete_summary_statistics = False):
        df.to_json(f"{self.save_path}/eval_by_sample.jsonl", orient='records', lines=True)
        self.progress_dict['eval_by_sample'] = 'done'
        self.save_progress_dict()

    def load_eval_by_sample(self):
        return pd.read_json(f"{self.save_path}/eval_by_sample.jsonl", orient='records', lines=True)
    
    def save_eval_by_key(self, df, delete_summary_statistics = False):
        df.to_json(f"{self.save_path}/eval_by_key.jsonl", orient='records', lines=True)
        self.progress_dict['eval_by_key'] = 'done'
        if delete_summary_statistics:
            os.remove(f"{self.save_path}/summary_statistics.jsonl")
        self.save_progress_dict()

    def load_eval_by_key(self):
        return pd.read_json(f"{self.save_path}/eval_by_key.jsonl", orient='records', lines=True)