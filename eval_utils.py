import jsonlines

none_match  = "None_Match"
none_mismatch = "None_Mismatch"

def get_eval_dict(pred_dict, gold_dict, string_match_score):
    eval_dict = {}
    for (field, gold), (pred_field, pred) in zip(gold_dict.items(), pred_dict.items()):
        if field != pred_field:
            raise ValueError(f"field mismatch: {field} != {pred_field}")
        
        if gold.type != pred.type:
            raise ValueError(f"type mismatch: {gold.type} != {pred.type}")

        if gold.type == list:
            eval_dict[field] = [eval(pred_, gold_, string_match_score) for pred_, gold_ in zip(pred, gold)]
        
        if gold.type == dict:
            eval_dict[field] = eval(pred, gold, string_match_score)
        
        if gold.type == str and pred.type == str:
            if gold == "None":
                if pred == "None":
                    eval_dict[field] = none_match
                else :
                    eval_dict[field] = none_mismatch
            if pred == "None":
                eval_dict[field] = none_mismatch
            else:
                eval_dict[field] = string_match_score(pred, gold)
        
        else :
            raise ValueError(f"unexpected type: {gold.type}")

    return eval_dict
        
