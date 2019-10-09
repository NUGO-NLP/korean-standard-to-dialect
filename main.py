import json
import os

from model.rule_base import ruleBase

current_path = os.path.dirname(os.path.abspath( __file__ ))
sent_gs_valid_filename = os.path.join(current_path, 'data/sent_gs_valid.json')
sent_jl_valid_filename = os.path.join(current_path, 'data/sent_jl_valid.json')
result_gs_filename = os.path.join(current_path, 'output/sent_result_gs.json')
result_jl_filename = os.path.join(current_path, 'output/sent_result_jl.json')

rule_base_model_gs = ruleBase('gs')
rule_base_model_jl = ruleBase('jl')

def test_and_write(model, input_filename, output_filename):
    with open(input_filename) as input_file:
        sentence_valid = json.load(input_file)
    with open(output_filename, 'w', encoding='UTF-8') as output_file:
        for sentence in sentence_valid:
            standard = sentence['standard']
            inference = model.inference_sentence(standard)
            sentence['inference'] = inference
        output_file.write(json.dumps(sentence_valid, ensure_ascii=False, indent='\t'))

if __name__ == "__main__":
    
    sent_gs_valid = dict()
    sent_jl_valid = dict()

    test_and_write(rule_base_model_gs, sent_gs_valid_filename, result_gs_filename)
    test_and_write(rule_base_model_jl, sent_jl_valid_filename, result_jl_filename)
