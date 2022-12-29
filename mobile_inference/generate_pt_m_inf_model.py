import sys
sys.path.insert(0,'..')
from models.classification import ClassificationLM
from models.test_model import BertClassifier

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

class PytorchMobileModelGenerator():
    def __init__(self, model=None, model_name='test'):
        self.model_name = model_name
        if model == None:
            if model_name == 'test':
                self.cls_model = BertClassifier()
            else:
                self.cls_model = ClassificationLM(model_name)
        else:
            self.cls_model = model
        self.cls_model.eval()
    
    def get_sample_input(self):
        if self.model_name == 'test':
            test_sentence = "Hello, my dog is cute"
            tokenized = self.cls_model.get_tokenizer().encode_plus(
                test_sentence,
                add_special_tokens=True,
                max_length=512,
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids = tokenized['input_ids'].to('cpu')
            attention_mask = tokenized['attention_mask'].to('cpu')
            return input_ids, attention_mask
        return None

    def generate_mobile_model(self, out_path='./../out/pt_mobile_models/model.pt', manual_input=None):
        sample_input = self.get_sample_input()
        if sample_input == None:
            sample_input = manual_input
        traced_script_module = torch.jit.trace(self.cls_model, sample_input)
        traced_script_module_optimized = optimize_for_mobile(traced_script_module)
        # traced_script_module_optimized._save_for_lite_interpreter(out_path)
        torch.jit.save(traced_script_module_optimized, out_path)

if __name__ == '__main__':
    ptMMGenerator = PytorchMobileModelGenerator(model_name='test')
    ptMMGenerator.generate_mobile_model('./../out/pt_mobile_models/test_model.pt')