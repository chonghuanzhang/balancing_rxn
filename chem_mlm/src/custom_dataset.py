# Create a customised data collator class to learn from default mask
# since HuggingFace does not have a default mask mlm transformer

# from transformers import DefaultDataCollator
from torch.utils.data import Dataset,DataLoader
class CustomDataset(Dataset):
    def __init__(self,text,label):
        self.text = text
        self.label = label
    def __getitem__(self,idx):
        return {'text':self.text[idx],'label':self.label[idx]}
    def __len__(self):
        return len(self.text)
    
class CustomDataCollatorForLanguageModeling(object):
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer(tokenizer.mask_token)['input_ids'][0]
    def __call__(self,examples):
        
        labels = [example['label'] for example in examples]

        
        texts = [example['text'] for example in examples]
        tokenizer_output = self.tokenizer(texts, truncation=True, padding=True,return_tensors='pt',return_token_type_ids=False,max_length=512)
        

        labels = self.tokenizer(labels, truncation=True, padding=True,return_tensors='pt',return_token_type_ids=False,max_length=512)['input_ids']
        
        delta = tokenizer_output['input_ids'].shape[1] - labels.shape[1]
        if  delta >0:
            deltas = -100 * torch.ones(labels.shape[0],delta)
            deltas = deltas.long()
            labels = torch.hstack([labels,deltas])
            
        
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        output_dict = dict(labels=labels, **tokenizer_output)
        return output_dict