import pickle
import pandas as pd
import numpy as np
from collections import OrderedDict
import os

from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import LineByLineTextDataset
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from pathlib import Path
from transformers import RobertaForMaskedLM
from transformers import pipeline

from tokenizers import ByteLevelBPETokenizer ## Use the general BERT tokenization method 

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing 
from transformers import AutoModelWithLMHead
from tqdm import tqdm
# from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

from collections import Counter
# from rdkit import Chem

import torch
from src.custom_dataset import CustomDataset, CustomDataCollatorForLanguageModeling


# def smi_tokenizer(smi):
#     """
#     Tokenize a SMILES molecule or reaction
#     """
#     import re
#     pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
#     regex = re.compile(pattern)
    
#     tokens = [token for token in regex.findall(smi)]
#     return tokens
#     #assert smi == ''.join(tokens)
#     #return ' '.join(tokens)


def mlm_tokenizer(fname, 
              tokenizer_file,
              pretrain_tokenizer_file,
              vocab_size=52000, 
              min_frequency=1,
              ):

    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(files=fname, 
                    vocab_size=vocab_size, 
                    min_frequency=min_frequency
                    )

    # tokenizer._tokenizer.post_processor = BertProcessing(
    #     ("</s>", tokenizer.token_to_id("</s>")),
    #     ("<s>", tokenizer.token_to_id("<s>")),
    # )

    tokenizer.enable_padding()
    tokenizer.enable_truncation(max_length=512) 
    # manually add special tokens for the following encoding steps
    tokenizer.add_special_tokens(['<begin>','<end>','<unk>','<pad>','<cls>','<mask>'])
    tokenizer.save(tokenizer_file)

    # Convert the tokenizer into a pretrained tokenizer以指定特殊字符
    # 需要将自定义tokenizer转化为pretraintokenizer才能正常进行文件的划分和读取
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, \
                                        model_max_length=512,bos_token='<begin>',eos_token='<end>', \
                                        unk_token='<unk>',pad_token='<pad>',cls_token='<cls>',mask_token='<mask>')
   
    # 需要注意的是，我们自定义的tokenizer可以通过 tokenizer = RobertaTokenizerFast.from_pretrained转化为 
    # huggingface中特殊的 pretrain tokenizer，需要再进行一次保存，保存为带有json文件的tokenizer 文件夹，
    # 才能和pretrain的model一起用于后续其它的应用,如下，最终存储了多个配置文件
    tokenizer.save_pretrained(pretrain_tokenizer_file)
    return tokenizer


def get_tokenizer(pretrain_tokenizer_file):
    # From saved tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrain_tokenizer_file)
    return tokenizer


def convert_mask_style(masked_data_file, tokenizer):
    # Convert "@@@" mask of molecule into "<mask>" mask of tokens
    # for example:
    # before: 'Cc1ccc2c(Cl)nccc2c1[N+](=O)[O-].Nc1cccc(C(F)(F)F)c1>>@@@.Cl'
    # after: 'Cc1ccc2c(Cl)nccc2c1[N+](=O)[O-].Nc1cccc(C(F)(F)F)c1>><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask>.Cl'

    data = pd.read_csv(masked_data_file,index_col=0)
    outputs = data.output.tolist()
    lens = []
    for item in tqdm(outputs):
        lens.append(len(tokenizer(item,add_special_tokens=False)['input_ids']))
    inputs = []
    for i,item in enumerate(data.input.tolist()):
        inputs.append(item.replace('@@@',tokenizer.mask_token*lens[i]))

    data['input'] = inputs

    # filter reactions with token indcies sequence longer than the specified maximum sequence lenght -- 512
    data['lens'] = lens
    data = data[data['lens']<=512]

    data.to_csv(masked_data_file)
    # inputs = data.input.tolist()
    # label = data.output.tolist()


# split train validation test datasets
def split_trn_evl_tst(dataset, test_size=0.1, seed=42):
    def _split_train_test(dataset, test_size=test_size, seed=seed):
        len_dtset = len(dataset)
        len_trn_dtset = round((1-test_size)*len_dtset)
        return torch.utils.data.random_split(dataset, [len_trn_dtset, len_dtset-len_trn_dtset], generator=torch.Generator().manual_seed(42))
    dataset_trn, dataset_tst = _split_train_test(dataset, seed=seed)
    dataset_evl, dataset_tst = _split_train_test(dataset_tst, test_size=0.5, seed=seed)
    return dataset_trn, dataset_evl, dataset_tst


def get_dataset(mask_method, 
                masked_data_file, 
                data_file, 
                tokenizer,
                test_size=0.1,
                seed=42
                ):
    if mask_method == 'default':

        # Read masked token data
        data = pd.read_csv(masked_data_file,index_col=0)
        dataset = CustomDataset(data.input.tolist(),data.output.tolist())
        del data

    elif mask_method == 'random':
        dataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=data_file,
            block_size=256,
        )

    dataset_trn, dataset_evl, dataset_test = split_trn_evl_tst(dataset, 
                                                               test_size=test_size, 
                                                               seed=seed
                                                               )
    
    return dataset_trn, dataset_evl, dataset_test


def get_complete_data(mask_method, masked_data_file, data_file, tokenizer, mode=None):
    # Colab would have memory error load the complete data file.
    # Use the func below to avoid error get complete dataset and
    TRN, EVL, TST = [], [], []
    DIR = './data/complete_sentences/'
    for file in os.listdir(DIR):
        if file.startswith('complete_sentences_000'):
            data_file = os.path.join(DIR, file)
            print(data_file)
            dataset_trn, dataset_evl, dataset_tst = get_dataset(
                mask_method, masked_data_file, data_file, tokenizer,test_size=0.01)
            TRN.append(dataset_trn)
            EVL.append(dataset_evl)
            TST.append(dataset_tst)

    TRN = torch.utils.data.ConcatDataset(TRN)
    EVL = torch.utils.data.ConcatDataset(EVL)
    TST = torch.utils.data.ConcatDataset(TST)

    # Save files
    # pickle.dump(TRN, open(os.path.join('./data/complete_sentences/', 'dataset_trn.pickle'),"wb"))
    # pickle.dump(EVL, open(os.path.join('./data/complete_sentences/', 'dataset_evl.pickle'),"wb"))
    # pickle.dump(TST, open(os.path.join('./data/complete_sentences/', 'dataset_tst.pickle'),"wb"))
    if not mode:
        return TRN, EVL, TST
    elif mode == 'train':
        return TRN, EVL
    elif mode == 'test':
        return TST


def possible_missing_mols(rxn_smi, fill_mask, mask_side='right'):
    def _predict(smi):
        result = fill_mask(smi)
        if type(result[0])==dict:
            missing_mol = result[0]['token_str']
        else:
            missing_mol = [item for item in result]
            missing_mol = ''.join([item[0]['token_str'] for item in missing_mol])
        return missing_mol

    print(rxn_smi)

    if mask_side == 'right':
        masked_smi = rxn_smi + '.@@@'
    else:
        masked_smi = '@@@.' + rxn_smi

    possible_masks = [masked_smi.replace('@@@','<mask>'*i) for i in range(1,333)]
    possible_mols = []
    for smi in possible_masks:
        possible_mols.append(_predict(smi))

    return possible_mols


def _remove_duplicate(smi):
    rcts, prds = smi.split('>>')
    rcts = rcts.split('.')
    prds = prds.split('.')
    
    rcts = list(OrderedDict.fromkeys(rcts))
    prds = list(OrderedDict.fromkeys(prds))

    rcts = '.'.join(rcts)
    prds = '.'.join(prds)

    return '>>'.join([rcts, prds])


def pred_rhs_result(input_data_file, pred_data_file, fill_mask, data_count=50, seed=42, smi_his=None):
    # Predict rhs missing molecules from raw ChemBalancer results

    data = pd.read_csv(input_data_file,index_col=0)
    imbalanced_rhs=data.loc[data.msg=='RHS species insufficient']
    # imbalanced_rhs.head(50)
    smi_list = imbalanced_rhs.rxnsmiles0.sample(data_count, 
                                                random_state=seed
                                                )
    smi_list = smi_list.apply(_remove_duplicate).to_list()

    if smi_his:
        smi_his.extend(smi_list)
        smi_list = list(OrderedDict.fromkeys(smi_his).keys())

    SMI = []
    PRED_MOLS = []
    for smi in tqdm(smi_list):
        possible_mols = possible_missing_mols(smi,fill_mask)
        for mol in possible_mols:
            SMI.append(smi)
            PRED_MOLS.append(mol)
      
    rhs_result = pd.DataFrame(data={'predicted_mols': PRED_MOLS},index=SMI)
    rhs_result.to_csv(pred_data_file)

    return rhs_result



def pred_missing_mol(input_data_file, pred_data_file, fill_mask, data_count=50, seed=42, smi_his=None, mask_side='right'):
    
    # Predict LHS or RHS missing molecules from reaction SMILES list or raw ChemBalancer result dataframe. 

    data = pd.read_csv(input_data_file,index_col=0)
    if mask_side == 'right':
        message = 'RHS species insufficient'
    else:
        message = 'LHS'

    if 'msg' in data.columns:
        imbalanced_rxn = data.loc[data.msg==message]
        # imbalanced_rhs.head(50)
    else:
        imbalanced_rxn = data

    smi_list = imbalanced_rxn.rxnsmiles0.sample(data_count, 
                                                random_state=seed
                                                )
    smi_list = smi_list.apply(_remove_duplicate).to_list()

    if smi_his:
        smi_his.extend(smi_list)
        smi_list = list(OrderedDict.fromkeys(smi_his).keys())

    SMI = []
    PRED_MOLS = []
    for smi in tqdm(smi_list):
        possible_mols = possible_missing_mols(smi,fill_mask,mask_side=mask_side)
        for mol in possible_mols:
            SMI.append(smi)
            PRED_MOLS.append(mol)
      
    rhs_result = pd.DataFrame(data={'predicted_mols': PRED_MOLS},index=SMI)
    rhs_result.to_csv(pred_data_file)

    return rhs_result


