# Author: Ning Li, contact: edwardln@stu.xjtu.edu.cn or l.ning@itv.rwth-aachen.de
# Referenced to https://github.com/rxn4chemistry/rxnfp/blob/master/nbs/09_fine_tune_bert_on_uspto_1k_tpl.ipynb

import os
import numpy as np
import pandas as pd
import torch
import logging
import random
import pkg_resources
import sklearn
from rxnfp.models import SmilesClassificationModel
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
logger = logging.getLogger(__name__)

# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())

ans = torch.load(r'..\data\database.npy')
rxn_smiles = list(ans['rxn_smiles'])
labels = list(ans['rc'])
for i in range(len(labels)):
    labels[i] -= 1

# here all data are used, as classification accuracy is not important here
train_ed = pd.DataFrame({'text': rxn_smiles, 'labels': labels})
eval_ed = pd.DataFrame({'text': rxn_smiles, 'labels': labels})
print('go go go!')

model_args = {
    'wandb_project': None, 'num_train_epochs': 5, 'overwrite_output_dir': True,
    'learning_rate': 2e-5, 'gradient_accumulation_steps': 1,
    'regression': False, "num_labels":  9, "fp16": False,
    "evaluate_during_training": True, 'manual_seed': 42,
    "max_seq_length": 512, "train_batch_size": 8, "warmup_ratio": 0.00,
    'output_dir': '../rxnfp_schwaller/models/transformers/test',
    'thread_count': 8,
    }


if __name__ == '__main__':
    model_path = pkg_resources.resource_filename("rxnfp", "models/transformers/bert_pretrained")
    model = SmilesClassificationModel("bert", model_path, num_labels=9, args=model_args, use_cuda=torch.cuda.is_available())
    model.train_model(train_ed, eval_df=eval_ed, acc=sklearn.metrics.accuracy_score, mcc=sklearn.metrics.matthews_corrcoef)

