import psycopg2
import sys
import re
import numpy as np
import pandas as pd
import sqlite3
import xml.etree.ElementTree as ET
from datasets import concatenate_datasets,load_from_disk,Dataset
from tqdm import tqdm
import json
import hashlib
import shutil
import warnings

warnings.filterwarnings("ignore", message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`.")

sys.path.append('../TapexGraph')
from add_utils import deserializ_tapex_linear_table,escape_special_characters,fix_sql_in_tapex_query
from utils import get_sqlite_data,get_tapex_execution_plan
from datasets import concatenate_datasets,load_from_disk,Dataset

#from datasets.utils.logging import disable_progress_bar
#disable_progress_bar()
from concurrent.futures import ProcessPoolExecutor
import itertools
from tqdm import tqdm
# Чтение данных из файло
def read_questions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            yield line.strip()
#questions = list(read_questions('tapex_pretrain/train.src'))
#answers = list(read_questions('tapex_pretrain/train.tgt'))
# Проверка, что количество вопросов и ответов совпадает
#assert len(questions) == len(answers), "Количество вопросов и ответов должно совпадать."
# Создание словаря для Dataset

# Создание Dataset


def map_function_for_question_change(example):

    #print(example['question'])
    #try:
    tab_name='f'+hashlib.md5(example['question'].encode('utf-8')).hexdigest()
    query,plan = get_tapex_execution_plan(example['question'], tab_name=tab_name)
    example['xml_plan'] = plan if plan != None else "None"
    example['question'] = query
    #except Exception as e:
     #   print(e)
      #  print(example['query'])
    return example
    
start_chank_id = 0
chank_id = 0
bach_size = 100000
questions_gen = read_questions('../TapexGraph/tapex_pretrain/train.src')
answ_gen = read_questions('../TapexGraph/tapex_pretrain/train.tgt')
for i in range(start_chank_id):
    list(itertools.islice(questions_gen, bach_size))
    list(itertools.islice(answ_gen, bach_size))
chank_id = start_chank_id

while True:
    question = list(itertools.islice(questions_gen, bach_size))
    answer = list(itertools.islice(answ_gen, bach_size))
    print(f'----------------{chank_id}---------------------')
    if not question:
        break
    dataset = Dataset.from_dict({
        'question': question,
        'answer': answer
    })
    print("Загрузил данные")
    print("Start processing")
    dataset = dataset.map(map_function_for_question_change,num_proc=17)
    print(dataset[0])

    print("DATA TRANSFORMED. START SAVING")
    dataset.save_to_disk(f'./converved_to_plan_tapex_data_{chank_id}')
    chank_id+=1
# In[10]:
dataset = Dataset.from_dict({
        'question': [],
        'answer': [],
        'xml_plan': []
    })
for chank in range(chank_id):
    dataset_path = f'./converved_to_plan_tapex_data_{chank}'
    data = load_from_disk(dataset_path)
    dataset = concatenate_datasets([dataset,data])
    del data
    shutil.rmtree(f'./converved_to_plan_tapex_data_{chank}')
print("DATA TRANSFORMED. START SAVING")
dataset.save_to_disk(f'./converved_to_plan_tapex_data_full')
