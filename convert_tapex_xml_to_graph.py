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

from utils import parse_xml
from datasets import concatenate_datasets,load_from_disk,Dataset
from plan_graph import PlanGraph
#from datasets.utils.logging import disable_progress_bar
#disable_progress_bar()
from concurrent.futures import ProcessPoolExecutor
import itertools
from tqdm import tqdm
# Чтение данных из файло
#questions = list(read_questions('tapex_pretrain/train.src'))
#answers = list(read_questions('tapex_pretrain/train.tgt'))
# Проверка, что количество вопросов и ответов совпадает
#assert len(questions) == len(answers), "Количество вопросов и ответов должно совпадать."
# Создание словаря для Dataset

# Создание Dataset


def map_function_for_question_change(example):

    #print(example['question'])
    #try:
    if example['xml_plan'] != "None":
        df = []
        parse_xml(ET.fromstring(example['xml_plan']),odjekts_description= df)
        example['graph_answer'] = PlanGraph(pd.DataFrame(df)).linearize(node_sep=' NODE ')
    else:
        example['graph_answer'] = "None"
    #except Exception as e:
     #   print(e)
      #  print(example['query'])
    return example
dataset = load_from_disk('./converved_to_plan_tapex_data_full')    
dataset = dataset.map(map_function_for_question_change,num_proc=17)
print("DATA TRANSFORMED. START SAVING")
dataset.save_to_disk(f'./converved_to_graph_plan_tapex_data_full')