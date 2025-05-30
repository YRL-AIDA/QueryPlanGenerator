import re
import sys

sys.path.append('../TapexGraph')

from lib.dbengine import DBEngine
from lib.query import Query
import json
import numpy as np
import pandas as pd
import hashlib
from tqdm import tqdm
from add_utils import sort_graphe_execute_nodes, serialize_table_to_tapex_format, to_num_datatime
from utils import get_wikisql_execution_plan
from datasets import load_from_disk,Dataset

import multiprocessing as mp
out = 'train'
def process_line(args):
    ls,tables_dict, bd_path = args
    try:
        engine = DBEngine(bd_path)
        xml_plan = None
        eg = json.loads(ls)
        sql = Query.from_dict(eg['sql'])
        sql_str, table_id, where_map, schema, gold = engine.execute_query(eg['table_id'], sql, lower=True)
        for key in where_map:
            sql_str = re.sub(f':{key}', str(where_map[key]), sql_str)
        table = pd.DataFrame(engine.execute_str(f'SELECT  * FROM {table_id}').as_dict())
        schema['agg'] = 'real'
        table['agg'] = np.zeros(table.shape[0])
        # table = table.apply(to_num_datatime)  # если нужно, раскомментируйте
        
        plan = get_wikisql_execution_plan(sql_str, table, tab_name=table_id, schema=schema)
        xml_plan = plan if plan is not None else "None"
        
        if xml_plan != "None":
            sql_query = f"{sql_str} {serialize_table_to_tapex_format(table)}"
            table.columns = [*tables_dict[eg['table_id']]['header'],'agg']
            nl_query = f"{eg['question']} {serialize_table_to_tapex_format(table)}"
            return sql_query, nl_query, str(gold), str(xml_plan), 1
        else:
            return None
    except Exception as e:
        print(e)
        return None


def convert_wikisql_to_queryplan_format_multi(querys_t, tables_t, db_file_t, num_workers=4):
    sql_queries = []
    nl_queries = []
    answers = []
    xml_plans = []
    ppp = 0
    for qf, tf, bd in zip(querys_t, tables_t, db_file_t):
        bd_path = f'../TapexGraph/data/{bd}.db'
        qf_path = f'../TapexGraph/data/{qf}.jsonl'
        tf_path = f'../TapexGraph/data/{tf}.tables.jsonl'
        with open(tf_path) as fp:
            tables_dict = {}
            for table_str in fp:
                table = json.loads(table_str)
                tables_dict[table['id']] = table
            
        with open(qf_path) as fs:
            ls_list = list(fs)
            args_list = [(ls,tables_dict, bd_path) for ls in ls_list]
            with mp.Pool(processes=num_workers) as pool:
                results = list(tqdm(pool.imap(process_line, args_list), total=len(args_list)))
            for res in results:
                if res is not None:
                    sql_question,nl_question, gold, xml_plan, cnt = res
                    sql_queries.append(sql_question)
                    nl_queries.append(nl_question)
                    answers.append(gold)
                    xml_plans.append(xml_plan)
                    ppp += cnt
    return sql_queries, nl_queries, answers, xml_plans, ppp

test = convert_wikisql_to_queryplan_format_multi([out],[out],[out],num_workers=30)
dataset = Dataset.from_dict({
        'sql_question': test[0],
        'nl_question': test[1],
        'answer': test[2],
        'xml_plan': test[3]
    })
dataset.save_to_disk(f'./converved_to_plan_wikisql_{out}_data_full2')
with open(f'{out}.json','w') as outf:
    json.dump(test,outf,indent = 2)