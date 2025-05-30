import re
import pandas as pd
import json
import numpy as np

from sql_graph_translate.seq_to_graph import parse
from sql_graph_translate.nodes import create_nodes
from sql_graph_translate.sql_edges import create_edges,simple_create_edges
from sql_graph_translate.metrics import target_values_map, flexible_denotation_accuracy, to_value_list
from sql_graph_translate.utils import find_first_edges,find_last_edges
#from memory_profiler import profile
import psycopg2
import sqlite3
from sqlglot import expressions as exp
from sqlglot import parse_one

def identify_type(input_string):
    try:
        # Попробуем преобразовать строку в целое число
        value = int(input_string)
        return value
    except ValueError:
        try:
            # Попробуем преобразовать строку в дробное число
            value = float(input_string)
            return value
        except ValueError:
            # Если ничего не сработало, это текст
            return input_string
    except TypeError:
        return input_string
        
def deserializ_tapex_linear_table(linear_table):
    
    head_pattern = r" col :(.*) row 1 : "
    row_pattern = r" row \d+ : "
    coll_delimetr = " | "
    heads = [h.strip() for h in 
             re.search(head_pattern,linear_table)[0].replace(" col : ","",1).replace(" row 1 : ","",1).split(coll_delimetr)]
    rows = [[identify_type(r.strip()) for r in row.split(coll_delimetr)] for row in re.split(row_pattern,linear_table)[1:]]
    return pd.DataFrame(rows,columns=heads)

def serialize_table_to_tapex_format(df):
    head_pattern = " col : "
    row_pattern = " row {num} : "
    coll_delimetr = " | "
    
    lin_table = head_pattern+coll_delimetr.join(df.columns)
    for i,row in df.iterrows():
        #print(row_pattern.format(num=i+1))
        lin_table+=row_pattern.format(num=i+1)+coll_delimetr.join(str(r) for r in row.values)
    
    return lin_table
    
#@profile
def translate_query_to_graph_form(query,answer=None,flatten_mode = 'preorder',
                                  Omega_include=["P","C","S","GB","H","OB","A","OP","L"],task='tapex'):
    
    pattern = ' col : '
    try:
        sql,_ = query.split(pattern)
    except Exception as e:
        print('translate_query_to_graph_form ',e)
        return "None", "None"
    sql  = sql.strip()
    df = deserializ_tapex_linear_table(" col : "+query.split(" col : ")[1])
    
    new_column_names = [f'"{"_".join(head.split(" "))}"' for head in df.columns]
    #print(new_column_names)
    for head in sorted(set(df.columns),key=len, reverse=True):
        sql = sql.replace(head,f'{"_".join(head.split(" "))}')
    #print(sql)
    for head in set(df.columns):  
        #print(escape_special_characters("_".join(head.split(" "))))
        sql = re.sub(f' {escape_special_characters("_".join(head.split(" ")))} '
                     ,f' "{"_".join(head.split(" "))}" ',sql) 
        #sql = sql.replace(head,f'"{"_".join(head.split(" "))}"') 
        #print(sql)
    df.columns = new_column_names
    del new_column_names
    df['agg'] = np.zeros(df.shape[0])
    edges, _ = create_edges(sql)
    if len(find_last_edges(edges)) == 0:
        #print('Find Cycle graph')
        return "None", "None"


    del sql
    if task == 'tapex':
        sort_nodes = sort_graphe_execute_nodes(edges)
        return ' NODE '+f' NODE '.join(sort_nodes)+serialize_table_to_tapex_format(df), answer

def translate_query_to_graph_form_new(query,answer=None,flatten_mode = 'preorder',
                                  Omega_include=["P","C","S","GB","H","OB","A","OP","L"],task='tapex'):
    
    sql,df = get_sql_and_table(query)
    edges, _ = simple_create_edges(sql)
    if len(find_last_edges(edges)) == 0:
        #print('Find Cycle graph')
        return "None", "None"


    del sql
    if task == 'tapex':
        sort_nodes = sort_graphe_execute_nodes(edges)
        return ' NODE '+f' NODE '.join(sort_nodes)+serialize_table_to_tapex_format(df), answer

def escape_special_characters(string):
    special_symbols = '*.^$+?{}[]\|()'
    
    return ''.join([f'//{w}' if w in special_symbols else w for w in string])

def find_neighboors(node_name, edges):
    return set(edge[1] for edge in list(filter(lambda t: t[0] == node_name, edges)))

def find_layer_neighboors(parent_node_names,edges):
    layer_neighboors = []
    for node in parent_node_names:
        layer_neighboors.extend(find_neighboors(node,edges))
    return set(layer_neighboors)

def sort_graphe_execute_nodes(edges):
    sorted_nodes = []
    sorted_nodes.extend(find_first_edges(edges))
    layer_neighboors = sorted_nodes
    
    while len(layer_neighboors) > 0:
        #print(layer_neighboors)
        layer_neighboors = find_layer_neighboors(layer_neighboors,edges)
        sorted_nodes.extend(layer_neighboors)
    return [node for node in sorted_nodes if node != None]

def add_from_into_sql(sql:str, tbl_name: str) -> str:
    parsed_sql = parse_one(sql)
    for node in parsed_sql.find_all(exp.Select):
        if not node.args.get('from'):
    
            node.set('from', exp.From(this=exp.Identifier(this=tbl_name)))
    return parsed_sql.sql()

def fix_sql_in_tapex_query(sql: str, tab_name = 'w'):
    try:
        sql  = sql.strip()
        sql = add_from_into_sql(sql,tab_name)
        
        return sql.lower()
    except Exception as e:
        print(f'fix_sql_in_tapex_query {e}')
        print(sql)
        #print(query)
        return None
        
def fromat_squall_to_tapex_data(example: str) -> str:
    return f'{" ".join([s[1] for s in example["sql"]])} {serialize_table_to_tapex_format(get_sqlite_data(example["tbl"]))}'

def to_num_datatime(x):
    y = pd.to_numeric(x,errors='coerce')
    if y[y.notnull()].shape[0]>0:
        #print (y.isnull().shape[0])
        return y
    else:
        y = pd.to_datetime(x,errors='coerce')
        if y[y.notnull()].shape[0]>0:
            return y
        else:
            return x
    #try:
    #    return pd.to_numeric(x)
    #except ValueError:
    #    return x
        
def get_sql_and_table(example):
    pattern = ' col : '
    try:
        sql,table = example.split(pattern)
    
        sql  = sql.strip()+' '
        df = deserializ_tapex_linear_table(" col : "+table)
        
        new_column_names = [f'col_{i}' for i,head in enumerate(df.columns)]
        col_dict = {head : ["_".join(head.split(" ")), f'col_{i}'] for i,head in enumerate(df.columns)}
        #print(new_column_names)
        for head in sorted(set(df.columns),key=len, reverse=True):
            #print(head)
            sql = sql.replace(f' {head} ',f' {col_dict[head][0]} ')
            #print (sql)
        #print(sql)
        for head in set(df.columns):  
            #print(escape_special_characters("_".join(head.split(" "))))
            sql = re.sub(f' {re.escape(col_dict[head][0])} ',f' {col_dict[head][1]} ',sql) 
            #print(sql)
            #sql = sql.replace(head,f'"{"_".join(head.split(" "))}"') 
        #print(sql)
        if ' not null' in sql and ' is not null' not in sql:
            sql = sql.replace(' not null',' is not null')
        df.columns = new_column_names
        df = df.apply(to_num_datatime)
        del new_column_names
        df['agg'] = np.zeros(df.shape[0])
    except Exception as e:
        print('get_sql_and_table',e)
        return "None", "None"
    return sql,df