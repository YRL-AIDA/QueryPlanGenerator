import sys
import re
import numpy as np
import pandas as pd
import sqlite3
import psycopg2
from psycopg2 import sql
import sys
sys.path.append('../TapexGraph')
from add_utils import deserializ_tapex_linear_table,escape_special_characters,get_sql_and_table,fix_sql_in_tapex_query,serialize_table_to_tapex_format
def get_sqlite_data(tbl):
    db_file = f"/media/sunveil/Data/header_detection/poddubnyy/postgraduate/squall/tables/db/{tbl}.db"
    conn = sqlite3.connect(db_file)
    df = pd.read_sql_query("SELECT * FROM w", conn)
    del df['id']
    return df

def read_questions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            yield line.strip()

#def get_sql_and_table(example):
 #   pattern = ' col : '
  #  try:
   #     sql,table = example.split(pattern)
    #except Exception as e:
        #print(e)
#        return "None", "None"
 #   sql  = sql.strip()
  #  df = deserializ_tapex_linear_table(" col : "+table)
   # 
   # new_column_names = [f'"{"_".join(head.split(" "))}"' for head in df.columns]
   # #print(new_column_names)
   # for head in sorted(set(df.columns),key=len, reverse=True):
   #     sql = sql.replace(head,f'{"_".join(head.split(" "))}')
   # #print(sql)
   # for head in set(df.columns):  
   #     #print(escape_special_characters("_".join(head.split(" "))))
   #     sql = re.sub(f' {escape_special_characters("_".join(head.split(" ")))} '
   #                  ,f" '{'_'.join(head.split(' '))}' ",sql) 
   #     sql = sql.replace(head,f'"{"_".join(head.split(" "))}"') 
   #     #print(sql)
   # df.columns = new_column_names
   # del new_column_names
   # df['agg'] = np.zeros(df.shape[0])
   #return sql,df

def covichki(p):
    return f"'{p}'"
    
def join_params(params):
    #return f"({','.join(params.apply(lambda x:covichki(x) if type(x) == str else str(x)).values)})"
    return f"({','.join(params.apply(lambda x:str(x)).values)})"
    
def get_psql_type(dtype,schema=None,column = None):
    if schema != None:
        return schema[column]
    return {
        'int64': 'BIGINT',
        'float64': 'DOUBLE PRECISION',
        'object': 'TEXT',
        'datetime64[ns]': 'TIMESTAMP'
    }.get(str(dtype), 'TEXT')

def convert_nat_to_none(x):

    return None if pd.isna(x) else x


def get_query_execution_plan(table,sql_,tab_name='w',schema=None):
    host = '192.168.19.148' #'192.168.19.46' #'192.168.19.148'
    database = 'tapex'
    user = 'postgres'
    password = '0000'
    port = 5432 #5434
    answer = None
    cursor = None
    connection = None
    try:
        explain_sql = 'EXPLAIN (ANALYZE,FORMAT XML) ' + sql_
        explain_sql = sql.SQL(explain_sql)
        create_table_query = sql.SQL(
            "CREATE TABLE IF NOT EXISTS {tbl} (id SERIAL PRIMARY KEY, {columns})"
        ).format(
            tbl=sql.Identifier(tab_name),
            columns=sql.SQL(', ').join(
                sql.SQL("{} {}").format(
                    sql.Identifier(column), sql.SQL(get_psql_type(table.dtypes[column],schema=schema,column=column))
                    ) for column in table.columns
                )
            )
        #insert_data_query = f'INSERT INTO {tab_name} ({",".join([key for key in data.columns if key !="id"])}) VALUES '\
                            #f'{",".join(["%s" for _,x in data.iterrows()])};'
        insert_data_query = sql.SQL(
            "INSERT INTO {tbl} ({fields}) VALUES ({placeholders})"
        ).format(
            tbl=sql.Identifier(tab_name),
            fields=sql.SQL(', ').join(map(sql.Identifier, table.columns)),
            placeholders=sql.SQL(', ').join(sql.Placeholder() for _ in table.columns)
        )
        drop_table_query = sql.SQL('DROP TABLE IF EXISTS {tbl}').format(tbl=sql.Identifier(tab_name))
        # Подключение к базе данных
        connection = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            port=port
        )
        cursor = connection.cursor()
        # Получение информации о версии postgres
        
        
        cursor.execute(create_table_query)
            
        # Используем executemany для вставки множества строк
    
        cursor.executemany(insert_data_query, [(x.apply(convert_nat_to_none)).to_list() for _,x in table.iterrows()])  
        cursor.execute(explain_sql)
        answer = cursor.fetchall()
        cursor.execute(drop_table_query)
        connection.commit()
        #version = cursor.fetchone()[0]
        #print(f"Версия PostgreSQL: {version}")
    except Exception as e:
        pd.set_option('display.max_rows', None)
        print("Ошибка при работе с БД:",e)
        print(sql_)
        print(table)
        print(table.dtypes)
    finally:
        if cursor:
            cursor.close()  # Закрытие курсора
        if connection:
            connection.close()  # Закрытие соединения
        return answer



def get_sqall_execution_plan(squall_example):
    table = get_sqlite_data(squall_example['tbl'])
    sql = ' '.join([s[1] for s in squall_example['sql']])
    answer = get_query_execution_plan(table,sql)
    
    return answer[0][0] if answer != None else answer

def get_tapex_execution_plan(question,tab_name='w'):
    sql,tbl = get_sql_and_table(question)
    answer = None
    sql = fix_sql_in_tapex_query(sql,tab_name=tab_name)
    if sql != None or "None":
        
        answer = get_query_execution_plan(tbl,sql,tab_name=tab_name)
        #print("PLANING",answer)
    return f'{sql} {serialize_table_to_tapex_format(tbl)}',answer[0][0] if answer != None else answer

def get_wikisql_execution_plan(sql,tbl,tab_name,schema=None):
    answer = None
    answer = get_query_execution_plan(tbl,sql,tab_name=tab_name,schema=schema)
        #print("PLANING",answer)
    return answer[0][0] if answer != None else answer

    
def replace_depth(node_name,max_depth):
    text_arr = node_name.split('_')
    correct_depth = int(text_arr[-1]) - max_depth
    text_arr[-1] = str(np.sign(correct_depth)*correct_depth)
    return '_'.join(text_arr)
    
def reflect_depth_indexes(df):
    depth_max = int(df['node_name'].apply(lambda x: int(x.split('_')[-1]) if x!= None else None).max())
    print(depth_max)
    df['node_name'] = df['node_name'].apply(lambda x: replace_depth(x,depth_max) if x!= None else None)


def parse_xml(element, indent=0,odjekts_description = [],parent = None,node_name = None,width = 0,depth = 0,print_=False):
    
    ignore_names = {'Startup-Cost',
                    'Total-Cost',
                    'Plan-Rows',
                    'Plan-Width',
                    'Actual-Startup-Time',
                    'Actual-Total-Time',
                    'Actual-Rows',
                    'Actual-Loops',
                    'Planning-Time',
                    'Triggers',
                    'Execution-Time',
                    'Parallel-Aware',
                    'Async-Capable',
                    'Rows-Removed-by-Filter',
                    'Partial-Mode',
                    'Strategy',
                    'Parent-Relationship',
                    'Sort-Space-Used',
                    'Sort-Space-Type',
                    'Sort-Method',
                    'Rows-Removed-by-Index-Recheck',
                    'HashAgg-Batches',
                    'Peak-Memory-Usage',
                    'Disk-Usage',
                    'Subplans-Removed',
                    'Rows-Removed-by-Join-Filter',
                    'Hash-Buckets',
                    'Original-Hash-Buckets',
                    'Hash-Batches',
                    'Original-Hash-Batches',
                    'Heap-Fetches',
                
                   }
    space = ' ' * indent
    pattern = r'\{.+\}'
    

    # Отображаем тег элемента
    tag = re.sub(pattern,'',element.tag)
    
    if tag not in ['Plan','Plans']:
        
        if tag not in ignore_names:
            if print_:
                print(f'Node {node_name}')
                print(f"{space}Тег: <{tag}>")
                print(f"{space}Parent: {parent}")
                # Отображаем атрибуты элемента, если они есть
        
            if print_:
                if element.attrib:
        
                    print(f"{space}  Атрибуты: {element.attrib}")
        
                
        
            # Печать текста, если он присутствует
            if print_:
                if element.text and element.text.strip():# and (tag=='Node-Type' or tag=='Parent-Relationship' or tag=='Relation-Name'):
            
                    print(f"{space}  Текст: {element.text.strip()}")
            
            if tag == 'Node-Type':
                node_name = element.text.strip()+f'_{width}_{depth}'
                odjekts_description.append({'teg':str(tag),
                                            'node_name': node_name,
                                        'attr':[*element.attrib] if element.attrib else [],
                                        'parent_teg': parent,
                                        'text': node_name
                                       })
                
                return {'node_name': node_name}
            else:
                
                odjekts_description.append({'teg':str(tag),
                                            'node_name': node_name,
                                        'attr':[*element.attrib] if element.attrib else [],
                                        'parent_teg': parent,
                                        'text': element.text.strip() if element.text and element.text.strip() else ''
                                       })
            #odjekts_description.append({'teg':str(tag),
                #                            'node_name': element.text.strip() if element.text and element.text.strip() else ''
                  #                      'attr':[*element.attrib] if element.attrib else [],
                   #                     'parent_teg': parent,
                    #                    'text': element.text.strip() if element.text and element.text.strip() else ''
                     #                  })
        # Рекурсивно проходим по всем дочерним элементам
            
                for child in element:
            
                    parse_xml(child, indent + 4,odjekts_description,tag,node_name,width,depth)
                return {'somesing_else': ""}
    else:
        if tag == 'Plans':
            
            
            width = 0
            parent = node_name
            node_name=None
            for child in element:
                
                parse_xml(child, indent + 4,odjekts_description,parent,node_name,width,depth+1)
                width+=1
            return {'somesing_else': ""}
        if tag == 'Plan':
            for child in element:
                type_name = parse_xml(child, indent + 4,odjekts_description,parent,node_name,width,depth)
                if type_name != None and 'node_name' in type_name.keys():
                    node_name = type_name['node_name']
                    parent =  node_name
                    indent+=2
            return {'somesing_else': ""}