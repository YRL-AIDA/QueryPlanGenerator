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
sys.path.append('../TapexGraph')
from add_utils import deserializ_tapex_linear_table,escape_special_characters
from utils import get_sqlite_data,get_sqall_execution_plan


def parse_xml(element, indent=0,odjekts_description = [],parent = None,node_name = None,width = 0,depth = 0):
    
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
                    'Async-Capable'
                   }
    space = ' ' * indent
    pattern = r'\{.+\}'
    

    # Отображаем тег элемента
    tag = re.sub(pattern,'',element.tag)
    
    if tag not in ['Plan','Plans']:
        
        if tag not in ignore_names:
            #print(f'Node {node_name}')
            #print(f"{space}Тег: <{tag}>")
            #print(f"{space}Parent: {parent}")
            # Отображаем атрибуты элемента, если они есть
        
            #if element.attrib:
        
             #   print(f"{space}  Атрибуты: {element.attrib}")
        
                
        
            # Печать текста, если он присутствует
        
            #if element.text and element.text.strip():# and (tag=='Node-Type' or tag=='Parent-Relationship' or tag=='Relation-Name'):
        
             #   print(f"{space}  Текст: {element.text.strip()}")
        
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
            
            
            width = 1
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
                    indent+=2
            return {'somesing_else': ""}




if __name__ == '__main__':
    OBDJECTS_FILE_PATH = './squall_objects.json'
    NEW_SQUALL_FILE_PATH = './squall_with_plan.json'
    SQUALL_DATASET_JSON_PATH = '/media/sunveil/Data/header_detection/poddubnyy/postgraduate/squall/data/squall.json'
    with open(SQUALL_DATASET_JSON_PATH,'r') as inf:
        squall = json.load(inf)
    odjects=[] 
    print('Start data processin')
    for i,data in enumerate(tqdm(squall)):
        print(i)
        xml_str_exec_plan = get_sqall_execution_plan(data) 
        squall[i]['xml_plan'] = xml_str_exec_plan if xml_str_exec_plan != None else ''
        if xml_str_exec_plan != None:
            parse_xml(ET.fromstring(xml_str_exec_plan),odjekts_description=odjects)
    print('End data processin')
    print('Start save data')
    with open(OBDJECTS_FILE_PATH,'w') as outf:
        json.dump(odjects,outf)
    print('Odjects is saved')
    with open(NEW_SQUALL_FILE_PATH,'w') as outf:
        json.dump(squall,outf)
    print('New Squall data is saved')