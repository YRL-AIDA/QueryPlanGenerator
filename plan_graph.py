
from plan_node import PLanNode
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
class PlanGraph:
    def __init__(self, df):
        self.nodes = []
        self.edges = []
        self._preprocess_data(df)
        for node in df.groupby('node_name'):
            name_delimiter = '_'
            node_name_arr = node[0].split(name_delimiter)
            parent = node[1][node[1]['teg']== 'Node-Type']['parent_teg']
            self.edges.extend((parent,node[0]))
            args = {self._create_arg_name(item):item['text'] for i,item in node[1][node[1]['teg'] != 'Node-Type'].iterrows() 
                    if item['teg'] not in node[1]['parent_teg'].values}
            
            #args = None
            if len(node_name_arr) == 3:
                node_type,width,depth = node_name_arr
            elif len(node_name_arr) == 1:
                node_type = node_name_arr[0]
                width,depth = 0,0
            self.nodes.append(PLanNode(node_type,width,depth,parent,args=args,name_delimiter=name_delimiter))

    def linearize(self,node_sep= ' ', delimiter='|',insert_width_depth=True,insert_parent = True):
        return node_sep + node_sep.join([node.linearize(delimiter=delimiter,
                                       insert_width_depth=insert_width_depth,
                                       insert_parent = insert_parent) for node in self.nodes])
            
    def _replace_depth(self,node_name,max_depth):
        text_arr = node_name.split('_')
        if len(text_arr) == 3:
            correct_depth = int(text_arr[-1]) - max_depth
            text_arr[-1] = str(np.sign(correct_depth)*correct_depth)
            return '_'.join(text_arr)
        else :
            return node_name
    def _reflect_depth_indexes(self,df):
        depth_max = int(df['node_name'].apply(lambda x: int(x.split('_')[-1]) if x!= None else None).max())
        #print(depth_max)
        df.loc[:, 'node_name'] = df['node_name'].apply(lambda x: self._replace_depth(x,depth_max) if x!= None else None)
        df.loc[:, 'parent_teg'] = df['parent_teg'].apply(lambda x: self._replace_depth(x,depth_max) if x!= None else None)
        def execute(self,table):
            pass
    def _preprocess_data(self,df):
        self._reflect_depth_indexes(df)
        df = df[df['node_name'].notna()]
        order = sorted(df['node_name'].unique(), key=self._sort_key)
        category_dtype = CategoricalDtype(categories=order, ordered=True) 
        df.loc[:, 'node_name'] = df['node_name'].astype(category_dtype)
        
    def _sort_key(self,x):
        parts = x.split('_')
        depth = int(parts[-1])
        width = int(parts[-2])
        return (depth, width)
    def _create_arg_name(self,item):
        return item['teg'] if item['parent_teg']==item['node_name'] else f'{item["parent_teg"]}_{item["teg"]}'
