class PLanNode: 
    def __init__(self, node_type,width,depth,parent,args=None,name_delimiter = '_'):
        self.node_type = node_type
        self.parent = parent
        self.args = args if args != None else {}
        self.width = width
        self.depth = depth
        self.node_name = name_delimiter.join([node_type,width,depth])

    def linearize(self,delimiter='|',insert_width_depth=True,insert_parent = True):
        exit_arr = [self.node_type]
        if insert_width_depth:
            exit_arr.extend([self.width,self.depth])
        if insert_parent:
            exit_arr.extend(self.parent)
        exit_arr.extend([f'{key} : {val}'for key,val in self.args.items() ])
        return delimiter.join(exit_arr)
    def execute_node(self,table):
        pass