class LazyTable:

    def __init__(self, plan):
        self.plan = plan
    
    def execute(self):
        if isinstance(self.plan, PlanNode):
            return self.plan.execute()
        else:
            return self.plan
        
    def show_plan(self):
        return repr(self.plan)
    
class PlanNode:

    def execute(self):
        raise NotImplementedError("Subclasses must implement execute()")
    
class MapNode(PlanNode):

    def __init__(self, map_obj, source):
        self.map_obj = map_obj
        self.source = source

    def execute(self):
        if isinstance(self.source, LazyTable):
            table = self.source.execute()
        else:
            table = self.source 

        return self.map_obj(self.map_obj.llm, table)
    
class FilterNode(PlanNode):
    def __init__(self, filter_obj, source):
        self.filter_obj = filter_obj
        self.source = source

    def execute(self):
        if isinstance(self.source, LazyTable):
            table = self.source.execute()
        else:
            table = self.source 

        return self.filter_obj(self.filter_obj.llm, table)



    
