class LazyTable:

    def __init__(self, plan):
        self.plan = plan

    def to_ibis(self):
        if isinstance(self.plan, PlanNode):
            return self.plan.to_ibis()
        return self.plan
    
    def execute(self):
        ibis_table = self.to_ibis()
        return ibis_table.execute()
        
    def show_plan(self):
        return repr(self.plan)
    
class PlanNode:

    def to_ibis(self):
        raise NotImplementedError
    
class MapNode(PlanNode):
    def __init__(self, map_obj, source):
        self.map_obj = map_obj
        self.source = source

    def to_ibis(self):
        table = self.source.to_ibis()
        return self.map_obj(self.map_obj.llm, table)
    
class FilterNode(PlanNode):
    def __init__(self, filter_obj, source):
        self.filter_obj = filter_obj
        self.source = source

    def to_ibis(self):
        table = self.source.to_ibis()
        return self.filter_obj(self.filter_obj.llm, table)



    
