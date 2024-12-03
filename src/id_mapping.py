from typing_extensions import Any

class MappingStrID:
    def __init__(self, include_padding:bool = False):
        if include_padding:
            self.obj_to_id = { "<PAD>": 0 }
            self.id_to_obj = { 0: "<PAD>" }
        else:
            self.obj_to_id = dict()
            self.id_to_obj = dict()

    def register_obj(self, obj: Any) -> int:
        if obj in self.obj_to_id:
            return self.obj_to_id[obj]
        else:
            id = len(self.obj_to_id)
            self.obj_to_id[obj] = id
            self.id_to_obj[id] = obj
            return id
    
    def obj_to_id(self, string:str) -> int:
        return self.obj_to_id.get(string, "<UNK>")

    def id_to_obj(self, id:int) -> str:
        return self.id_to_obj.get(id, len(self.id_to_obj))
    
    def __len__(self):
        return len(self.id_to_obj) + 1