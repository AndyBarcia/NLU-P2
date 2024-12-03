from typing_extensions import Any

class MappingStrID:
    def __init__(self, include_padding:bool = False):
        if include_padding:
            self.obj_to_id_map = { "<PAD>": 0 }
            self.id_to_obj_map = { 0: "<PAD>" }
        else:
            self.obj_to_id_map = dict()
            self.id_to_obj_map = dict()

    def register_obj(self, obj: Any) -> int:
        if obj in self.obj_to_id_map:
            return self.obj_to_id_map[obj]
        else:
            id = len(self.obj_to_id_map)
            self.obj_to_id_map[obj] = id
            self.id_to_obj_map[id] = obj
            return id
    
    def obj_to_id(self, string:str) -> int:
        return self.obj_to_id_map.get(string, len(self.id_to_obj_map))

    def id_to_obj(self, id:int) -> str:
        return self.id_to_obj_map.get(id, "<UNK>")
    
    def __len__(self):
        return len(self.id_to_obj_map)