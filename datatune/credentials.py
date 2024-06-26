from datatune.api import API
from datatune.entity import Entity
from typing import Optional, TYPE_CHECKING
from datatune.workspace import Workspace


class Credentials:
    def __init__(self, id: str, workspace: Workspace):
        self.id = id
        self.workspace = workspace

    @property
    def entity(self) -> Entity:
        return self.workspace.entity

    @property
    def name(self) -> str:
        return self.workspace.api.get_credentials(
            self.id
        )["name"]

    @property
    def api(self) -> API:
        return self.workspace.api
    
    @property
    def details(self) -> dict:
        credentials_info =  self.workspace.api.get_credentials(
            self.id
        )
        return {
            "namespace": credentials_info['namespace'],
            "name": credentials_info['name'], 
            "path": credentials_info['path'],
            "description": credentials_info['description'] , 
            "credentials_id": credentials_info['id']
        }
    
 
 