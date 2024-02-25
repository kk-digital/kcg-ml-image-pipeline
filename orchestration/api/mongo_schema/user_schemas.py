from pydantic import BaseModel, Field, constr, validator
from typing import List, Union, Optional
import re


class User(BaseModel):
    username: str = Field(...)
    password: str = Field(...)
    role: constr(pattern='^(admin|user)$') = Field(...)

    def to_dict(self):
        return {
            "username": self.username,
            "password": self.password,
            "role": self.role
        }
    
class LoginRequest(BaseModel):
    username: str = Field(...)
    password: str = Field(...)