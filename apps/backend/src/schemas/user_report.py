from pydantic import BaseModel, EmailStr, condecimal
from typing import Optional
from datetime import datetime

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserRead(BaseModel):
    id: int
    name: str
    email: str
    created_at: datetime

    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class ReportCreate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    lat: condecimal(gt=-90, lt=90)
    lon: condecimal(gt=-180, lt=180)

class ReportRead(BaseModel):
    id: int
    title: Optional[str]
    description: Optional[str]
    image_path: Optional[str]
    created_at: datetime

    class Config:
        orm_mode = True
