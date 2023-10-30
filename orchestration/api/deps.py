from datetime import datetime
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from orchestration.api.jwt import (
    ALGORITHM,
    JWT_SECRET_KEY
)
from dotenv import dotenv_values 
from jose import jwt
from pydantic import ValidationError
from orchestration.api.mongo_schemas import TokenPayload

reuseable_oauth = OAuth2PasswordBearer(
    tokenUrl="/users/login",
    scheme_name="tokens"
)

config = dotenv_values("./orchestration/api/.env")

async def is_authenticated(request: Request, token: str = Depends(reuseable_oauth)):
    token_data=get_user_payload(token=token)
    user=request.app.users_collection.find_one({"username": token_data.sub})
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Could not find user",
        )
    
    elif user['is_active']==False:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="user is deactivated",
        )
    
    return user

async def is_admin(request: Request, token: str = Depends(reuseable_oauth)):

    token_data=get_user_payload(token=token)
    user=request.app.users_collection.find_one({"username": token_data.sub})
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Could not find user",
        )
    
    elif user['is_active']==False:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="user is deactivated",
        )
    
    elif user['role']!="admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="user doesn't have neccessary rights",
        )
    
    return user

def get_user_payload(token: str):
    try:
        payload = jwt.decode(
            token, JWT_SECRET_KEY, algorithms=[ALGORITHM]
        )
        token_data = TokenPayload(**payload)
        
        if datetime.fromtimestamp(token_data.exp) < datetime.now():
            raise HTTPException(
                status_code = status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except(jwt.JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return token_data
        