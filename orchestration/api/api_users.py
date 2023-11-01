from orchestration.api.deps import is_authenticated,is_admin
from orchestration.api.mongo_schemas import User
from fastapi import status, HTTPException, Depends, APIRouter, Request, Query
from fastapi.security import OAuth2PasswordRequestForm
from orchestration.api.jwt import (
    get_hashed_password,
    create_access_token,
    create_refresh_token,
    verify_password
)
from uuid import uuid4

router = APIRouter()

@router.post('/users/create', summary="Create new user")
def create_user(request: Request, data: User):
    # querying database to check if user already exist
    user=request.app.users_collection.find_one({"username": data.username})
    if user is not None:
            raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this name already exist"
        )
    
    user = {
        'username': data.username,
        'password': get_hashed_password(data.password),
        'role': data.role,
        'is_active': True,
        'uuid': str(uuid4())
    }
    request.app.users_collection.insert_one(user)
    # remove the auto generated field
    user.pop('_id', None)
    return user


@router.post('/users/login', summary="Create access and refresh tokens for user")
def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    user=request.app.users_collection.find_one({"username": form_data.username})
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect username or password"
        )

    hashed_pass = user['password']
    if not verify_password(form_data.password, hashed_pass):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect username or password"
        )
    
    return {
        "access_token": create_access_token(user['username']),
        "refresh_token": create_refresh_token(user['username']),
    }

# Deactivate a user by username
@router.put("/users/deactivate")
def deactivate_user(request:Request, username: str= Query(...), user: User = Depends(is_admin)):
    # Define the update to apply to the document
    update = {
        "$set": {
            "is_active": False
        }
    }
    request.app.users_collection.update_one({"username":username}, update)
    return {'message':f"user {username} deactivated successfully"}

# Reactivate a user by username
@router.put("/users/reactivate")
def reactivate_user(request:Request, username: str= Query(...), user: User = Depends(is_admin)):
    # Define the update to apply to the document
    update = {
        "$set": {
            "is_active": True
        }
    }
    request.app.users_collection.update_one({"username":username}, update)
    return {'message':f"user {username} reactivated successfully"}

# Delete a user by username
@router.delete("/users/delete")
def delete_user(request:Request, username: str= Query(...), user: User = Depends(is_admin)):
    request.app.users_collection.delete_one({"username":username})
    return {'message':f"user {username} deleted successfully"}

#list of users
@router.get('/users/list')
def list_users(request:Request, user: User = Depends(is_admin)):
    users = list(request.app.users_collection.find({}))

    for user in users:
        user.pop('_id', None)

    return users
