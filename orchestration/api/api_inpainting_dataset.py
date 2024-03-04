from fastapi import Request, HTTPException, APIRouter, Response, Query
from utility.minio import cmd

router = APIRouter()


@router.get("/datasets-inpainting/list")
def get_datasets(request: Request):
    objects = cmd.get_list_of_objects(request.app.minio_client, "datasets-inpainting")

    return objects