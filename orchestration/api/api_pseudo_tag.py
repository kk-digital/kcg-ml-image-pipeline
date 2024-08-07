from datetime import datetime
from fastapi import APIRouter, Request, HTTPException, Query
from typing import List, Dict
from orchestration.api.mongo_schema.pseudo_tag_schemas import ImagePseudoTagRequest, ImagePseudoTag, ListImagePseudoTag, ImagePseudoTagRequestV1, ListImagePseudoTagScores
from typing import Union
from .api_utils import PrettyJSONResponse, validate_date_format, ApiResponseHandlerV1, ErrorCode, StandardSuccessResponseV1, WasPresentResponse, VectorIndexUpdateRequest, PseudoTagIdResponse, TagCountResponse, ListImageTag
import traceback
from bson import ObjectId
import logging
import time
from typing import Optional



router = APIRouter()



@router.delete("/pseudotag/remove-all",
            description="Remove all documents from the image ranks collection",
            tags=["image-ranks"],
            status_code=200)
async def remove_all_image_ranks(request: Request):
    try:
        delete_result = request.app.pseudo_tag_images_collection.delete_many({})
        return {"status": "success", "deleted_count": delete_result.deleted_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
