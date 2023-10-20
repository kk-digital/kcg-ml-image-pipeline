from datetime import datetime
from fastapi import APIRouter, Request, HTTPException, Query
from typing import List, Dict
from orchestration.api.mongo_schemas import TagDefinition, ImageTag
from typing import Union
from .api_utils import PrettyJSONResponse


router = APIRouter()

@router.put("/tags/add_new_tag_definition")
def add_new_tag_definition(request: Request, tag_data: TagDefinition):
    date_now = datetime.now()
    
    # Find the maximum tag_id in the collection
    last_entry = request.app.tag_definitions_collection.find_one({}, sort=[("tag_id", -1)])
    
    if last_entry and "tag_id" in last_entry:
        new_tag_id = last_entry["tag_id"] + 1
    else:
        new_tag_id = 0

    # Check if the tag definition exists
    query = {"tag_string": tag_data.tag_string}
    existing_tag = request.app.tag_definitions_collection.find_one(query)

    if existing_tag is None:
        # If tag definition doesn't exist, add it
        tag_data.tag_id = new_tag_id
        tag_data.creation_time = date_now.strftime('%Y-%m-%d %H:%M:%S')
        request.app.tag_definitions_collection.insert_one(tag_data.to_dict())
        return {"status": "success", "message": "Tag definition added successfully.", "tag_id": new_tag_id}
    else:
        # If tag definition already exists, update its details 
        new_values = {
            "$set": {
                "tag_category": tag_data.tag_category,
                "creation_time": date_now.strftime('%Y-%m-%d %H:%M:%S'),
                "user_who_created": tag_data.user_who_created
            }
        }
        request.app.tag_definitions_collection.update_one(query, new_values)
        return {"status": "success", "message": "Tag definition updated successfully.", "tag_id": existing_tag["tag_id"]}


@router.put("/tags/rename_tag_definition")
def rename_tag_definition(request: Request, tag_id: int, new_tag_string: str):
    # Find the tag definition using the provided tag_id
    query = {"tag_id": tag_id}
    existing_tag = request.app.tag_definitions_collection.find_one(query)
    
    if existing_tag:
        # Update the tag name if it exists
        new_values = {"$set": {"tag_string": new_tag_string}}
        request.app.tag_definitions_collection.update_one(query, new_values)
        return {"status": "success", "message": "Tag definition renamed successfully."}
    else:
        raise HTTPException(status_code=404, detail="Tag definition not found.")


@router.delete("/tags/delete_tag_definition")
def delete_tag_definition(request: Request, tag_id: int):
    # Immediately raise an error without any other implementation
    raise HTTPException(status_code=400, detail="Tag deletion is not supported.")


@router.get("/tags/list_tag_definition", response_class=PrettyJSONResponse)
def list_tag_definitions(request: Request):
    # Query all the tag definitions
    tags_cursor = request.app.tag_definitions_collection.find({})
    result = []

    for tag in tags_cursor:
        tag_data = {
            "tag_id": tag["tag_id"],
            "tag_string": tag["tag_string"],
            "tag_category": tag["tag_category"],
            "tag_vector_index": tag.get("tag_vector_index", -1),  # Use default value if tag_vector_index is absent
            "tag_description": tag["tag_description"],
            "user_who_created": tag["user_who_created"]
        }
        result.append(tag_data)

    return result


@router.put("/tags/set_tag_vector_index")
def set_tag_vector_index(request: Request, tag_id: int, vector_index: int):
    # Find the tag definition using the provided tag_id
    query = {"tag_id": tag_id}
    tag = request.app.tag_definitions_collection.find_one(query)

    if not tag:
        raise HTTPException(status_code=404, detail="Tag definition not found.")

    # Check if any other tag has the same vector index
    existing_tag = request.app.tag_definitions_collection.find_one({"tag_vector_index": vector_index})
    if existing_tag and existing_tag["tag_id"] != tag_id:
        raise HTTPException(status_code=400, detail="Another tag already has the same vector index.")

    # Update the tag vector index
    update_query = {"$set": {"tag_vector_index": vector_index}}
    request.app.tag_definitions_collection.update_one(query, update_query)

    return {"status": "success", "message": "Tag vector index updated successfully."}


@router.get("/tags/get_tag_vector_index")
def get_tag_vector_index(request: Request, tag_id: int):
    # Find the tag definition using the provided tag_id
    query = {"tag_id": tag_id}
    tag = request.app.tag_definitions_collection.find_one(query)

    if not tag:
        raise HTTPException(status_code=404, detail="Tag definition not found.")

    vector_index = tag.get("tag_vector_index", -1)
    return {"tag_vector_index": vector_index}




@router.post("/tags/add_tag_to_image", response_model=ImageTag)
def add_tag_to_image(request: Request, image_tag: ImageTag):
    date_now = datetime.now().isoformat()
    
    # Check if the tag exists by tag_id in the tag_definitions_collection
    existing_tag = request.app.tag_definitions_collection.find_one({"tag_id": image_tag.tag_id})
    if not existing_tag:
        raise HTTPException(status_code=404, detail="Tag does not exist!")

    # Create association between image and tag in the image_tags_collection
    image_tag_data = {
        "tag_id": existing_tag["tag_id"],
        "image_id": image_tag.image_id,
        "image_hash": image_tag.image_hash,
        "user_who_created": existing_tag["user_who_created"],
        "creation_time": date_now
    }

    request.app.image_tags_collection.insert_one(image_tag_data)
    return image_tag_data


@router.delete("/tags/remove_tag_from_image")
def remove_image_tag(request: Request, image_hash: str):
    query = {
        "image_hash": image_hash
    }
    
    result = request.app.image_tags_collection.delete_many(query)  
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Tags for the given image hash not found!")
    
    return {"status": "success", "message": f"tags removed successfully."}


@router.get("/tags/get_tag_list_for_image", response_model=List[TagDefinition])
def get_tag_list_for_image(request: Request, image_hash: str):
    # Fetch image tags based on image_hash
    image_tags_cursor = request.app.image_tags_collection.find({"image_hash": image_hash})
    
    tag_ids = [tag_data["tag_id"] for tag_data in image_tags_cursor]
    
    # Fetch the actual TagDefinition using tag_ids
    tags_cursor = request.app.tag_definitions_collection.find({"tag_id": {"$in": tag_ids}})
    tags_list = [TagDefinition(**tag_data) for tag_data in tags_cursor]
    
    return tags_list

@router.get("/tags/get_images_by_tag", response_model=List[ImageTag])
def get_tagged_images(request: Request, tag_id: int):
    # Fetch image details for this tag
    image_tags_cursor = request.app.image_tags_collection.find({"tag_id": tag_id})
    
    image_info_list = [
        ImageTag(
            tag_id=int(tag_data["tag_id"]),
            image_hash=str(tag_data["image_hash"]),
            image_id=tag_data["image_id"],
            user_who_created=tag_data["user_who_created"],
            creation_time=tag_data.get("creation_time", None)  # Use get() method with default None if not found
        ) 
        for tag_data in image_tags_cursor 
        if tag_data.get("image_hash") and tag_data.get("image_id") and tag_data.get("user_who_created")
    ]
    
    # If no image details found, raise an exception
    if not image_info_list:
        raise HTTPException(status_code=404, detail="No images found for the given tag!")
    
    return image_info_list


@router.get("/tags/get_all_tagged_images", response_model=List[ImageTag])
def get_all_tagged_images(request: Request):
    # Fetch all tagged image details
    image_tags_cursor = request.app.image_tags_collection.find({})

    image_info_list = [
        ImageTag(
            tag_id=int(tag_data["tag_id"]),
            image_hash=str(tag_data["image_hash"]),
            image_id=tag_data["image_id"],
            user_who_created=tag_data["user_who_created"],
            creation_time=tag_data.get("creation_time", None)  # Use get() method with default None if not found
        ) 
        for tag_data in image_tags_cursor 
        if tag_data.get("image_hash") and tag_data.get("image_id") and tag_data.get("user_who_created")
    ]

    # If no tagged image details found, raise an exception
    if not image_info_list:
        raise HTTPException(status_code=404, detail="No tagged images found!")

    return image_info_list
