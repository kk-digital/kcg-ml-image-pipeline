from datetime import datetime
from fastapi import APIRouter, Request
from typing import List

from orchestration.api.mongo_schema.ranking_models_schemas import RankingModel, RequestRanking_model, ListRankingModels
from .api_utils import ErrorCode, StandardSuccessResponseV1, ApiResponseHandlerV1

router = APIRouter()


def get_next_ranking_model_id_sequence(request: Request):
    # get ranking model counter
    counter = request.app.counters_collection.find_one({"_id": "ranking_models"})
    # create counter if it doesn't exist already
    if counter is None:
        request.app.counters_collection.insert_one({"_id": "ranking_models", "seq":0})
    counter_seq = counter["seq"] if counter else 0 
    counter_seq += 1

    try:
        ret = request.app.counters_collection.update_one(
            {"_id": "ranking_models"},
            {"$set": {"seq": counter_seq}})
    except Exception as e:
        raise Exception("Updating of ranking model counter failed: {}".format(e))

    return counter_seq

@router.post("/ranking-models/register-ranking-model", 
             tags=["Ranking models"],
             description="Adds a new ranking model or updates an existing one with the same type and rank ID",
             response_model=StandardSuccessResponseV1[RankingModel],
             responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
async def create_ranking_model(request: Request, ranking_model_data: RequestRanking_model):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        # Verify rank_id exists in rank_models_collection
        rank_id = ranking_model_data.rank_id

        if not request.app.rank_model_models_collection.find_one({"rank_model_id": rank_id}):
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS, 
                error_string=f"Rank ID {rank_id} not found in tag definitions.",
                http_status_code=400
            )

        # Check for an existing raning model with the same name and rank ID
        existing_ranking_model = request.app.ranking_models_collection.find_one({
            "model_type": ranking_model_data.model_type,
            "rank_id": ranking_model_data.rank_id
        })

        if existing_ranking_model:
            # Update existing ranking model
            update_fields = {
                "latest_model_creation_time": datetime.now().isoformat(),
                "model_path": ranking_model_data.model_path
            }
            request.app.ranking_models_collection.update_one(
                {"ranking_model_id": existing_ranking_model["ranking_model_id"]},
                {"$set": update_fields}
            )

            # Fetch updated ranking model data
            updated_ranking_model = request.app.ranking_models_collection.find_one({
                "ranking_model_id": existing_ranking_model["ranking_model_id"]
            })

            updated_ranking_model.pop('_id', None)

            return response_handler.create_success_response_v1(
                response_data=updated_ranking_model,
                http_status_code=200
            )
        else:
            # Calculate internal fields for new ranking model
            new_ranking_model_id = get_next_ranking_model_id_sequence(request)
            creation_time = datetime.now().isoformat()

            # Merge user-provided data with calculated values
            full_ranking_model_data = RankingModel(
                ranking_model_id=new_ranking_model_id,
                model_name=ranking_model_data.model_name,
                model_type= ranking_model_data.model_type,
                rank_id=ranking_model_data.rank_id,
                model_path=ranking_model_data.model_path,
                latest_model_creation_time= creation_time,
                creation_time=creation_time
            )

            # Insert new ranking model into the collection
            request.app.ranking_models_collection.insert_one(full_ranking_model_data.dict())

            return response_handler.create_success_response_v1(
                response_data=full_ranking_model_data.dict(),
                http_status_code=201
            )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=f"Failed to add or update ranking model: {str(e)}",
            http_status_code=500
        )

    
@router.get("/ranking-models/list-ranking-models", 
            response_model=StandardSuccessResponseV1[ListRankingModels],
            description="list ranking models",
            tags=["Ranking models"],
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([422, 500]))
async def list_ranking_models(request: Request):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        ranking_model_cursor = request.app.ranking_models_collection.find({})
        ranking_model_list = list(ranking_model_cursor)

        result = []
        for ranking_model in ranking_model_list:
            # Convert MongoDB's ObjectId to string if needed, otherwise prepare as is
            ranking_model['_id'] = str(ranking_model['_id'])
            ranking_model.pop('_id', None)
            result.append(ranking_model)

        return response_handler.create_success_response_v1(response_data={"ranking_models": result}, http_status_code=200)

    except Exception as e:
        # Implement appropriate error handling
        print(f"Error: {str(e)}")
        return response_handler.create_error_response_v1(error_code=ErrorCode.OTHER_ERROR, error_string="Internal server error", http_status_code=500)