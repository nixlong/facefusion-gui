from fastapi import APIRouter
from typing import Any, cast

from facefusion import state_manager
from facefusion.uis.http_server.models import ApiResponse, StateItemRequest, StateBatchRequest

state_router = APIRouter()


@state_router.get("/api/v1/state", response_model=ApiResponse)
async def get_all_state():
    try:
        state = state_manager.get_state()
        return ApiResponse(success=True, data=dict(state))
    except Exception as e:
        return ApiResponse(success=False, message=str(e))


@state_router.get("/api/v1/state/{key}", response_model=ApiResponse)
async def get_state_item(key: str):
    try:
        value = state_manager.get_item(key)
        type_name = type(value).__name__
        return ApiResponse(success=True, data={key: value, "type": type_name})
    except Exception as e:
        return ApiResponse(success=False, message=str(e))


@state_router.post("/api/v1/state/{key}", response_model=ApiResponse)
async def set_state_item(key: str, request: StateItemRequest):
    try:
        value = state_manager.get_item(key)
        if value is not None :
            value_type = type(value)
            state_manager.set_item(key, cast(value_type, request.value))
        else:
            state_manager.set_item(key, request.value)
        return ApiResponse(success=True, message="State updated")
    except Exception as e:
        return ApiResponse(success=False, message=str(e))


@state_router.post("/api/v1/state/batch", response_model=ApiResponse)
async def set_batch_state(request: StateBatchRequest):
    try:
        batch_data = request.model_dump(exclude_none=True)
        for key, value in batch_data.items():
            state_manager.set_item(key, value)
        return ApiResponse(success=True, message="Batch state updated")
    except Exception as e:
        return ApiResponse(success=False, message=str(e))
