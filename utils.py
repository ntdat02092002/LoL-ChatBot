from pydantic import BaseModel
from typing import List, Any, Dict
from enum import Enum

def serialize_pydantic_model(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return {key: serialize_pydantic_model(value) for key, value in obj.__dict__.items()}
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, List):
        return [serialize_pydantic_model(item) for item in obj]
    elif isinstance(obj, Dict):
        return {key: serialize_pydantic_model(value) for key, value in obj.items()}
    else:
        return obj