from typing import Type, Any
from pydantic import BaseModel, create_model
from pydantic.json_schema import model_json_schema


def get_schema(cls: Type[Any]) -> dict:
    """Create a JSON schema dict from a dataclass or Pydantic model.

    Args:
        cls: A dataclass or Pydantic model type.

    Returns:
        A dict representing the JSON schema of the input class.

    Raises:
        TypeError: If the input is not a dataclass or Pydantic model.
    """
    if isinstance(cls, type) and issubclass(cls, BaseModel):
        return model_json_schema(cls)
    elif hasattr(cls, "__dataclass_fields__"):
        # Convert dataclass to Pydantic model
        fields = {
            field_name: (field.type, ...)
            for field_name, field in cls.__dataclass_fields__.items()
        }
        pydantic_model = create_model(cls.__name__, **fields)
        return model_json_schema(pydantic_model)
    else:
        raise TypeError("Input must be a dataclass or Pydantic model")
