import json
from dataclasses import fields
from typing import Any, Optional, Type

from pydantic import BaseModel, Field, create_model
from pydantic.json_schema import model_json_schema

from promptim.config import Config


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
        fields_dict = {}
        for field in fields(cls):
            field_info = {}
            if field.default is not field.default_factory:
                # Field has a default value or default factory
                field_info["default"] = field.default
            if field.metadata.get("description"):
                field_info["description"] = field.metadata["description"]

            if field_info:
                fields_dict[field.name] = (Optional[field.type], Field(**field_info))
            else:
                # Field is required
                fields_dict[field.name] = (field.type, ...)
        pydantic_model = create_model(cls.__name__, **fields_dict)
        return model_json_schema(pydantic_model)
    else:
        raise TypeError("Input must be a dataclass or Pydantic model")


config_schema = get_schema(Config)
config_schema["$schema"] = "http://json-schema.org/draft-07/schema#"

with open("config-schema.json", "w") as f:
    json.dump(config_schema, f, indent=2)

print("Schema generated and saved to config-schema.json")
