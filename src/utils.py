# utils.py
from typing import get_args, Dict, Any
from inspect import isclass
import models
import dataclasses

# Grab the Union[...] in Event.data
_data_union = models.Event.__annotations__['data']
# Extract the concrete classes (skip 'str')
_DATA_CLASSES = {
    cls.__name__: cls
    for cls in get_args(_data_union)
    if isclass(cls) and dataclasses.is_dataclass(cls)
}

def deserialize_event(raw: Dict[str, Any]) -> models.Event:
    """
    Deserialize a raw event dictionary into an Event object.
    Handles both data classes and raw strings.
    """

    event_type   = raw["type"]
    data_payload = raw["data"]

    camel = "".join(w.title() for w in event_type.split("_"))
    cls   = _DATA_CLASSES.get(camel)
    data_obj = cls(**data_payload) if cls else data_payload

    return models.Event(type=event_type, data=data_obj)
