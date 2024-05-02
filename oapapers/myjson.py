# -*- coding: UTF-8 -*-
"""
Created on 15.05.23
As I have tried many libraries for JSON serialization and deserialization I have created this interface that allows
me to easily switch between them.

:author:     Martin Dočekal
"""
import json
from typing import Any

import orjson


def json_dumps(obj):
    """
    Serializes object to JSON string.

    :param obj: Object to serialize.
    :type obj: Any
    :return: JSON string.
    :rtype: str
    """

    return json.dumps(obj, separators=(',', ':'), ensure_ascii=False)


def json_loads(json_str: str) -> Any:
    """
    Deserializes object from JSON string.

    :param json_str: JSON string.
    :type json_str: str
    :return: Deserialized object.
    :rtype: Any
    """

    return orjson.loads(json_str)

