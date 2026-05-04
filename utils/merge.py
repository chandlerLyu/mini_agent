"""Recursive dictionary merge."""

from __future__ import annotations

from copy import deepcopy


def recursive_merge(*items):
    result = {}
    for item in items:
        for key, value in item.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = recursive_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
    return result
