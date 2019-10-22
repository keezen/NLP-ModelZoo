# python: 3.6
# encoding: utf-8

import json
import os


def read_json(json_path):
    """Read json file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj, json_path):
    """Save object as json."""
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)