import os
import sys
import pickle
from typing import Any, Dict, List

import yaml

from src.exception import CustomException
from src.logger import logger


def create_directories(paths: List[str]) -> None:
    """
    Create directories if they do not already exist.
    """
    try:
        for path in paths:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Directory created: {path}")

    except Exception as e:
        raise CustomException(e, sys) from e


def save_object(file_path: str, obj: Any) -> None:
    """
    Save any Python object using pickle.
    """
    try:
        os.makedirs(
            os.path.dirname(file_path),
            exist_ok=True,
        )

        with open(file_path, "wb") as file_obj:
            pickle.dump(
                obj,
                file_obj,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        logger.info(f"Object saved at {file_path}")

    except Exception as e:
        raise CustomException(e, sys) from e


def load_object(file_path: str) -> Any:
    """
    Load a pickled object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)

        logger.info(f"Object loaded from {file_path}")

        return obj

    except Exception as e:
        raise CustomException(e, sys) from e


def write_yaml_file(
    file_path: str,
    content: Dict,
) -> None:
    """
    Write dictionary to YAML file.
    """
    try:
        os.makedirs(
            os.path.dirname(file_path),
            exist_ok=True,
        )

        with open(
            file_path,
            "w",
            encoding="utf-8",
        ) as yaml_file:

            yaml.safe_dump(
                content,
                yaml_file,
                sort_keys=False,
            )

        logger.info(f"YAML written to {file_path}")

    except Exception as e:
        raise CustomException(e, sys) from e


def read_yaml_file(file_path: str) -> Dict:
    """
    Read YAML file.
    """
    try:
        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as yaml_file:

            content = yaml.safe_load(yaml_file)

        logger.info(f"YAML loaded from {file_path}")

        return content

    except Exception as e:
        raise CustomException(e, sys) from e