import sys
from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection

from src.configuration.configuration import ConfigurationManager
from src.exception import CustomException
from src.logger import logger


class MongoDBClient:
    _client: Optional[MongoClient] = None

    def __init__(self):

        try:
            cfg = ConfigurationManager.get_mongo_config()

            if not cfg.uri:
                raise ValueError(
                    "MongoDB URI is missing in environment variables."
                )

            self._db_name = cfg.db_name
            self._collection_name = cfg.collection

            if MongoDBClient._client is None:

                logger.info("Connecting to MongoDB Atlas")

                MongoDBClient._client = MongoClient(
                    cfg.uri,
                    serverSelectionTimeoutMS=5000,
                )

                MongoDBClient._client.admin.command("ping")

                logger.info("MongoDB connection established")

            self.client = MongoDBClient._client
            self.db = self.client[self._db_name]

        except Exception as e:
            raise CustomException(e, sys)

    def get_collection(self) -> Collection:
        return self.db[self._collection_name]