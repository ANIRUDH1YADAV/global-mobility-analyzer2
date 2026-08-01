import os

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

mongo_uri = os.getenv("MONGODB_URI") or os.getenv("MONGODB_URL")
db_name = os.getenv("DATABASE_NAME", "global_mobility")
collection_name = os.getenv("COLLECTION_NAME", "visa_data")

client = MongoClient(mongo_uri)

db = client[db_name]
collection = db[collection_name]

df = pd.read_csv("easy_visa_dataset(original)/EasyVisa.csv")

records = df.to_dict(orient="records")

if records:
	collection.insert_many(records)

print(f"{len(records)} documents inserted successfully into {db_name}.{collection_name}.")