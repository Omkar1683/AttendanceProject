"""
database/connection.py
----------------------
MongoDB connection singleton.
Call get_db() anywhere in the app to get the live database handle.
"""
from pymongo import MongoClient
from pymongo.database import Database

_client: MongoClient | None = None
_db: Database | None = None


def init_db(mongo_uri: str, db_name: str) -> Database | None:
    """
    Initialise the MongoDB connection.
    Called once by create_app() at startup.
    Returns the Database object, or None on failure.
    """
    global _client, _db
    try:
        _client = MongoClient(mongo_uri)
        _db = _client[db_name]
        # Quick connection test
        _client.admin.command('ping')
        print("✅ Connected to MongoDB Atlas")
        return _db
    except Exception as exc:
        print(f"❌ MongoDB connection failed: {exc}")
        return None


def get_db() -> Database | None:
    """Return the current database handle (None if not initialised)."""
    return _db
