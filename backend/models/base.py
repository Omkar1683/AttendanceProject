"""
models/base.py
--------------
BaseModel provides generic CRUD operations backed by a MongoDB collection.
All domain models inherit from this class.
"""
from typing import Optional, List, Dict, Any
from bson import ObjectId


class BaseModel:
    """Generic MongoDB model with common CRUD helpers."""

    def __init__(self, db, collection_name: str):
        self.db = db
        self.collection = db[collection_name]
        self.collection_name = collection_name

    # ── Read ──────────────────────────────────────────────────────────────────

    def find_by_id(self, doc_id: str) -> Optional[Dict]:
        """Find a document by its ObjectId string."""
        try:
            return self.collection.find_one({'_id': ObjectId(doc_id)})
        except Exception as exc:
            print(f"[{self.collection_name}] find_by_id error: {exc}")
            return None

    def find_one(self, query: Dict) -> Optional[Dict]:
        """Find the first document matching *query*."""
        return self.collection.find_one(query)

    def find_many(
        self,
        query: Dict = {},
        skip: int = 0,
        limit: int = 0,
        projection: Optional[Dict] = None,
        sort: Optional[List] = None,
    ) -> List[Dict]:
        """Find all documents matching *query*, with optional pagination."""
        cursor = self.collection.find(query, projection)
        if sort:
            cursor = cursor.sort(sort)
        if skip > 0:
            cursor = cursor.skip(skip)
        if limit > 0:
            cursor = cursor.limit(limit)
        return list(cursor)

    # ── Write ─────────────────────────────────────────────────────────────────

    def insert_one(self, document: Dict) -> Optional[str]:
        """Insert a document and return its string ID."""
        try:
            result = self.collection.insert_one(document)
            return str(result.inserted_id)
        except Exception as exc:
            print(f"[{self.collection_name}] insert_one error: {exc}")
            return None

    def update_one(self, query: Dict, update: Dict) -> bool:
        """Update the first document matching *query*. Returns True if modified."""
        try:
            result = self.collection.update_one(query, update)
            return result.modified_count > 0
        except Exception as exc:
            print(f"[{self.collection_name}] update_one error: {exc}")
            return False

    def delete_one(self, query: Dict) -> bool:
        """Delete the first document matching *query*. Returns True if deleted."""
        try:
            result = self.collection.delete_one(query)
            return result.deleted_count > 0
        except Exception as exc:
            print(f"[{self.collection_name}] delete_one error: {exc}")
            return False

    # ── Aggregate ─────────────────────────────────────────────────────────────

    def count(self, query: Dict = {}) -> int:
        """Count documents matching *query*."""
        return self.collection.count_documents(query)
