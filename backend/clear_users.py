"""
Script to clean and re-seed user data with new emails
"""
from pymongo import MongoClient
from urllib.parse import quote_plus

# Configuration
RAW_USERNAME = "omramjadhav9"
RAW_PASSWORD = "Jadhav@1683"
CLUSTER_ADDRESS = "cluster0.wrixxci.mongodb.net"

username = quote_plus(RAW_USERNAME)
password = quote_plus(RAW_PASSWORD)
MONGO_URI = f"mongodb+srv://{username}:{password}@{CLUSTER_ADDRESS}/?retryWrites=true&w=majority&appName=Cluster0"

try:
    client = MongoClient(MONGO_URI)
    db = client['attendai_db']
    users_col = db['users']
    
    # Delete all old users to start fresh
    result = users_col.delete_many({})
    print(f"🗑️  Deleted {result.deleted_count} old user(s)")
    
    print("\n✅ Database cleared. Now run: python db_utils.py to re-seed")
    print("\n📝 Login Credentials will be:")
    print("   Teacher: prof.XYZ@ves.ac.in / teacher123")
    print("   Student: rudransh@ves.ac.in / student123")
    
except Exception as e:
    print(f"❌ Error: {e}")
