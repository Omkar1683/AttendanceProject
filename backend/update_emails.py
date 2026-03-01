"""
Simple script to update user emails in the database
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
    
    # Update teacher email
    result1 = users_col.update_one(
        {'email': 'prof.omkar@vesit.edu'},
        {'$set': {'email': 'prof.XYZ@ves.ac.in', 'name': 'Prof. Xyz'}}
    )
    print(f"✅ Updated teacher email: {result1.modified_count} document(s)")
    
    # Update student emails
    result2 = users_col.update_one(
        {'email': 'rudransh@student.edu'},
        {'$set': {'email': 'rudransh@ves.ac.in'}}
    )
    print(f"✅ Updated student 1 email: {result2.modified_count} document(s)")
    
    result3 = users_col.update_one(
        {'email': 'omkar.student@vesit.edu'},
        {'$set': {'email': 'omkar@ves.ac.in'}}
    )
    print(f"✅ Updated student 2 email: {result3.modified_count} document(s)")
    
    print("\n📝 New Login Credentials:")
    print("   Teacher: prof.XYZ@ves.ac.in / teacher123")
    print("   Student: rudransh@ves.ac.in / student123")
    
except Exception as e:
    print(f"❌ Error: {e}")
