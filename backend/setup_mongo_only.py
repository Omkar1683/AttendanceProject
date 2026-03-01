import face_recognition
import numpy as np
import os
import base64
from pymongo import MongoClient
from urllib.parse import quote_plus

# --- ⚠️ CONFIGURATION (UPDATED) ⚠️ ---
# 1. Your Credentials
RAW_USERNAME = "omramjadhav9"
RAW_PASSWORD = "Jadhav@1683"  # The '@' here is now handled safely

# 2. Your Cluster Address
CLUSTER_ADDRESS = "cluster0.wrixxci.mongodb.net"

# Auto-generate the safe URI
username = quote_plus(RAW_USERNAME)
password = quote_plus(RAW_PASSWORD)
MONGO_URI = f"mongodb+srv://{username}:{password}@{CLUSTER_ADDRESS}/?retryWrites=true&w=majority&appName=Cluster0"

# --- STUDENT DATA ---
students_data = [
    ("rudransh.jpg", "Rudransh Gupta", "MCA-13"),
    ("omkar.jpg", "Omkar Jadhav", "MCA-14"),
    ("pushkar.jpg", "Pushkar Jaju", "MCA-15"),
    ("Devesh.jpg", "Devesh Mahajan", "MCA-16")
]

def init_db():
    print("🚀 Starting MongoDB-Only Setup...")
    print(f"🔌 Connecting to: {CLUSTER_ADDRESS}...")

    try:
        client = MongoClient(MONGO_URI)
        db = client['attendai_db'] 
        students_col = db['students']
        # Test connection
        client.admin.command('ping')
        print("✅ Connected to MongoDB successfully!")
    except Exception as e:
        print(f"❌ MongoDB Connection Error: {e}")
        return

    print("\n📸 Processing Images...")
    
    count = 0
    for filename, name, roll in students_data:
        if os.path.exists(filename):
            try:
                # 1. Convert Image to Base64 String
                with open(filename, "rb") as image_file:
                    image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

                # 2. Generate Face Encoding
                image = face_recognition.load_image_file(filename)
                encodings = face_recognition.face_encodings(image)

                if len(encodings) > 0:
                    encoding = encodings[0]
                    
                    # 3. Save to MongoDB
                    student_doc = {
                        "name": name,
                        "roll_no": roll,
                        "photo_base64": image_base64,
                        "encoding": encoding.tolist()
                    }
                    students_col.insert_one(student_doc)
                    print(f"   💾 Saved: {name}")
                    count += 1
                else:
                    print(f"   ⚠️ No face found in {filename}")

            except Exception as e:
                print(f"   ❌ Error processing {name}: {e}")
        else:
            print(f"   ⚠️ File not found: {filename}")

    print(f"\n🎉 Setup Complete! {count} students added to MongoDB.")

if __name__ == "__main__":
    init_db()