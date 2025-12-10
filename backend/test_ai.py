import face_recognition

# 1. Load your "Training" image (The reference photo)
print("Loading known face...")
# MAKE SURE YOU HAVE 'rudransh.jpg' IN THIS FOLDER
known_image = face_recognition.load_image_file("rudransh.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# 2. Load the "Unknown" image to test
print("Loading unknown face to check...")
# MAKE SURE YOU HAVE 'test_image.jpg' IN THIS FOLDER
unknown_image = face_recognition.load_image_file("test_image.jpg")

# 3. Find all faces in the unknown image
try:
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
except IndexError:
    print("Error: No face found in the test image!")
    exit()

# 4. Compare faces
print("Comparing...")
results = face_recognition.compare_faces([known_encoding], unknown_encoding)

if results[0] == True:
    print("✅ MATCH FOUND: It is Rudransh!")
else:
    print("❌ NO MATCH: Unknown Person.")