from dotenv import load_dotenv
load_dotenv()
import os
private_key = os.getenv("secret_key")
jwt_key = os.getenv("jwt_secret_key")
mongodb_public_url = os.getenv("mongodb_url")

from flask import Flask, request, jsonify, send_file
from sentence_transformers import SentenceTransformer
import faiss
import json
from io import BytesIO
from gtts import gTTS
from flask_pymongo import PyMongo
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required,
    get_jwt_identity, verify_jwt_in_request
)
import datetime
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# --- Configuration ---
app.config["secret_key"] = private_key
app.config["jwt_secret_key"] = jwt_key
# Use your Railway MongoDB URI (or any valid MongoDB connection string)
app.config["mongodb_url"] = mongodb_public_url

# Initialize extensions
mongo = PyMongo(app)
jwt = JWTManager(app)


# --- Bible Data and FAISS Setup ---
filePath = "genesis.json"

# Initialize your SentenceTransformer model.
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def load_bible_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


# Load the Bible data
bible_data = load_bible_data(filePath)

# Create embeddings for each record's "text" and convert them to float32.
texts = [record["text"] for record in bible_data]
embeddings = embedder.encode(texts, convert_to_numpy=True).astype("float32")

# Create a FAISS index for fast similarity search.
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)


def retrieve_bible_verse(question):
    """
    Given a question, encode it and search for the most similar Bible verse.
    Returns the matching record or None if no match is found.
    """
    query_embedding = embedder.encode([question], convert_to_numpy=True).astype("float32")
    _, indices = index.search(query_embedding, 1)
    if indices is not None and len(indices[0]) > 0 and indices[0][0] != -1:
        return bible_data[indices[0][0]]
    else:
        return None


def answer_question(question):
    """
    Retrieves the Bible verse record that best matches the query and formats the reply.
    """
    record = retrieve_bible_verse(question)
    if record:
        reply = f"{record['book']} {record['chapter']}:{record['verse']} - {record['text']}"
        return reply
    else:
        return "No relevant verse found."


@app.route('/ask', methods=['POST'])
def query():
    data = request.json
    question = data.get("question")
    response_type = data.get("response_type", "text")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    response = answer_question(question)

    if response_type == "voice":
        tts = gTTS(response, lang="en")
        audio_file = BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        return send_file(audio_file, mimetype="audio/mpeg", as_attachment=False, download_name="answer.mp3")
    else:
        return jsonify({"answer": response})


# --- User Registration & Login Endpoints ---
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    if mongo.db.usersDetails.find_one({"username": username}):
        return jsonify({"error": "Username already exists"}), 400

    hashed_password = generate_password_hash(password)
    mongo.db.usersDetails.insert_one({
        "username": username,
        "password": hashed_password
    })
    return jsonify({"message": "User registered successfully"}), 201


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    user = mongo.db.usersDetails.find_one({"username": username})
    if user and check_password_hash(user["password"], password):
        access_token = create_access_token(
            identity=str(user["_id"]),
            expires_delta=datetime.timedelta(hours=1)
        )
        return jsonify({
            "message": "Login successful",
            "access_token": access_token,
            "user_id": str(user["_id"]),
            "username": username
        }), 200

    return jsonify({"error": "Invalid credentials"}), 401


@app.route("/protected", methods=["GET"])
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    return jsonify({"message": f"Hello user {current_user}, you have access!"}), 200


if __name__ == '__main__':
    port = 5000  # or use os.environ.get('PORT', 5000)
    print(f"Server running on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=True)
