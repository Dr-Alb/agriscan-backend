from flask import Flask, request, jsonify, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
import numpy as np
import os, io
from PIL import Image
import datetime
import openai
from flask import render_template
from twilio.rest import Client
import requests
from dotenv import load_dotenv
from flask import Flask, send_from_directory

load_dotenv()
app = Flask(__name__,static_folder='static')
app.secret_key = 'your-secret-key'
CORS(app, supports_credentials=True)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agriscan.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

interpreter = tf.lite.Interpreter(model_path="plant_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = [ 
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
 ]  

disease_info = {
    "Apple___Apple_scab": {
        "description": "Apple scab is a fungal disease that causes dark, scabby lesions on leaves and fruit.",
        "treatment": "Apply fungicides during the early growing season and remove fallen leaves."
    },
    "Apple___Black_rot": {
        "description": "Black rot affects apples, causing cankers on limbs and rotting of fruit.",
        "treatment": "Prune affected branches, remove mummified fruits, and apply fungicide sprays."
    },
    "Apple___Cedar_apple_rust": {
        "description": "Cedar apple rust is a fungal disease requiring both apple and cedar trees to complete its lifecycle.",
        "treatment": "Remove nearby cedar trees and apply fungicides in early spring."
    },
    "Apple___healthy": {
        "description": "The plant is healthy.",
        "treatment": "No action needed."
    },
    "Blueberry___healthy": {
        "description": "The plant is healthy.",
        "treatment": "No action needed."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "description": "Powdery mildew causes a white, powdery growth on leaves, reducing fruit quality.",
        "treatment": "Use sulfur-based fungicides and prune for better air circulation."
    },
    "Cherry_(including_sour)___healthy": {
        "description": "The plant is healthy.",
        "treatment": "No action needed."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "description": "A fungal disease causing grayish lesions on leaves, reducing photosynthesis.",
        "treatment": "Rotate crops, use resistant varieties, and apply fungicides."
    },
    "Corn_(maize)___Common_rust_": {
        "description": "Common rust appears as reddish-brown pustules on leaves.",
        "treatment": "Use resistant hybrids and fungicides if infection is severe."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "description": "Causes elongated gray-green lesions, leading to yield loss.",
        "treatment": "Use resistant varieties and fungicides."
    },
    "Corn_(maize)___healthy": {
        "description": "The plant is healthy.",
        "treatment": "No action needed."
    },
    "Grape___Black_rot": {
        "description": "Black rot causes black spots on leaves and rots the fruit.",
        "treatment": "Remove infected parts and apply fungicides regularly."
    },
    "Grape___Esca_(Black_Measles)": {
        "description": "A fungal disease that leads to leaf scorch and internal wood rot.",
        "treatment": "Prune infected canes and avoid excessive vine stress."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "description": "Small dark spots on leaves that merge into larger blighted areas.",
        "treatment": "Remove infected leaves and apply protective fungicides."
    },
    "Grape___healthy": {
        "description": "The plant is healthy.",
        "treatment": "No action needed."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "description": "A deadly bacterial disease spread by psyllids, causing yellow shoots and bitter fruit.",
        "treatment": "No cure. Control insect vector and remove infected trees."
    },
    "Peach___Bacterial_spot": {
        "description": "Causes dark lesions on leaves and sunken spots on fruits.",
        "treatment": "Use copper-based sprays and disease-resistant varieties."
    },
    "Peach___healthy": {
        "description": "The plant is healthy.",
        "treatment": "No action needed."
    },
    "Pepper,_bell___Bacterial_spot": {
        "description": "Bacterial infection that creates dark water-soaked spots on leaves and fruits.",
        "treatment": "Use certified seed and copper-based bactericides."
    },
    "Pepper,_bell___healthy": {
        "description": "The plant is healthy.",
        "treatment": "No action needed."
    },
    "Potato___Early_blight": {
        "description": "Dark concentric spots appear on leaves, leading to defoliation.",
        "treatment": "Use fungicides and rotate crops."
    },
    "Potato___Late_blight": {
        "description": "Rapid browning and death of leaves; caused the Irish Potato Famine.",
        "treatment": "Apply systemic fungicides and destroy infected plants."
    },
    "Potato___healthy": {
        "description": "The plant is healthy.",
        "treatment": "No action needed."
    },
    "Raspberry___healthy": {
        "description": "The plant is healthy.",
        "treatment": "No action needed."
    },
    "Soybean___healthy": {
        "description": "The plant is healthy.",
        "treatment": "No action needed."
    },
    "Squash___Powdery_mildew": {
        "description": "White powdery fungus that reduces photosynthesis and yield.",
        "treatment": "Apply sulfur-based or neem oil sprays."
    },
    "Strawberry___Leaf_scorch": {
        "description": "Dark purple spots on leaves, causing them to wither and die.",
        "treatment": "Remove infected leaves and apply fungicides."
    },
    "Strawberry___healthy": {
        "description": "The plant is healthy.",
        "treatment": "No action needed."
    },
    "Tomato___Bacterial_spot": {
        "description": "Dark water-soaked lesions on leaves, stems, and fruits.",
        "treatment": "Use resistant seeds and copper sprays."
    },
    "Tomato___Early_blight": {
        "description": "Brown concentric rings on leaves, causing defoliation.",
        "treatment": "Use crop rotation and fungicides."
    },
    "Tomato___Late_blight": {
        "description": "Grayish spots on leaves and brown lesions on fruit.",
        "treatment": "Destroy infected plants and apply fungicides."
    },
    "Tomato___Leaf_Mold": {
        "description": "Yellowing and moldy growth on underside of leaves.",
        "treatment": "Increase airflow and apply fungicides."
    },
    "Tomato___Septoria_leaf_spot": {
        "description": "Small water-soaked circular spots that spread rapidly.",
        "treatment": "Remove infected leaves and use fungicide sprays."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "Tiny mites that feed on leaves, causing yellowing and webbing.",
        "treatment": "Spray with miticides or insecticidal soap."
    },
    "Tomato___Target_Spot": {
        "description": "Dark concentric spots on leaves and stems.",
        "treatment": "Improve air circulation and use fungicides."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "Leaves curl and turn yellow; plant stunts in growth.",
        "treatment": "Control whiteflies and remove infected plants."
    },
    "Tomato___Tomato_mosaic_virus": {
        "description": "Mosaic pattern on leaves and stunted growth.",
        "treatment": "Remove infected plants and disinfect tools."
    },
    "Tomato___healthy": {
        "description": "The plant is healthy.",
        "treatment": "No action needed."
    }
  }


class User(db.Model):
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Reminder(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    datetime = db.Column(db.DateTime, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224)).convert('RGB')
    img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
    return img_array

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, "index.html")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def show_login():
    return render_template('login.html')

@app.route('/signup')
def show_signup():
    return render_template('signup.html')

@app.route('/result')
def show_result():
    return render_template('result.html')


@app.route('/api/scan', methods=['POST'])
def scan():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        # Load and preprocess image
        image = Image.open(file.stream).convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_index = int(np.argmax(output_data))
        probability = float(np.max(output_data))
        prediction = class_names[predicted_index]

        response = {
            "prediction": prediction,
            "probability": probability,
            "description": disease_info[prediction]["description"],
            "treatment": disease_info[prediction]["treatment"]
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chatbot():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'reply': 'Please provide a message.'}), 400

    if "disease" in user_message.lower():
        reply = "If your plant has a disease, upload a clear photo on the Scan page to get instant results!"
    elif "weather" in user_message.lower():
        reply = "🌤 Weather alerts are sent via SMS daily if rain or sun might affect your crops."
    else:
        reply = "🤖 I'm AgriBot! Ask me about leaf diseases, treatments, reminders, or scanning tips."

    return jsonify({'reply': reply})

@app.route('/api/reminders', methods=['POST'])
def create_reminder():
    data = request.json
    title = data.get('title')
    datetime_str = data.get('datetime')
    user_email = session.get('user')

    if not title or not datetime_str:
        return jsonify({'error': 'Missing title or datetime'}), 400

    try:
        parsed_time = datetime.datetime.fromisoformat(datetime_str)
    except:
        return jsonify({'error': 'Invalid datetime format'}), 400

    user = User.query.filter_by(email=user_email).first() if user_email else None
    reminder = Reminder(title=title, datetime=parsed_time, user_id=user.id if user else None)
    db.session.add(reminder)
    db.session.commit()

    return jsonify({'message': 'Reminder saved to DB'})

@app.route('/api/reminders', methods=['GET'])
def get_reminders():
    user_email = session.get('user')
    user = User.query.filter_by(email=user_email).first() if user_email else None
    reminders = Reminder.query.filter_by(user_id=user.id if user else None).order_by(Reminder.datetime.asc()).all()
    return jsonify([
        {'id': r.id, 'title': r.title, 'datetime': r.datetime.isoformat()} for r in reminders
    ])

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already exists'}), 409

    hashed_pw = generate_password_hash(password)
    user = User(email=email, password=hashed_pw)
    db.session.add(user)
    db.session.commit()
    return jsonify({'message': 'User registered'})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password, password):
        return jsonify({'error': 'Invalid credentials'}), 401

    session['user'] = user.email
    return jsonify({'message': 'Logged in', 'email': user.email})

@app.route('/api/profile', methods=['GET'])
@jwt_required()
def get_profile():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify({
        "id": user.id,
        "email": user.email,
        "joined": user.created_at.strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/logout')
def logout():
    session.clear()
    return jsonify({'message': 'Logged out'})

@app.route('/save_report', methods=['POST'])
@jwt_required()
def save_report():
    user_id = get_jwt_identity()
    data = request.get_json()
    new_report = Report(
        user_id=user_id,
        prediction=data.get('prediction'),
        probability=data.get('probability'),
        description=data.get('description'),
        treatment=data.get('treatment'),
    )
    db.session.add(new_report)
    db.session.commit()
    return jsonify({"msg": "Report saved"}), 200


@app.before_request
def create_tables_once():
    if not hasattr(app, 'db_initialized'):
        db.create_all()
        app.db_initialized = True

port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port)

if __name__ == '__main__':
    app.run(debug=True)
