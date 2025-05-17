import os
import cv2
import dlib
import base64
from flask import jsonify
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from datetime import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog, local_binary_pattern
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Constants
N_IMAGES = 300
CONFIDENCE_THRESHOLD = 0.97
MIN_CONFIRMATIONS = 3
TIME_DIFF_MINUTES = 30
FACES_DIR = 'static/faces'
ATTENDANCE_DIR = 'Attendance'
MODEL_PATH = 'models/face_model.pkl'
USER_FILE = 'users.csv'
DATETODAY = datetime.today().strftime("%m_%d_%y")
ATTENDANCE_FILE = f'{ATTENDANCE_DIR}/Attendance-{DATETODAY}.csv'

# Dlib Models
PREDICTOR_PATH = 'models/shape_predictor_68_face_landmarks.dat'
FACE_RECOG_PATH = 'models/dlib_face_recognition_resnet_model_v1.dat'
HAAR_PATH = 'models/haarcascade_frontalface_default.xml'

# Global models dictionary
global_models = {
    'detector': dlib.get_frontal_face_detector(),
    'landmark_predictor': dlib.shape_predictor(PREDICTOR_PATH),
    'face_rec_model': dlib.face_recognition_model_v1(FACE_RECOG_PATH),
    'haar': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
}

print("\n✅ All models loaded successfully:")
print(" - Dlib Face Detector ✔")
print(" - Dlib Shape Predictor ✔")
print(" - Dlib Face Recognition ResNet Model ✔")
print(" - Haar Cascade Classifier ✔")

# Create folders if not exist
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# Init CSVs
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=['Name', 'EmployeeID', 'Phone', 'In-Time', 'Out-Time']).to_csv(ATTENDANCE_FILE, index=False)
if not os.path.exists(USER_FILE):
    pd.DataFrame(columns=['Name', 'EmployeeID', 'Phone']).to_csv(USER_FILE, index=False)

# Global variables for tracking confirmations
confirmation_tracker = {}

def preprocess(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0, (8,8)).apply(img)
    return cv2.GaussianBlur(clahe, (3,3), 0)

def extract_features(img):
    try:
        # Resize and convert to BGR if needed
        img = cv2.resize(img, (100, 100))
        if len(img.shape) == 2:
            bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            bgr_img = img

        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        
        # Try both face detection methods
        haar_faces = global_models['haar'].detectMultiScale(gray, 1.1, 5)
        dlib_faces = global_models['detector'](gray, 1)
        
        # If we found faces using either method
        if len(haar_faces) > 0 or len(dlib_faces) > 0:
            # Use Dlib if we found faces with it
            if len(dlib_faces) > 0:
                shape = global_models['landmark_predictor'](gray, dlib_faces[0])
                dlib_feat = np.array(global_models['face_rec_model'].compute_face_descriptor(bgr_img, shape))
            else:  # Otherwise use Haar
                x, y, w, h = haar_faces[0]
                face_roi = gray[y:y+h, x:x+w]
                shape = global_models['landmark_predictor'](face_roi, dlib.rectangle(0, 0, w, h))
                dlib_feat = np.array(global_models['face_rec_model'].compute_face_descriptor(bgr_img, shape))
        else:
            # If no faces detected, return zeros
            dlib_feat = np.zeros(128)

        # Extract HOG features
        hog_feat = hog(gray, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        
        # Extract LBP features
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
        lbp_feat = hist / (np.sum(hist) + 1e-7)

        # Combine all features
        features = np.concatenate([dlib_feat, hog_feat, lbp_feat])
        
        # Validate feature dimensions
        if features.size == 0 or np.isnan(features).any():
            print("Warning: Invalid features extracted")
            return None
            
        return features
        
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None

def process_image(user_path, img_name):
    img_path = os.path.join(user_path, img_name)
    try:
        img = cv2.imread(img_path)
        if img is not None:
            features = extract_features(preprocess(img))
            return features
    except Exception as e:
        print(f"Error processing image {img_name}: {str(e)}")
    return None

def train_model():
    # Check if faces directory exists
    if not os.path.exists(FACES_DIR):
        print(f"❌ Error: Faces directory '{FACES_DIR}' not found. Please create it and add face images.")
        return False

    # Get list of users from faces directory
    users = os.listdir(FACES_DIR)
    if len(users) < 2:
        print(f"❌ Error: Need at least 2 users for training. Found {len(users)} users:")
        for user in users:
            print(f" - {user}")
        print("\nPlease add more users and their face images to the faces directory.")
        return False

    faces, labels = [], []
    for user in users:
        user_path = os.path.join(FACES_DIR, user)
        if not os.path.isdir(user_path):
            continue

        # Get face images for this user
        images = [f for f in os.listdir(user_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            print(f"Warning: No face images found for user '{user}'")
            continue

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda img: process_image(user_path, img), images))
        for feat in results:
            if feat is not None:
                faces.append(feat)
                labels.append(user)

    if not faces:
        print("❌ Error: No valid face features extracted from images")
        return False

    X, y = np.array(faces), np.array(labels)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    joblib.dump({'model': model, 'scaler': scaler}, MODEL_PATH)
    print(f"\n✅ Training model with {len(faces)} samples from {len(set(labels))} users...")
    print("✅ Model trained and dumped to disk ✔")
    return True

def predict_face(img):
    if not os.path.exists(MODEL_PATH): return ("Unknown", 0)
    print(f"\n📦 Loading trained model from: {MODEL_PATH}")
    data = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully ✔")
    model, scaler = data['model'], data['scaler']
    features = extract_features(preprocess(img))
    scaled = scaler.transform(features.reshape(1, -1))
    pred = model.predict(scaled)[0]
    conf = model.predict_proba(scaled).max()
    print(f"Prediction: {pred} with confidence {conf:.2f}")
    return (pred, conf)

def mark_attendance(name):
    df = pd.read_csv(ATTENDANCE_FILE)
    username, userid = name.split('_')
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    today = now.strftime("%Y-%m-%d")

    # Check if there's any existing entry for this user
    if (df['EmployeeID'] == int(userid)).any():
        # Get all entries for this user
        user_entries = df[df['EmployeeID'] == int(userid)]

        # Check if there's already an entry for today
        has_today_entry = False
        last_entry = None
        for _, entry in user_entries.iterrows():
            in_date = datetime.strptime(entry['In-Time'], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
            if in_date == today:
                has_today_entry = True
                last_entry = entry
                break

        if has_today_entry:
            # Check if it's been at least 2 hours since In-Time
            in_time = datetime.strptime(last_entry['In-Time'], "%Y-%m-%d %H:%M:%S")
            time_diff = now - in_time
            
            if last_entry['Out-Time'] == '-':
                if time_diff.total_seconds() >= 7200:  # 2 hours in seconds
                    # Update Out-Time
                    df.loc[user_entries.index[-1], 'Out-Time'] = now_str
                    print(f"✅ Out-Time marked for {name} at {now_str}")
                else:
                    remaining_minutes = int((7200 - time_diff.total_seconds()) / 60)
                    print(f"⚠️ Cannot mark Out-Time yet. Please wait {remaining_minutes} more minutes.")
                    return False
            else:
                print(f"⚠️ Skipping duplicate entry for {name} - Already has both In-Time and Out-Time for today")
                return False
        else:
            # New In-Time entry for today
            df = pd.concat([df, pd.DataFrame([[username, userid, '', now_str, '-']], columns=df.columns)])
            print(f"✅ In-Time marked for {name} at {now_str}")
    else:
        # First entry for this user
        df = pd.concat([df, pd.DataFrame([[username, userid, '', now_str, '-']], columns=df.columns)])
        print(f"✅ In-Time marked for {name} at {now_str}")

    df.to_csv(ATTENDANCE_FILE, index=False)
    return True

def initialize_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap if cap.isOpened() else None

@app.route('/')
def home():
    df = pd.read_csv(ATTENDANCE_FILE)
    return render_template('home.html', names=df['Name'], rolls=df['EmployeeID'],
                           in_times=df['In-Time'], out_times=df['Out-Time'],
                           l=len(df), totalreg=len(os.listdir(FACES_DIR)),
                           datetoday2=datetime.today().strftime("%d-%B-%Y"))

@app.route('/verify_face', methods=['POST'])
def verify_face():
    try:
        if not request.json or 'image' not in request.json:
            return jsonify({'success': False, 'error': 'No image data received'}), 400

        # Get base64 image from request
        image_data = request.json['image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False, 'error': 'Invalid image data'}), 400

        # Process frame
        gray = preprocess(frame)
        
        # Try both Haar cascade and Dlib face detection
        haar_faces = global_models['haar'].detectMultiScale(gray, 1.1, 5)
        dlib_faces = global_models['detector'](gray, 1)
        
        # Combine results
        faces = []
        if len(haar_faces) > 0:
            faces.extend(haar_faces)
        if len(dlib_faces) > 0:
            faces.extend([(d.left(), d.top(), d.width(), d.height()) for d in dlib_faces])
        
        if len(faces) == 0:
            return jsonify({'success': False, 'error': 'No face detected in image. Please ensure your face is clearly visible'}), 404
            
        # Use first detected face
        x, y, w, h = faces[0]
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            return jsonify({'success': False, 'error': 'Face detection coordinates out of bounds'}), 400

        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0 or face_img.shape[0] == 0 or face_img.shape[1] == 0:
            return jsonify({'success': False, 'error': 'Invalid face crop'}), 400

        if not os.path.exists(MODEL_PATH):
            return jsonify({'success': False, 'error': 'Face recognition model not found. Please train the model first'}), 404

        name, conf = predict_face(face_img)
        
        if conf > CONFIDENCE_THRESHOLD:
            try:
                # Extract name and ID
                username, userid = name.split('_')
                now = datetime.now()
                now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                
                # Mark local attendance
                mark_attendance(name)
                
                # Post to attendance API
                from attendance_api import post_attendance_swipe
                post_attendance_swipe(username, userid, now_str, "In-Time")
                
                return jsonify({'success': True, 'name': username, 'confidence': float(conf)}), 200
            except Exception as e:
                print(f"Error marking attendance: {str(e)}")
                return jsonify({'success': False, 'error': 'Failed to mark attendance'}), 500
        
        return jsonify({'success': False, 'error': f'Face not recognized with sufficient confidence (got {conf:.2f}, need >{CONFIDENCE_THRESHOLD})'}), 400
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({'success': False, 'error': f'Processing error: {str(e)}'}), 500

@app.route('/start')
def start():
    return render_template('camera.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        if not request.json or 'image' not in request.json:
            return jsonify({'success': False, 'error': 'No image data received'}), 400

        # Get base64 image from request
        image_data = request.json['image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False, 'error': 'Invalid image data'}), 400

        # Process frame
        gray = preprocess(frame)
        
        # Try both Haar cascade and Dlib face detection
        haar_faces = global_models['haar'].detectMultiScale(gray, 1.1, 5)
        dlib_faces = global_models['detector'](gray, 1)
        
        # Combine results
        faces = []
        if len(haar_faces) > 0:
            faces.extend(haar_faces)
        if len(dlib_faces) > 0:
            faces.extend([(d.left(), d.top(), d.width(), d.height()) for d in dlib_faces])
        
        if len(faces) == 0:
            return jsonify({'success': False, 'error': 'No face detected in image. Please ensure your face is clearly visible'}), 404
            
        # Use first detected face
        x, y, w, h = faces[0]
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            return jsonify({'success': False, 'error': 'Face detection coordinates out of bounds'}), 400

        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0 or face_img.shape[0] == 0 or face_img.shape[1] == 0:
            return jsonify({'success': False, 'error': 'Invalid face crop'}), 400

        if not os.path.exists(MODEL_PATH):
            return jsonify({'success': False, 'error': 'Face recognition model not found. Please train the model first'}), 404

        name, conf = predict_face(face_img)
        
        if conf > CONFIDENCE_THRESHOLD:
            try:
                # Extract name and ID
                username, userid = name.split('_')
                
                # Update confirmation tracker
                if name not in confirmation_tracker:
                    confirmation_tracker[name] = {
                        'count': 0,
                        'last_time': None,
                        'confidences': []
                    }
                
                now = datetime.now()
                tracker = confirmation_tracker[name]
                
                # Reset if more than TIME_DIFF_MINUTES have passed
                if tracker['last_time'] and (now - tracker['last_time']).total_seconds() > TIME_DIFF_MINUTES * 60:
                    tracker['count'] = 0
                    tracker['confidences'] = []
                
                # Update tracker
                tracker['count'] += 1
                tracker['last_time'] = now
                tracker['confidences'].append(float(conf))
                
                # Calculate average confidence
                avg_confidence = sum(tracker['confidences']) / len(tracker['confidences'])
                
                if tracker['count'] >= MIN_CONFIRMATIONS:
                    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Mark local attendance
                    if mark_attendance(name):
                        # Determine swipe type based on existing entries
                        df = pd.read_csv(ATTENDANCE_FILE)
                        user_entries = df[df['EmployeeID'] == int(userid)]
                        swipe_type = "Out-Time" if user_entries.iloc[-1]['Out-Time'] != '-' else "In-Time"
                        
                        # Post to attendance API
                        from attendance_api import post_attendance_swipe
                        api_response = post_attendance_swipe(username, userid, now_str, swipe_type)
                        
                        # Consider both 200 and 201 as success
                        if api_response and api_response.status_code in [200, 201]:
                            print("\n✅ Attendance API Response:")
                            print(f"Status Code: {api_response.status_code}")
                            print(f"Response: {api_response.json()}")
                            
                            # Reset confirmation tracker for this user
                            confirmation_tracker[name] = {
                                'count': 0,
                                'last_time': None,
                                'confidences': []
                            }
                            
                            return jsonify({
                                'success': True,
                                'name': username,
                                'id': userid,
                                'confidence': float(avg_confidence),
                                'message': f'{swipe_type} marked successfully for {username} (ID: {userid})'
                            }), 200
                        else:
                            print("\n⚠️ Attendance API Error:")
                            print(f"Status Code: {api_response.status_code if api_response else 'No response'}")
                            print(f"Response: {api_response.text if api_response else 'No response'}")
                            return jsonify({
                                'success': False,
                                'error': 'Failed to sync with attendance API'
                            }), 500
                    else:
                        return jsonify({
                            'success': False,
                            'error': 'Cannot mark attendance - Please wait 2 hours between In-Time and Out-Time'
                        }), 400
                else:
                    return jsonify({
                        'success': True,
                        'name': username,
                        'id': userid,
                        'confidence': float(avg_confidence),
                        'message': f'Confirmation {tracker["count"]}/{MIN_CONFIRMATIONS} for {username} (ID: {userid})'
                    }), 200
                    
            except Exception as e:
                print(f"\n❌ Error marking attendance: {str(e)}")
                return jsonify({'success': False, 'error': 'Failed to mark attendance'}), 500
        
        return jsonify({'success': False, 'error': f'Face not recognized with sufficient confidence (got {conf:.2f}, need >{CONFIDENCE_THRESHOLD})'}), 400
        
    except Exception as e:
        print(f"\n❌ Error processing image: {str(e)}")
        return jsonify({'success': False, 'error': f'Processing error: {str(e)}'}), 500

@app.route('/add', methods=['POST'])
def add():
    try:
        name = request.form['newusername']
        uid = request.form['newuserid']
        phone = request.form['newuserphone']
        
        # Validate inputs
        if not name or not uid or not phone:
            return jsonify({'error': 'All fields are required'}), 400
            
        # Create user folder
        folder = os.path.join(FACES_DIR, f"{name}_{uid}")
        os.makedirs(folder, exist_ok=True)
        
        # Initialize camera
        cap = initialize_camera()
        if cap is None:
            return jsonify({'error': 'Could not access camera'}), 500
            
        count = 0
        success = False
        
        while count < N_IMAGES:
            ret, frame = cap.read()
            if not ret: continue
            gray = preprocess(frame)
            
            # Try both Haar cascade and Dlib face detection
            faces = global_models['haar'].detectMultiScale(gray, 1.1, 5)
            if len(faces) == 0:
                dets = global_models['detector'](gray, 1)
                faces = [(d.left(), d.top(), d.width(), d.height()) for d in dets]
            
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                path = os.path.join(folder, f"{count}.jpg")
                if cv2.imwrite(path, face_img):
                    count += 1
                    success = True
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(frame, f"{count}/{N_IMAGES}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            cv2.imshow('Register', frame)
            if cv2.waitKey(1) == 27: break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Only save user info and train model if we successfully captured images
        if success:
            # Read existing users and append new user
            try:
                df = pd.read_csv(USER_FILE)
                new_user = pd.DataFrame([[name, uid, phone]], columns=['Name', 'EmployeeID', 'Phone'])
                df = pd.concat([df, new_user], ignore_index=True)
                df.to_csv(USER_FILE, index=False)
                
                # Train model only if we have enough users
                if len(os.listdir(FACES_DIR)) >= 2:
                    train_model()
                
                return jsonify({'message': 'User added successfully'}), 200
            except Exception as e:
                return jsonify({'error': f'Failed to save user information: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Failed to capture face images'}), 500
            
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/retrain')
def retrain():
    train_model()
    return home()

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        train_model()
    app.run(debug=True)