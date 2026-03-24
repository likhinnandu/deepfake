from flask import Flask, render_template, request, redirect, url_for, session
import os
import shutil
from datetime import datetime
import json
from time import time as current_time
import importlib

app = Flask(__name__, static_folder='static')
app.secret_key = 'deepfake_detection_secret_key'

# Create these directories at startup to ensure they exist
STATIC_DIR = 'static'
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'videos')  # Consistent folder path
IMAGES_DIR = os.path.join(STATIC_DIR, 'img')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'webm'}

# Create default placeholder images if they don't exist
def ensure_placeholder_images():
    placeholders = {
        'video-poster.jpg': 'https://images.unsplash.com/photo-1618609377864-68609b857e90?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=928&q=80',
        'video-placeholder.jpg': 'https://images.unsplash.com/photo-1590856029826-c7a73142bbf1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1473&q=80',
        'video-error.jpg': 'https://images.unsplash.com/photo-1594322436404-5a0526db4d13?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1529&q=80'
    }
    
    for filename, url in placeholders.items():
        filepath = os.path.join(IMAGES_DIR, filename)
        if not os.path.exists(filepath):
            try:
                import requests
                response = requests.get(url)
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"Created placeholder image: {filepath}")
            except Exception as e:
                print(f"Could not create placeholder image {filename}: {str(e)}")

ensure_placeholder_images()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

# Handle file upload and redirect to the result page
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        try:
            # Save the uploaded file
            timestamp = int(current_time())
            filename = f"uploaded_video_{timestamp}.mp4"
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)

            # Define the output file path
            output_filename = filename  # Just use the same filename for now
            output_path = video_path    # Use the same path to avoid copy issues

            # Run the detection
            try:
                module = importlib.import_module("deepfake_detector")
                function = getattr(module, "run")
                result_percent, reasoning_data = function(video_path, output_path)
            except Exception as e:
                print(f"Error in deepfake detection: {str(e)}")
                result_percent = 50  # Default value if detection fails
                reasoning_data = {
                    'total_frames': 0, 'frames_processed': 0, 'faces_detected': 0,
                    'deepfake_frames': 0, 'low_similarity_count': 0, 'no_face_frames': 0,
                    'avg_similarity': 0, 'min_similarity': 0, 'max_similarity': 0,
                    'face_detection_rate': 0, 'execution_time': 0,
                    'threshold_similarity': 0.99, 'threshold_consecutive': 15,
                    'reasoning_points': [{'icon': 'fa-circle-exclamation', 'title': 'Analysis Error', 'detail': f'An error occurred during detection: {str(e)}. Default score of 50% applied.'}],
                    'video_resolution': 'N/A', 'video_fps': 0
                }

            # Generate proper URL for the video that will work in the browser
            # Make sure the path is relative to the static folder structure
            video_url = f'/static/videos/{filename}'

            # Get video information
            video_info = {
                'name': file.filename,
                'size': f"{os.path.getsize(video_path) / (1024):.2f} KB",
                'user': 'Guest', 
                'source': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
                'per': result_percent
            }

            # Render the result page with the video information
            return render_template('result.html', video_url=video_url, video_info=video_info, reasoning=reasoning_data)
        
        except Exception as e:
            print(f"Error processing upload: {str(e)}")
            return render_template('error.html', error_message="An error occurred while processing your upload.")
    
    # If file type is not allowed
    return redirect(url_for('index'))

@app.route('/result')
def result():
    if 'video_info' not in session or 'video_url' not in session:
        return redirect(url_for('index'))
    
    video_info = session['video_info']
    video_url = session['video_url']
    
    # Clear session data after retrieving it
    # session.pop('video_info', None)
    # session.pop('video_url', None)
    
    return render_template('result.html', video_url=video_url, video_info=video_info)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error_message="Page not found."), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', error_message="Internal server error."), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
