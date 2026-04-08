from flask import Flask, render_template, request, redirect, url_for
import importlib
import os
import mimetypes
from datetime import datetime, timezone
from time import time as current_time

from explanation_service import generate_media_explanation

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    import cv2
except Exception:
    cv2 = None

if load_dotenv is not None:
    load_dotenv(override=False)

app = Flask(__name__, static_folder="static")
app.secret_key = "deepfake_detection_secret_key"

# Preload heavy detection module so model files/cache are created
# at startup instead of during a request (prevents unexpected file
# writes while the reloader/watchdog is running).
try:
    import deepfake_detector
except Exception:
    deepfake_detector = None

try:
    import audio_detector
except Exception:
    audio_detector = None

# Create these directories at startup to ensure they exist
STATIC_DIR = "static"
UPLOAD_FOLDER = os.path.join(STATIC_DIR, "videos")
IMAGES_DIR = os.path.join(STATIC_DIR, "img")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max upload

VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "wmv", "webm", "mkv"}
AUDIO_EXTENSIONS = {"wav", "mp3", "m4a", "aac", "flac", "ogg", "opus"}
ALLOWED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

VIDEO_MIME_BY_EXT = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".ogv": "video/ogg",
    ".ogg": "video/ogg",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".wmv": "video/x-ms-wmv",
}

AUDIO_MIME_BY_EXT = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
}

# Create default placeholder images if they don't exist
def ensure_placeholder_images():
    placeholders = {
        "video-poster.jpg": "https://images.unsplash.com/photo-1618609377864-68609b857e90?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=928&q=80",
        "video-placeholder.jpg": "https://images.unsplash.com/photo-1590856029826-c7a73142bbf1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1473&q=80",
        "video-error.jpg": "https://images.unsplash.com/photo-1594322436404-5a0526db4d13?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1529&q=80",
    }
    
    for filename, url in placeholders.items():
        filepath = os.path.join(IMAGES_DIR, filename)
        if not os.path.exists(filepath):
            try:
                import requests

                response = requests.get(url)
                with open(filepath, "wb") as f:
                    f.write(response.content)
                print(f"Created placeholder image: {filepath}")
            except Exception as e:
                print(f"Could not create placeholder image {filename}: {str(e)}")

ensure_placeholder_images()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _file_extension(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower() if ext else ""


def _media_type_from_extension(ext):
    if ext.startswith("."):
        ext = ext[1:]
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    return "unknown"


def _is_video_decodable(file_path):
    if not file_path or not os.path.exists(file_path):
        return False

    try:
        # Tiny clips can be valid and still be small; only reject empty files.
        if os.path.getsize(file_path) <= 0:
            return False
    except OSError:
        return False

    if cv2 is None:
        return True

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return False

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, _ = cap.read()
    cap.release()

    return frame_count > 0 and width > 0 and height > 0 and ok


def _video_mime_type(filename):
    ext = _file_extension(filename)
    if ext in VIDEO_MIME_BY_EXT:
        return VIDEO_MIME_BY_EXT[ext]
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "video/mp4"


def _audio_mime_type(filename):
    ext = _file_extension(filename)
    if ext in AUDIO_MIME_BY_EXT:
        return AUDIO_MIME_BY_EXT[ext]
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "audio/mpeg"


def _transcode_to_browser_mp4(source_path, target_path):
    if cv2 is None:
        return False

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    if width <= 0 or height <= 0:
        cap.release()
        return False

    # Write to a temporary file first, then atomically replace the target
    tmp_path = f"{target_path}.part"
    # ensure no stale temp file
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception:
        pass

    out = None
    for codec in ("mp4v", "avc1", "H264"):
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            candidate = cv2.VideoWriter(tmp_path, fourcc, float(fps), (width, height))
            if candidate.isOpened():
                out = candidate
                break
            candidate.release()
        except Exception:
            continue

    if out is None:
        cap.release()
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False

    wrote_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame)
        wrote_frames += 1

    cap.release()
    out.release()

    if wrote_frames <= 0 or not _is_video_decodable(tmp_path):
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False

    try:
        os.replace(tmp_path, target_path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False

    return True


def _select_video_preview(uploaded_filename, uploaded_path, processed_filename, output_path, timestamp, reasoning_data):
    if _is_video_decodable(output_path):
        return processed_filename, _video_mime_type(processed_filename)

    if os.path.exists(output_path):
        reasoning_data["reasoning_points"].append(
            {
                "icon": "fa-circle-info",
                "title": "Preview Fallback",
                "detail": "Processed output could not be validated for playback, so the original upload is shown for reliable preview.",
            }
        )

    uploaded_decodable = _is_video_decodable(uploaded_path)

    if uploaded_decodable and _file_extension(uploaded_filename) == ".mp4":
        reasoning_data["reasoning_points"].append(
            {
                "icon": "fa-file-video",
                "title": "Preview Fallback",
                "detail": "Showing original uploaded MP4 because processed preview could not be generated.",
            }
        )
        return uploaded_filename, _video_mime_type(uploaded_filename)

    if uploaded_decodable:
        transcoded_filename = f"preview_fallback_{timestamp}.mp4"
        transcoded_path = os.path.join(app.config["UPLOAD_FOLDER"], transcoded_filename)
        if _transcode_to_browser_mp4(uploaded_path, transcoded_path):
            reasoning_data["reasoning_points"].append(
                {
                    "icon": "fa-repeat",
                    "title": "Preview Transcoding",
                    "detail": "Converted uploaded video to MP4 for browser playback because processed preview was unavailable.",
                }
            )
            return transcoded_filename, _video_mime_type(transcoded_filename)

    return uploaded_filename, _video_mime_type(uploaded_filename)

    return uploaded_filename


def _default_reasoning(error_message=None):
    reasoning_points = []
    if error_message:
        reasoning_points.append(
            {
                "icon": "fa-circle-exclamation",
                "title": "Analysis Error",
                "detail": error_message,
            }
        )

    return {
        "total_frames": 0,
        "frames_processed": 0,
        "faces_detected": 0,
        "deepfake_frames": 0,
        "low_similarity_count": 0,
        "no_face_frames": 0,
        "avg_similarity": 0,
        "min_similarity": 0,
        "max_similarity": 0,
        "face_detection_rate": 0,
        "execution_time": 0,
        "threshold_similarity": 0.99,
        "threshold_consecutive": 15,
        "reasoning_points": reasoning_points,
        "video_resolution": "N/A",
        "video_fps": 0,
    }


def _append_audio_reasoning(reasoning_data, audio_results):
    if audio_results.get("has_audio"):
        audio_prob = int(audio_results.get("fake_probability", 0))
        reasoning_data["reasoning_points"].append(
            {
                "icon": "fa-microphone",
                "title": "Audio Analysis",
                "detail": f"Audio track analyzed. Detected audio manipulation probability: {audio_prob}%.",
            }
        )

        anomalies = audio_results.get("anomalies") or []
        if anomalies:
            reasoning_data["reasoning_points"].append(
                {
                    "icon": "fa-volume-up" if audio_prob >= 50 else "fa-check-circle",
                    "title": "Audio Anomalies",
                    "detail": f"Detected {len(anomalies)} audio anomaly indicators in spectral features.",
                }
            )
    else:
        reasoning_data["reasoning_points"].append(
            {
                "icon": "fa-microphone-slash",
                "title": "Audio Analysis Skipped",
                "detail": audio_results.get("message", "No audio track available or extraction failed."),
            }
        )

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
            timestamp = int(current_time())
            ext = _file_extension(file.filename)
            if not ext:
                ext = ".mp4"

            media_type = _media_type_from_extension(ext)
            if media_type == "unknown":
                return redirect(url_for("index"))

            uploaded_filename = f"uploaded_{media_type}_{timestamp}{ext}"
            media_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_filename)
            file.save(media_path)

            result_percent = 0
            reasoning_data = _default_reasoning()
            audio_results = {}
            video_url = None
            audio_url = None
            video_mime_type = None
            audio_mime_type = None

            if media_type == "video":
                processed_filename = f"processed_video_{timestamp}.mp4"
                output_path = os.path.join(app.config["UPLOAD_FOLDER"], processed_filename)

                try:
                    if deepfake_detector is None:
                        module = importlib.import_module("deepfake_detector")
                        result_percent, reasoning_data = module.run(media_path, output_path)
                    else:
                        result_percent, reasoning_data = deepfake_detector.run(media_path, output_path)
                except Exception as e:
                    print(f"Error in deepfake detection: {str(e)}")
                    result_percent = 50
                    reasoning_data = _default_reasoning(
                        f"An error occurred during video detection: {str(e)}. Default score of 50% applied."
                    )

                try:
                    if audio_detector is None:
                        audio_module = importlib.import_module("audio_detector")
                    else:
                        audio_module = audio_detector

                    detector = audio_module.AudioDeepfakeDetector()
                    audio_results = detector.run_detection(media_path)
                    _append_audio_reasoning(reasoning_data, audio_results)

                    if audio_results.get("has_audio"):
                        audio_prob = int(audio_results.get("fake_probability", 0))
                        if audio_prob > result_percent and audio_prob > 50:
                            old_pct = int(result_percent)
                            result_percent = min(int((result_percent + audio_prob * 2) / 3), 95)
                            reasoning_data["reasoning_points"].append(
                                {
                                    "icon": "fa-scale-balanced",
                                    "title": "Multimodal Verdict Adjustment",
                                    "detail": f"Audio anomalies raised the overall probability from {old_pct}% to {result_percent}%.",
                                }
                            )
                except Exception as e:
                    print(f"Error in audio detection: {str(e)}")
                    reasoning_data["reasoning_points"].append(
                        {
                            "icon": "fa-circle-exclamation",
                            "title": "Audio Analysis Error",
                            "detail": f"An error prevented audio analysis: {str(e)}",
                        }
                    )

                display_filename, video_mime_type = _select_video_preview(
                    uploaded_filename=uploaded_filename,
                    uploaded_path=media_path,
                    processed_filename=processed_filename,
                    output_path=output_path,
                    timestamp=timestamp,
                    reasoning_data=reasoning_data,
                )
                video_url = f"/static/videos/{display_filename}"
            else:
                try:
                    if audio_detector is None:
                        audio_module = importlib.import_module("audio_detector")
                    else:
                        audio_module = audio_detector

                    detector = audio_module.AudioDeepfakeDetector()
                    audio_results = detector.run_detection(media_path)

                    if audio_results.get("has_audio"):
                        result_percent = int(audio_results.get("fake_probability", 0))
                        _append_audio_reasoning(reasoning_data, audio_results)
                    else:
                        result_percent = 0
                        reasoning_data["reasoning_points"].append(
                            {
                                "icon": "fa-microphone-slash",
                                "title": "Audio Analysis Skipped",
                                "detail": audio_results.get("message", "Audio could not be analyzed."),
                            }
                        )
                except Exception as e:
                    print(f"Error in audio detection: {str(e)}")
                    result_percent = 50
                    reasoning_data = _default_reasoning(
                        f"An error occurred during audio detection: {str(e)}. Default score of 50% applied."
                    )

                reasoning_data["reasoning_points"].insert(
                    0,
                    {
                        "icon": "fa-music",
                        "title": "Audio-Only Analysis",
                        "detail": "This upload was detected as an audio file. Video-frame analysis was skipped.",
                    },
                )
                audio_url = f"/static/videos/{uploaded_filename}"
                audio_mime_type = _audio_mime_type(uploaded_filename)

            ai_enrichment = generate_media_explanation(
                media_name=file.filename,
                media_type=media_type,
                result_percent=int(result_percent),
                reasoning_data=reasoning_data,
                audio_results=audio_results,
            )

            final_explanation = ai_enrichment.get("summary") or "Analysis completed successfully."

            existing_titles = {point.get("title") for point in reasoning_data.get("reasoning_points", [])}
            for point in ai_enrichment.get("reasoning_points", []):
                if point.get("title") not in existing_titles:
                    reasoning_data["reasoning_points"].append(point)

            if ai_enrichment.get("search_summary") and "Google Search Check" not in existing_titles:
                reasoning_data["reasoning_points"].append(
                    {
                        "icon": "fa-magnifying-glass",
                        "title": "Google Search Check",
                        "detail": ai_enrichment["search_summary"],
                    }
                )

            video_info = {
                "name": file.filename,
                "size": f"{os.path.getsize(media_path) / 1024:.2f} KB",
                "user": "Guest",
                "source": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "per": int(result_percent),
                "media_type": media_type,
                "analysis_type": "Video + Audio" if media_type == "video" else "Audio Only",
            }

            return render_template(
                "result.html",
                video_url=video_url,
                audio_url=audio_url,
                video_mime_type=video_mime_type,
                audio_mime_type=audio_mime_type,
                video_info=video_info,
                reasoning=reasoning_data,
                final_explanation=final_explanation,
            )

        except Exception as e:
            print(f"Error processing upload: {str(e)}")
            return render_template("error.html", error_message="An error occurred while processing your upload.")
    
    # If file type is not allowed
    return redirect(url_for('index'))

@app.route('/result')
def result():
    return redirect(url_for('index'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error_message="Page not found."), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', error_message="Internal server error."), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    # Disable the watchdog/reloader for heavy file/model operations
    # so that runtime writes (model downloads, cache, etc.) don't
    # cause the dev server to restart while processing uploads.
    app.run(host="0.0.0.0", port=port, use_reloader=False)
