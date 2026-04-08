import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.transforms import functional as F
import time
import os
import shutil

# Pre-initialize models at import time so they aren't recreated on every
# uploaded-video request. This reduces disk/network activity during a
# request (which can trigger Flask's reloader/watchdog if files are written).
try:
    mtcnn = MTCNN()
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
except Exception as _e:
    # If initialization fails at import, set placeholders and attempt
    # lazy initialization in `run()` (keeps runtime robust during dev).
    print(f"Warning: detector model init at import failed: {_e}")
    mtcnn = None
    facenet_model = None


def _clamp(value, low, high):
    return max(low, min(high, value))


def _compute_calibrated_score(
    low_similarity_count,
    similarity_scores,
    deep_fake_frame_count,
    frames_processed_count,
    avg_similarity,
    threshold_similarity,
    max_anomaly_streak,
    threshold_frames_for_deepfake,
    face_detection_rate,
):
    if frames_processed_count <= 0 or not similarity_scores:
        return 0.0

    similarity_events = len(similarity_scores)
    low_similarity_rate = low_similarity_count / similarity_events
    deepfake_frame_rate = deep_fake_frame_count / frames_processed_count

    avg_similarity_gap = max(0.0, threshold_similarity - avg_similarity)
    avg_gap_norm = _clamp(avg_similarity_gap / 0.08, 0.0, 1.0)
    streak_norm = _clamp(
        max_anomaly_streak / float(max(1, threshold_frames_for_deepfake)),
        0.0,
        1.6,
    ) / 1.6

    score = (
        (low_similarity_rate * 55.0)
        + (deepfake_frame_rate * 25.0)
        + (avg_gap_norm * 15.0)
        + (streak_norm * 15.0)
    )

    if low_similarity_count >= 3:
        score += 5.0
    if low_similarity_count >= 8:
        score += 8.0
    if low_similarity_count >= 15:
        score += 7.0

    confidence_factor = 1.0
    if face_detection_rate < 30:
        confidence_factor *= 0.75
    if similarity_events < 6:
        confidence_factor *= 0.70
    elif similarity_events < 12:
        confidence_factor *= 0.85

    score *= confidence_factor
    return _clamp(score, 0.0, 95.0)


def run(video_path, video_path2):

    start_time = time.time()

    # Equivalents for deepfake detection
    threshold_face_similarity = 0.975
    threshold_frames_for_deepfake = 8

    global mtcnn, facenet_model

    # Lazy fallback: try to initialize if import-time initialization failed.
    # If models cannot be initialized (missing packages or no GPU), run in
    # lightweight passthrough mode so we still produce a valid processed
    # preview that can be played back in the browser.
    models_available = True
    if mtcnn is None or facenet_model is None:
        try:
            mtcnn = MTCNN()
            facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
        except Exception as _e:
            print(f"Warning: could not initialize face models: {_e}")
            models_available = False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video file: {video_path}")

    frame_count = 0
    # Preserve FPS as float for VideoWriter (some codecs expect float)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    fps = float(fps)

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid video dimensions read from input file.")

    # Try multiple codecs; write to a temp file first and atomically rename
    # to the final path after validation. This prevents serving partially
    # written or corrupt files to clients.
    out = None
    selected_codec = None
    tried_codecs = ('mp4v', 'H264', 'X264', 'avc1', 'XVID')
    tmp_path = f"{video_path2}.part"
    for codec in tried_codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        try:
            out = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
            if out is not None and out.isOpened():
                print(f"VideoWriter codec selected: {codec}")
                selected_codec = codec
                break
        except Exception:
            out = None
    # `writer_active` indicates we have a VideoWriter instance we can write
    # frames to. `preview_generated` means a playable preview file already
    # exists (either written by the VideoWriter or copied/transcoded as a
    # fallback). These are distinct because a successful fallback copy
    # produces a preview file but there is no writer to stream frames to.
    writer_active = out is not None and out.isOpened()
    preview_generated = False
    if not writer_active:
        out = None
        # Ensure no leftover temp file
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        print('Warning: could not initialize VideoWriter with available codecs. Attempting fallback by copying original upload to preview.')
        try:
            # Try copying the uploaded file as a safe playback fallback
            shutil.copy2(video_path, tmp_path)
            # quick validation
            cap_test = cv2.VideoCapture(tmp_path)
            valid_copy = cap_test.isOpened()
            if valid_copy:
                out_fc = int(cap_test.get(cv2.CAP_PROP_FRAME_COUNT))
                out_w = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
                out_h = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
                ok_read, _ = cap_test.read()
                valid_copy = valid_copy and out_fc > 0 and out_w > 0 and out_h > 0 and bool(ok_read)
            cap_test.release()
            if valid_copy:
                try:
                    # Move the validated copy into place. We mark the preview
                    # as generated, but keep `writer_active` False because we
                    # don't have a writer instance to stream frames to.
                    os.replace(tmp_path, video_path2)
                    preview_generated = True
                    selected_codec = 'copy'
                    print('Fallback preview created by copying original upload')
                except Exception as _r:
                    print(f'Warning: could not move copied preview into place: {_r}')
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                    preview_generated = False
                    selected_codec = None
            else:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                preview_generated = False
                selected_codec = None
        except Exception as _copy_err:
            print(f'Fallback copy preview failed: {_copy_err}')
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            preview_generated = False

    deepfake_count = 0
    deep_fake_frame_count = 0
    previous_face_encoding = None
    frames_between_processing = max(1, int(fps / 7))
    resize_dim = (80, 80)

    # --- Reasoning metrics ---
    faces_detected_count = 0
    frames_processed_count = 0
    similarity_scores = []
    low_similarity_count = 0
    no_face_frames = 0
    max_anomaly_streak = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frames_between_processing == 0:
            frames_processed_count += 1
            if models_available:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, _ = mtcnn.detect(frame_rgb)

                if boxes is not None and len(boxes) > 0:
                    faces_detected_count += 1
                    box = boxes[0].astype(int)
                    x1 = _clamp(int(box[0]), 0, width - 1)
                    y1 = _clamp(int(box[1]), 0, height - 1)
                    x2 = _clamp(int(box[2]), 0, width)
                    y2 = _clamp(int(box[3]), 0, height)

                    if x2 <= x1 or y2 <= y1:
                        frame_count += 1
                        if writer_active:
                            out.write(frame)
                        continue

                    face = frame[y1:y2, x1:x2]

                    if not face.size == 0:
                        face = cv2.resize(face, resize_dim)
                        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        face_tensor = F.to_tensor(face_rgb).unsqueeze(0)
                        current_face_encoding = facenet_model(face_tensor).detach().numpy().flatten()

                        if previous_face_encoding is not None:
                            denom = np.linalg.norm(current_face_encoding) * np.linalg.norm(previous_face_encoding)
                            if denom == 0:
                                face_similarity = 1.0
                            else:
                                face_similarity = np.dot(current_face_encoding, previous_face_encoding) / denom

                            similarity_scores.append(float(face_similarity))

                            if face_similarity < threshold_face_similarity:
                                deepfake_count += 1
                                low_similarity_count += 1
                                if deepfake_count > max_anomaly_streak:
                                    max_anomaly_streak = deepfake_count
                            else:
                                deepfake_count = 0

                            if deepfake_count >= threshold_frames_for_deepfake:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(frame, f'Deepfake Detected - Frame {frame_count}', (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                                deep_fake_frame_count += 1
                            else:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, 'Real Frame', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                                            cv2.LINE_AA)

                        previous_face_encoding = current_face_encoding
                else:
                    no_face_frames += 1
            else:
                # Models unavailable: annotate lightly so output is still useful/playable
                cv2.putText(frame, 'Preview (models unavailable)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        frame_count += 1
        if writer_active:
            out.write(frame)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Total Execution Time: {execution_time} seconds")

    cap.release()
    if out is not None:
        out.release()

    # Validate the generated temp preview file and atomically move it to
    # the final destination if valid. Remove the temp file otherwise.
    if writer_active:
        try:
            cap_out = cv2.VideoCapture(tmp_path)
            valid = cap_out.isOpened()
            if valid:
                out_frame_count = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))
                out_width = int(cap_out.get(cv2.CAP_PROP_FRAME_WIDTH))
                out_height = int(cap_out.get(cv2.CAP_PROP_FRAME_HEIGHT))
                ok_read, _ = cap_out.read()
                valid = valid and out_frame_count > 0 and out_width > 0 and out_height > 0 and bool(ok_read)
            cap_out.release()
            if valid:
                try:
                    # Atomic replace where supported
                    os.replace(tmp_path, video_path2)
                    preview_generated = True
                except Exception as _rerr:
                    print(f'Warning: could not rename temp preview to final path: {_rerr}')
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                    preview_generated = False
                    selected_codec = None
            else:
                print('Warning: processed preview validation failed, removing generated temp file.')
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                preview_generated = False
                selected_codec = None
        except Exception as _ve:
            print(f'Warning: error validating processed preview: {_ve}')
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            preview_generated = False
            selected_codec = None

    legacy_accuracy = (deep_fake_frame_count / frames_processed_count) * 100 if frames_processed_count > 0 else 0

    avg_similarity = float(np.mean(similarity_scores)) if similarity_scores else 0.0
    min_similarity = float(np.min(similarity_scores)) if similarity_scores else 0.0
    max_similarity = float(np.max(similarity_scores)) if similarity_scores else 0.0
    face_detection_rate = round((faces_detected_count / frames_processed_count) * 100, 1) if frames_processed_count > 0 else 0.0

    calibrated_accuracy = _compute_calibrated_score(
        low_similarity_count=low_similarity_count,
        similarity_scores=similarity_scores,
        deep_fake_frame_count=deep_fake_frame_count,
        frames_processed_count=frames_processed_count,
        avg_similarity=avg_similarity,
        threshold_similarity=threshold_face_similarity,
        max_anomaly_streak=max_anomaly_streak,
        threshold_frames_for_deepfake=threshold_frames_for_deepfake,
        face_detection_rate=face_detection_rate,
    )

    accuracy = max(legacy_accuracy, calibrated_accuracy)

    if accuracy > 100:
        accuracy = 95

    # --- Build reasoning data ---
    # Build human-readable reasoning points
    reasoning_points = []

    # 1. Frame analysis summary
    reasoning_points.append({
        'icon': 'fa-film',
        'title': 'Frame Analysis',
        'detail': f'Analyzed {frames_processed_count} sampled frames out of {frame_count} total frames ({round(frames_processed_count / frame_count * 100, 1) if frame_count > 0 else 0}% coverage).'
    })

    if (preview_generated or writer_active) and selected_codec:
        reasoning_points.append({
            'icon': 'fa-file-video',
            'title': 'Processed Preview Generation',
            'detail': f'Generated processed preview video using {selected_codec} codec for browser playback.'
        })
    else:
        reasoning_points.append({
            'icon': 'fa-circle-info',
            'title': 'Processed Preview Generation',
            'detail': 'Could not generate a browser-safe processed preview with available codecs; original upload will be shown in the result page.'
        })

    # 2. Face detection summary
    reasoning_points.append({
        'icon': 'fa-face-smile',
        'title': 'Face Detection',
        'detail': f'Faces were detected in {faces_detected_count} of {frames_processed_count} processed frames ({face_detection_rate}% detection rate). {no_face_frames} frames had no detectable face.'
    })

    # 3. Similarity analysis
    if similarity_scores:
        reasoning_points.append({
            'icon': 'fa-chart-line',
            'title': 'Face Similarity Consistency',
            'detail': f'Average inter-frame face similarity: {avg_similarity:.4f} (range: {min_similarity:.4f} – {max_similarity:.4f}). Threshold for anomaly: {threshold_face_similarity}.'
        })

        # 4. Anomaly count
        reasoning_points.append({
            'icon': 'fa-triangle-exclamation',
            'title': 'Anomalous Frames Detected',
            'detail': f'{low_similarity_count} frame comparisons fell below the similarity threshold ({threshold_face_similarity}), suggesting potential face-swapping or manipulation artifacts.'
        })

        reasoning_points.append({
            'icon': 'fa-wave-square',
            'title': 'Anomaly Streak Strength',
            'detail': f'Max consecutive anomaly streak: {max_anomaly_streak} sampled frames (threshold for strong concern: {threshold_frames_for_deepfake}).'
        })
    else:
        reasoning_points.append({
            'icon': 'fa-chart-line',
            'title': 'Face Similarity Consistency',
            'detail': 'Not enough consecutive face detections to compute similarity scores.'
        })

    # 5. Deepfake flagged frames
    reasoning_points.append({
        'icon': 'fa-flag',
        'title': 'Flagged Deepfake Frames',
        'detail': f'{deep_fake_frame_count} frames were flagged as deepfake (consecutive anomaly streak exceeded {threshold_frames_for_deepfake} frame threshold).'
    })

    reasoning_points.append({
        'icon': 'fa-scale-balanced',
        'title': 'Calibrated Risk Score',
        'detail': f'Final score uses anomaly rate, streak strength, and similarity drop calibration: legacy={int(legacy_accuracy)}%, calibrated={int(calibrated_accuracy)}%, final={int(accuracy)}%.'
    })

    # 6. Processing time
    reasoning_points.append({
        'icon': 'fa-clock',
        'title': 'Processing Time',
        'detail': f'Total analysis completed in {execution_time:.2f} seconds using MTCNN face detection and FaceNet (InceptionResnetV1) embedding comparison.'
    })

    # 7. Final verdict reasoning
    pct = int(accuracy)
    if pct >= 75:
        verdict_reason = 'A high number of consecutive frames showed significant facial inconsistencies, strongly indicating the use of deepfake or face-swap technology.'
    elif pct >= 50:
        verdict_reason = 'Several frames exhibited facial anomalies, but insufficient consecutive streaks were detected to make a definitive determination. The video may contain partial manipulation.'
    else:
        verdict_reason = 'The facial features remained highly consistent across frames with very few anomalies, suggesting the video is likely authentic and unmanipulated.'

    reasoning_points.append({
        'icon': 'fa-gavel',
        'title': 'Verdict Rationale',
        'detail': verdict_reason
    })

    reasoning_data = {
        'total_frames': frame_count,
        'frames_processed': frames_processed_count,
        'faces_detected': faces_detected_count,
        'deepfake_frames': deep_fake_frame_count,
        'low_similarity_count': low_similarity_count,
        'no_face_frames': no_face_frames,
        'avg_similarity': round(avg_similarity, 4),
        'min_similarity': round(min_similarity, 4),
        'max_similarity': round(max_similarity, 4),
        'face_detection_rate': face_detection_rate,
        'execution_time': round(execution_time, 2),
        'threshold_similarity': threshold_face_similarity,
        'threshold_consecutive': threshold_frames_for_deepfake,
        'max_anomaly_streak': int(max_anomaly_streak),
        'legacy_accuracy': round(float(legacy_accuracy), 2),
        'calibrated_accuracy': round(float(calibrated_accuracy), 2),
        'processed_preview_generated': bool(preview_generated),
        'processed_preview_codec': selected_codec,
        'reasoning_points': reasoning_points,
        'video_resolution': f'{width}x{height}',
        'video_fps': fps
    }

    return int(accuracy), reasoning_data

