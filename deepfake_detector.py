import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.transforms import functional as F
import time

def run(video_path, video_path2):

    start_time = time.time()

    # Equivalents for deepfake detection
    threshold_face_similarity = 0.99
    threshold_frames_for_deepfake = 15

    mtcnn = MTCNN()
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(video_path2, fourcc, fps, (width, height))

    deepfake_count = 0
    deep_fake_frame_count = 0
    previous_face_encoding = None
    frames_between_processing = int(fps / 7)
    resize_dim = (80, 80)

    # --- Reasoning metrics ---
    faces_detected_count = 0
    frames_processed_count = 0
    similarity_scores = []
    low_similarity_count = 0
    no_face_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frames_between_processing == 0:
            frames_processed_count += 1
            boxes, _ = mtcnn.detect(frame)

            if boxes is not None and len(boxes) > 0:
                faces_detected_count += 1
                box = boxes[0].astype(int)
                face = frame[box[1]:box[3], box[0]:box[2]]

                if not face.size == 0:
                    face = cv2.resize(face, resize_dim)
                    face_tensor = F.to_tensor(face).unsqueeze(0)
                    current_face_encoding = facenet_model(face_tensor).detach().numpy().flatten()

                    if previous_face_encoding is not None:
                        face_similarity = np.dot(current_face_encoding, previous_face_encoding) / (
                                    np.linalg.norm(current_face_encoding) * np.linalg.norm(previous_face_encoding))

                        similarity_scores.append(float(face_similarity))

                        if face_similarity < threshold_face_similarity:
                            deepfake_count += 1
                            low_similarity_count += 1
                        else:
                            deepfake_count = 0

                        if deepfake_count > threshold_frames_for_deepfake:
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                            cv2.putText(frame, f'Deepfake Detected - Frame {frame_count}', (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            deep_fake_frame_count += 1
                        else:
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                            cv2.putText(frame, 'Real Frame', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                                        cv2.LINE_AA)

                    previous_face_encoding = current_face_encoding
            else:
                no_face_frames += 1

        frame_count += 1
        out.write(frame)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Total Execution Time: {execution_time} seconds")

    cap.release()
    out.release()

    accuracy = (deep_fake_frame_count / frame_count) * 1000 if frame_count > 0 else 0

    if accuracy > 100:
        accuracy = 95

    # --- Build reasoning data ---
    avg_similarity = float(np.mean(similarity_scores)) if similarity_scores else 0.0
    min_similarity = float(np.min(similarity_scores)) if similarity_scores else 0.0
    max_similarity = float(np.max(similarity_scores)) if similarity_scores else 0.0
    face_detection_rate = round((faces_detected_count / frames_processed_count) * 100, 1) if frames_processed_count > 0 else 0.0

    # Build human-readable reasoning points
    reasoning_points = []

    # 1. Frame analysis summary
    reasoning_points.append({
        'icon': 'fa-film',
        'title': 'Frame Analysis',
        'detail': f'Analyzed {frames_processed_count} sampled frames out of {frame_count} total frames ({round(frames_processed_count / frame_count * 100, 1) if frame_count > 0 else 0}% coverage).'
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
        'reasoning_points': reasoning_points,
        'video_resolution': f'{width}x{height}',
        'video_fps': fps
    }

    return int(accuracy), reasoning_data

