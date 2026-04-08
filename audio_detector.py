import numpy as np
import librosa
try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip
import os
import traceback


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}

class AudioDeepfakeDetector:
    def __init__(self):
        self.sr = 22050  # Target sample rate
        self.confidence_threshold = 0.5

    def extract_audio(self, video_path):
        """Extract audio from video file"""
        audio_path = video_path.rsplit('.', 1)[0] + '_temp_audio.wav'
        try:
            video = VideoFileClip(video_path)
            if video.audio is None:
                video.close()
                return None
            video.audio.write_audiofile(audio_path, logger=None, fps=self.sr)
            video.close()
            return audio_path
        except Exception as e:
            print(f"Error extracting audio: {str(e)}")
            return None

    def analyze_audio(self, audio_path):
        """Analyze audio for deepfake indicators"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)

            if y.ndim > 1:
                y = np.mean(y, axis=1)

            max_duration_seconds = 120
            if len(y) > sr * max_duration_seconds:
                y = y[: sr * max_duration_seconds]
            
            # Simple check if audio is empty or too short
            if len(y) < sr * 1.0: # Less than 1 second
                return {"has_audio": False, "message": "Audio too short"}

            # 1. High-frequency anomaly detection (deepfakes often lack or have distorted high frequencies)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)[0]
            avg_rolloff = float(np.mean(rolloff))
            
            # 2. MFCC Variance (synthetic speech often has highly regular/smooth MFCCs)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_var = float(np.mean(np.var(mfccs, axis=1)))
            
            # 3. Spectral contrast (differences between peaks and valleys in spectrum)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            avg_contrast = float(np.mean(contrast))

            # 4. Additional signals often unstable in generated audio
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = float(np.mean(zcr))
            zcr_var = float(np.var(zcr))

            flatness = librosa.feature.spectral_flatness(y=y)[0]
            avg_flatness = float(np.mean(flatness))

            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            avg_bandwidth = float(np.mean(bandwidth))

            rms = librosa.feature.rms(y=y)[0]
            rms_std = float(np.std(rms))

            # Heuristic scoring
            anomalies = []
            score = 0.0

            def add_anomaly(anomaly_type, details, confidence, weight):
                anomalies.append({
                    "type": anomaly_type,
                    "details": details,
                    "confidence": confidence
                })
                return weight
            
            if mfcc_var < 850:
                score += add_anomaly(
                    "unnatural_mel_variance",
                    f"Very low MFCC variance detected: {mfcc_var:.2f}",
                    0.82,
                    34,
                )
            elif mfcc_var < 1300:
                score += add_anomaly(
                    "unnatural_mel_variance",
                    f"Low MFCC variance detected: {mfcc_var:.2f}",
                    0.70,
                    20,
                )
                
            if avg_contrast < 12.5:
                score += add_anomaly(
                    "low_spectral_contrast",
                    f"Very low spectral contrast: {avg_contrast:.2f}",
                    0.76,
                    22,
                )
            elif avg_contrast < 16.5:
                score += add_anomaly(
                    "low_spectral_contrast",
                    f"Low spectral contrast: {avg_contrast:.2f}",
                    0.62,
                    12,
                )

            if avg_flatness > 0.18:
                score += add_anomaly(
                    "high_spectral_flatness",
                    f"Spectral flatness unusually high: {avg_flatness:.3f}",
                    0.74,
                    18,
                )
            elif avg_flatness > 0.11:
                score += add_anomaly(
                    "elevated_spectral_flatness",
                    f"Spectral flatness elevated: {avg_flatness:.3f}",
                    0.62,
                    10,
                )
            elif avg_flatness < 0.008:
                score += add_anomaly(
                    "overly_smooth_spectrum",
                    f"Spectral flatness unusually low: {avg_flatness:.3f}",
                    0.55,
                    8,
                )

            if rms_std < 0.012:
                score += add_anomaly(
                    "overly_consistent_amplitude",
                    f"Amplitude envelope variance is low: {rms_std:.4f}",
                    0.66,
                    12,
                )

            if zcr_mean < 0.025 or zcr_mean > 0.19:
                score += add_anomaly(
                    "abnormal_zero_crossing_rate",
                    f"Zero-crossing mean out of expected range: {zcr_mean:.4f}",
                    0.60,
                    10,
                )

            if zcr_var < 0.00035:
                score += add_anomaly(
                    "low_temporal_randomness",
                    f"Zero-crossing variance is unusually low: {zcr_var:.6f}",
                    0.58,
                    8,
                )

            if avg_bandwidth < 1250 or avg_bandwidth > 4200:
                score += add_anomaly(
                    "abnormal_spectral_bandwidth",
                    f"Spectral bandwidth outside typical range: {avg_bandwidth:.2f}",
                    0.60,
                    10,
                )

            if avg_rolloff < 2200 or avg_rolloff > 9000:
                score += add_anomaly(
                    "abnormal_high_frequency_profile",
                    f"Spectral rolloff outside expected range: {avg_rolloff:.2f}",
                    0.58,
                    8,
                )

            if len(anomalies) >= 3:
                score += 8
            if len(anomalies) >= 5:
                score += 10
                
            fake_probability = int(min(max(score, 12), 95)) if anomalies else 12
            model_confidence = float(min(0.98, 0.35 + (len(anomalies) * 0.09) + (fake_probability / 220.0)))
            
            # Construct summary for compatibility with visual anomalies
            anomaly_summary = []
            for a in anomalies:
                anomaly_summary.append({
                    'frame': 0, 'timestamp': 0, 
                    'anomaly': True, 
                    'type': a['type'], 
                    'confidence': a['confidence'], 
                    'details': a['details']
                })
                
            return {
                "has_audio": True,
                "fake_probability": fake_probability,
                "anomalies": anomaly_summary,
                "metrics": {
                    "avg_rolloff": float(avg_rolloff),
                    "avg_mfcc_var": float(mfcc_var),
                    "avg_contrast": float(avg_contrast),
                    "avg_flatness": float(avg_flatness),
                    "avg_bandwidth": float(avg_bandwidth),
                    "zcr_mean": float(zcr_mean),
                    "zcr_var": float(zcr_var),
                    "rms_std": float(rms_std),
                    "duration_seconds": round(float(len(y) / sr), 2)
                },
                "confidence": model_confidence,
            }
            
        except Exception as e:
            print(f"Error analyzing audio: {str(e)}")
            traceback.print_exc()
            return {"has_audio": False, "error": str(e)}

    def run_detection_on_audio(self, audio_path):
        """Run deepfake-audio analysis directly on an audio file."""
        print("Analyzing uploaded audio file...")
        return self.analyze_audio(audio_path)

    def run_detection(self, video_path):
        _, ext = os.path.splitext(video_path)
        if ext.lower() in AUDIO_EXTENSIONS:
            return self.run_detection_on_audio(video_path)

        print("Extracting audio for analysis...")
        audio_path = self.extract_audio(video_path)
        
        if not audio_path:
            return {"has_audio": False, "message": "No audio track found or extraction failed."}
            
        print("Analyzing audio signature...")
        results = self.analyze_audio(audio_path)
        
        # Cleanup
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
        return results
