import numpy as np
import librosa
try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip
import os
import traceback

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

            # Heuristic scoring
            anomalies = []
            score = 0
            
            if mfcc_var < 1200: 
                anomalies.append({
                    "type": "unnatural_mel_variance",
                    "details": f"Low MFCC variance detected: {mfcc_var:.2f}",
                    "confidence": 0.7
                })
                score += 30
                
            if avg_contrast < 15.0: 
                anomalies.append({
                    "type": "low_spectral_contrast",
                    "details": f"Abnormal spectral contrast: {avg_contrast:.2f}",
                    "confidence": 0.65
                })
                score += 20
                
            fake_probability = min(score + 10, 95) if anomalies else 15
            
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
                    "avg_contrast": float(avg_contrast)
                }
            }
            
        except Exception as e:
            print(f"Error analyzing audio: {str(e)}")
            traceback.print_exc()
            return {"has_audio": False, "error": str(e)}

    def run_detection(self, video_path):
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
