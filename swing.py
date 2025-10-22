from fastapi import FastAPI, UploadFile, File, Form, Response, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import base64
import time
from functools import lru_cache
import hashlib
import io
import json
import queue
import threading
from datetime import datetime
import uuid
from enum import Enum
from dataclasses import dataclass
import logging
import traceback
import psutil
import math

app = FastAPI()

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allow CORS for testing/dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"]
)

# File size limits
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_CACHE_SIZE = 100  # Limit cache size

# Middleware for file size limits
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.method == "POST" and ("swing/frame" in str(request.url) or "detect" in str(request.url)):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_FILE_SIZE:
            logger.warning(f"File too large: {content_length} bytes")
            return JSONResponse(
                status_code=413,
                content={"error": "File too large", "success": False}
            )
    return await call_next(request)

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "success": False}
    )

# Optimized thread pool for better performance
executor = ThreadPoolExecutor(max_workers=8)

model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
onnx_session = None
onnx_input_name = None
onnx_output_names = None

# Low-level performance knobs (accuracy-neutral)
try:
    # Enable cuDNN autotuner and TF32 where beneficial
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends, 'cuda'):
        torch.backends.cuda.matmul.allow_tf32 = True
    # Avoid CPU thread oversubscription
    torch.set_num_threads(max(1, torch.get_num_threads()))
except Exception:
    pass

try:
    # Limit OpenCV threading to avoid contention with Torch threads
    cv2.setNumThreads(1)
except Exception:
    pass

# Performance settings
TARGET_SIZE = 384  # Raised for better recall on thin putter
JPEG_QUALITY = 70  # Reduced quality for faster encoding and smaller payloads
CAPTURE_CROPS = True  # Enable per-point cropping so frontend can place putter images instead of dots
GENERATE_FIRST_FRAME = False  # Skip first annotated frame generation for speed
CACHE_SIZE = MAX_CACHE_SIZE   # Use the defined cache size
ENABLE_STREAMING = True  # Enable streaming responses

# Region of Interest (ROI) around the reference line for putter detection
ENABLE_ROI_CROPPING = True  # Enable/disable ROI cropping for testing
ROI_WIDTH = 120  # Width of the region around the center line (60 pixels on each side)
ROI_START_Y = 50  # Start Y coordinate (matches frontend reference line)
ROI_END_Y = TARGET_SIZE - 50  # End Y coordinate (matches frontend reference line)
ENABLE_ASYNC_POSTPROCESSING = True  # Enable async post-processing to return sooner
FRAME_SKIP_RATIO = 1  # Do not skip frames while stabilizing end logic
MAX_FPS = 15  # Increase to 15 FPS to catch more putts (was 10)
VIDEO_FRAME_SKIP = 1  # For video processing, do not skip frames

# ===== SWING RHYTHM ANALYSIS SYSTEM =====
# Swing timing analysis for rhythm measurement
SWING_STATE_WAITING = "waiting"
SWING_STATE_BACKSWING = "backswing"
SWING_STATE_DOWNSWING = "downswing"
SWING_STATE_COMPLETED = "completed"

# Timing thresholds for swing phases
BACKSWING_MIN_DURATION = 0.1  # Minimum backswing duration (seconds)
DOWNSWING_MIN_DURATION = 0.1  # Minimum downswing duration (seconds)
SWING_TIMEOUT = 3.0  # Maximum swing duration before timeout
STILLNESS_THRESHOLD = 5.0  # Pixels per second for stillness detection
MOTION_THRESHOLD = 10.0  # Pixels per second for motion detection

# Ideal rhythm ratio (2:1 backswing to downswing)
IDEAL_RATIO = 2.0
RHYTHM_TOLERANCE = 0.3  # Acceptable deviation from ideal ratio

# Session configuration
MAX_SWINGS = 10  # Number of swings to analyze
MIN_SWINGS_FOR_ANALYSIS = 3  # Minimum swings needed for analysis

# Optional: persist frames to disk for debugging/verification
SAVE_FRAMES = False
INPUT_SAVE_DIR = 'saved_inputs'
ANNOTATED_SAVE_DIR = 'saved_annotated'
OUTPUT_192_DIR = '192'

try:
    if SAVE_FRAMES:
        os.makedirs(INPUT_SAVE_DIR, exist_ok=True)
        os.makedirs(ANNOTATED_SAVE_DIR, exist_ok=True)
    # Always ensure the '192' directory exists for saving inferenced frames
    os.makedirs(OUTPUT_192_DIR, exist_ok=True)
except Exception:
    pass

# Global cache for processed frames with LRU eviction
frame_cache = {}
cache_access_times = {}  # Track access times for LRU

# Pre-allocated buffers for zero-copy operations
_image_buffer = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
# Use JPEG for annotated outputs for faster encoding
_jpg_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

# Async processing queue 
processing_queue = queue.Queue(maxsize=100)
result_cache = {}

# Performance monitoring
performance_stats = {
    'total_requests': 0,
    'cache_hits': 0,
    'avg_latency': 0.0,
    'min_latency': 0.0,
    'max_latency': 0.0
}

# Optional: in-memory WebSocket clients (not required for core logic)
_ws_clients = {}

# ===== SWING RHYTHM ANALYZER =====

class SwingState(Enum):
    WAITING = "waiting"
    BACKSWING = "backswing"
    DOWNSWING = "downswing"
    COMPLETED = "completed"

@dataclass
class SwingTiming:
    """Timing data for a single swing"""
    backswing_start: float
    backswing_end: float
    downswing_start: float
    downswing_end: float
    backswing_duration: float
    downswing_duration: float
    total_duration: float
    rhythm_ratio: float  # backswing / downswing
    rhythm_score: float  # 0.0 to 1.0, how close to ideal 2:1 ratio
    is_valid: bool
    putter_positions: List[tuple[float, float]]  # Track putter path

@dataclass
class SwingAnalysis:
    """Analysis of a single swing"""
    swing_number: int
    timing: SwingTiming
    confidence: float
    quality_score: float
    rhythm_feedback: str
    improvement_suggestions: List[str]

class SwingSession:
    """Manages a session of swing rhythm analysis"""
    def __init__(self, user_id: Optional[str] = None):
        self.session_id = f"swing_{uuid.uuid4().hex[:8]}"
        self.user_id = user_id or "anonymous"
        self.state = SwingState.WAITING
        self.started_at = datetime.utcnow()
        self.swings: List[SwingAnalysis] = []
        self.current_swing_start_ts: Optional[float] = None
        self.last_seen_ts: Optional[float] = None
        self.frame_index: int = 0
        self.missed_timeout_sec = 0.6
        self.min_swing_duration_sec = 0.2
        self.hard_no_detect_timeout_sec = 0.8
        self.hard_min_duration_sec = 0.15
        self.min_points_for_swing = 3
        
        # Adaptive FPS tracking
        self._last_frame_ts: Optional[float] = None
        self._ema_dt: Optional[float] = None
        
        # Swing phase tracking
        self._swing_phase = SWING_STATE_WAITING
        self._backswing_start_ts: Optional[float] = None
        self._downswing_start_ts: Optional[float] = None
        self._swing_end_ts: Optional[float] = None
        self._last_putter_position: Optional[tuple[float, float]] = None
        self._putter_positions: List[tuple[float, float]] = []
        self._motion_direction: str = "none"  # "backward", "forward", "none"
        self._stillness_frames: int = 0
        self._motion_frames: int = 0
        
        # Warmup frames
        self.warmup_frames = 4
        
        # Store last completed swing path points for UI overlay
        self.last_completed_points: List[tuple[float, float]] = []
        self.last_completed_images: List[str] = []
        
        # Merge-back guard
        self._end_grace_until: Optional[float] = None
        self._end_last_point: Optional[tuple[float, float]] = None

    def observe_detections(self, detections: List[dict], timestamp: float, frame=None):
        """Process detections and update swing state"""
        now = timestamp
        any_detected = len(detections) > 0

        # Update EMA of frame interval for adaptive timeout
        if self._last_frame_ts is not None:
            dt = max(1e-3, now - self._last_frame_ts)
            if self._ema_dt is None:
                self._ema_dt = dt
            else:
                self._ema_dt = 0.9 * self._ema_dt + 0.1 * dt
            approx_timeout = 12.0 * self._ema_dt
            self.missed_timeout_sec = float(min(1.5, max(0.6, approx_timeout)))
        self._last_frame_ts = now

        self.frame_index += 1
        self.last_seen_ts = now

        if any_detected:
            # Get the best detection (highest confidence)
            best_detection = max(detections, key=lambda d: d.get('confidence', 0))
            center_x = best_detection.get('center_x', 0)
            center_y = best_detection.get('center_y', 0)
            current_position = (center_x, center_y)
            
            # Track putter movement
            self._track_putter_movement(current_position, now)
            
            # Update swing state based on movement
            self._update_swing_state(current_position, now)
        else:
            # No detection - handle timeout
            self._handle_no_detection(now)

    def _track_putter_movement(self, current_position: tuple[float, float], timestamp: float):
        """Track putter movement and determine motion direction"""
        if self._last_putter_position is not None:
            # Calculate movement
            dx = current_position[0] - self._last_putter_position[0]
            dy = current_position[1] - self._last_putter_position[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Determine motion direction (assuming top-down view, Y increases downward)
            if abs(dy) > abs(dx):  # Vertical movement dominates
                if dy > 0:  # Moving down (backswing)
                    self._motion_direction = "backward"
                else:  # Moving up (downswing)
                    self._motion_direction = "forward"
            else:  # Horizontal movement
                self._motion_direction = "lateral"
            
            # Track stillness vs motion
            if distance < STILLNESS_THRESHOLD:
                self._stillness_frames += 1
                self._motion_frames = 0
            else:
                self._motion_frames += 1
                self._stillness_frames = 0
        
        self._last_putter_position = current_position
        self._putter_positions.append(current_position)

    def _update_swing_state(self, current_position: tuple[float, float], timestamp: float):
        """Update swing state based on putter movement"""
        if self._swing_phase == SWING_STATE_WAITING:
            # Waiting for swing to start
            if self._motion_frames >= 3:  # Motion detected for 3+ frames
                self._start_swing(timestamp)
        elif self._swing_phase == SWING_STATE_BACKSWING:
            # In backswing phase
            if self._motion_direction == "forward" and self._motion_frames >= 3:
                self._start_downswing(timestamp)
            elif self._stillness_frames >= 10:  # Too much stillness
                self._end_swing(timestamp, "timeout")
        elif self._swing_phase == SWING_STATE_DOWNSWING:
            # In downswing phase
            if self._stillness_frames >= 8:  # Swing completed
                self._end_swing(timestamp, "completed")
            elif timestamp - self._downswing_start_ts > SWING_TIMEOUT:
                self._end_swing(timestamp, "timeout")

    def _start_swing(self, timestamp: float):
        """Start a new swing"""
        self._swing_phase = SWING_STATE_BACKSWING
        self._backswing_start_ts = timestamp
        self.current_swing_start_ts = timestamp
        self.state = SwingState.BACKSWING
        self._putter_positions = []
        logger.info(f"Swing started at {timestamp}")

    def _start_downswing(self, timestamp: float):
        """Transition from backswing to downswing"""
        self._swing_phase = SWING_STATE_DOWNSWING
        self._downswing_start_ts = timestamp
        self.state = SwingState.DOWNSWING
        logger.info(f"Downswing started at {timestamp}")

    def _end_swing(self, timestamp: float, reason: str):
        """End the current swing"""
        if self._swing_phase == SWING_STATE_WAITING:
            return
        
        self._swing_end_ts = timestamp
        self._swing_phase = SWING_STATE_COMPLETED
        
        # Calculate timing
        timing = self._calculate_timing()
        
        # Analyze the swing
        analysis = self._analyze_swing(timing)
        
        # Add to session
        self.swings.append(analysis)
        self.last_completed_points = self._putter_positions.copy()
        
        # Update session state
        if len(self.swings) >= MAX_SWINGS:
            self.state = SwingState.COMPLETED
        else:
            self.state = SwingState.WAITING
            self._reset_swing_state()
        
        logger.info(f"Swing {len(self.swings)} completed: {reason}")

    def _calculate_timing(self) -> SwingTiming:
        """Calculate timing data for the current swing"""
        if not self._backswing_start_ts or not self._downswing_start_ts or not self._swing_end_ts:
            # Invalid timing data
            return SwingTiming(
                backswing_start=0, backswing_end=0, downswing_start=0, downswing_end=0,
                backswing_duration=0, downswing_duration=0, total_duration=0,
                rhythm_ratio=0, rhythm_score=0, is_valid=False,
                putter_positions=self._putter_positions
            )
        
        backswing_duration = self._downswing_start_ts - self._backswing_start_ts
        downswing_duration = self._swing_end_ts - self._downswing_start_ts
        total_duration = self._swing_end_ts - self._backswing_start_ts
        
        # Calculate rhythm ratio
        if downswing_duration > 0:
            rhythm_ratio = backswing_duration / downswing_duration
        else:
            rhythm_ratio = 0
        
        # Calculate rhythm score (how close to ideal 2:1 ratio)
        if rhythm_ratio > 0:
            ideal_ratio = IDEAL_RATIO
            deviation = abs(rhythm_ratio - ideal_ratio) / ideal_ratio
            rhythm_score = max(0, 1 - deviation / RHYTHM_TOLERANCE)
        else:
            rhythm_score = 0
        
        # Validate timing
        is_valid = (
            backswing_duration >= BACKSWING_MIN_DURATION and
            downswing_duration >= DOWNSWING_MIN_DURATION and
            total_duration >= self.min_swing_duration_sec and
            len(self._putter_positions) >= self.min_points_for_swing
        )
        
        return SwingTiming(
            backswing_start=self._backswing_start_ts,
            backswing_end=self._downswing_start_ts,
            downswing_start=self._downswing_start_ts,
            downswing_end=self._swing_end_ts,
            backswing_duration=backswing_duration,
            downswing_duration=downswing_duration,
            total_duration=total_duration,
            rhythm_ratio=rhythm_ratio,
            rhythm_score=rhythm_score,
            is_valid=is_valid,
            putter_positions=self._putter_positions
        )

    def _analyze_swing(self, timing: SwingTiming) -> SwingAnalysis:
        """Analyze a completed swing"""
        swing_number = len(self.swings) + 1
        
        # Calculate quality score
        quality_score = timing.rhythm_score * 0.7 + (1.0 if timing.is_valid else 0.0) * 0.3
        
        # Generate feedback
        rhythm_feedback = self._generate_rhythm_feedback(timing)
        improvement_suggestions = self._generate_improvement_suggestions(timing)
        
        return SwingAnalysis(
            swing_number=swing_number,
            timing=timing,
            confidence=quality_score,
            quality_score=quality_score,
            rhythm_feedback=rhythm_feedback,
            improvement_suggestions=improvement_suggestions
        )

    def _generate_rhythm_feedback(self, timing: SwingTiming) -> str:
        """Generate rhythm feedback for the swing"""
        if not timing.is_valid:
            return "Invalid swing - please try again"
        
        ratio = timing.rhythm_ratio
        if ratio < 1.5:
            return "Too fast on backswing - slow down your takeaway"
        elif ratio > 2.5:
            return "Too slow on backswing - speed up your takeaway"
        elif 1.8 <= ratio <= 2.2:
            return "Excellent rhythm! Perfect 2:1 ratio"
        else:
            return "Good rhythm, minor adjustments needed"

    def _generate_improvement_suggestions(self, timing: SwingTiming) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        if not timing.is_valid:
            suggestions.append("Ensure smooth, continuous motion")
            suggestions.append("Avoid pausing during swing")
            return suggestions
        
        ratio = timing.rhythm_ratio
        if ratio < 1.5:
            suggestions.append("Practice slower backswing")
            suggestions.append("Count 'one-two' for backswing, 'three' for downswing")
        elif ratio > 2.5:
            suggestions.append("Practice faster backswing")
            suggestions.append("Try a more compact backswing")
        elif 1.8 <= ratio <= 2.2:
            suggestions.append("Maintain this rhythm")
            suggestions.append("Focus on consistency")
        else:
            suggestions.append("Fine-tune your timing")
            suggestions.append("Practice with a metronome")
        
        return suggestions

    def _reset_swing_state(self):
        """Reset state for next swing"""
        self._swing_phase = SWING_STATE_WAITING
        self._backswing_start_ts = None
        self._downswing_start_ts = None
        self._swing_end_ts = None
        self._last_putter_position = None
        self._putter_positions = []
        self._motion_direction = "none"
        self._stillness_frames = 0
        self._motion_frames = 0

    def _handle_no_detection(self, timestamp: float):
        """Handle case when no putter is detected"""
        if self._swing_phase != SWING_STATE_WAITING:
            # We're in the middle of a swing but lost detection
            if timestamp - self.last_seen_ts > self.missed_timeout_sec:
                self._end_swing(timestamp, "lost_detection")

    def get_session_summary(self) -> dict:
        """Get summary of the current session"""
        if not self.swings:
            return {
                "total_swings": 0,
                "average_rhythm_ratio": 0,
                "average_rhythm_score": 0,
                "best_swing": None,
                "needs_improvement": []
            }
        
        valid_swings = [s for s in self.swings if s.timing.is_valid]
        if not valid_swings:
            return {
                "total_swings": len(self.swings),
                "average_rhythm_ratio": 0,
                "average_rhythm_score": 0,
                "best_swing": None,
                "needs_improvement": []
            }
        
        # Calculate averages
        avg_rhythm_ratio = sum(s.timing.rhythm_ratio for s in valid_swings) / len(valid_swings)
        avg_rhythm_score = sum(s.timing.rhythm_score for s in valid_swings) / len(valid_swings)
        
        # Find best swing
        best_swing = max(valid_swings, key=lambda s: s.timing.rhythm_score)
        
        # Find swings needing improvement
        needs_improvement = [s for s in valid_swings if s.timing.rhythm_score < 0.7]
        
        return {
            "total_swings": len(self.swings),
            "valid_swings": len(valid_swings),
            "average_rhythm_ratio": avg_rhythm_ratio,
            "average_rhythm_score": avg_rhythm_score,
            "best_swing": best_swing.__dict__ if best_swing else None,
            "needs_improvement": [s.__dict__ for s in needs_improvement]
        }

# Global session storage
swing_sessions = {}

# Request models
class StartSwingSessionReq(BaseModel):
    user_id: str

# ===== DETECTION AND INFERENCE =====

def detect_putter_optimized(image: np.ndarray, conf_threshold: float = 0.3) -> List[dict]:
    """Optimized putter detection using YOLO model"""
    global model
    
    if model is None:
        # Return dummy detection for testing
        return [{
            'center_x': TARGET_SIZE // 2,
            'center_y': TARGET_SIZE // 2,
            'width': 20,
            'height': 20,
            'confidence': 0.8,
            'class': 'putter'
        }]
    
    try:
        # Run inference
        results = model(image, conf=conf_threshold, verbose=False)
        
        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Calculate center and dimensions
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    detections.append({
                        'center_x': float(center_x),
                        'center_y': float(center_y),
                        'width': float(width),
                        'height': float(height),
                        'confidence': float(conf),
                        'class': cls
                    })
        
        return detections
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        return []

def process_frame_swing(image: np.ndarray, session_id: str) -> dict:
    """Process a single frame for swing analysis"""
    try:
        # Resize image to target size
        resized = cv2.resize(image, (TARGET_SIZE, TARGET_SIZE))
        
        # Run detection
        detections = detect_putter_optimized(resized)
        
        # Get session
        session = swing_sessions.get(session_id)
        if not session:
            return {"error": "Session not found", "success": False}
        
        # Process detections
        timestamp = time.time()
        session.observe_detections(detections, timestamp, resized)
        
        # Prepare response
        response = {
            "session_id": session_id,
            "state": session.state.value,
            "swings": [s.__dict__ for s in session.swings],
            "swing_completed": False,
            "analysis_size": {"w": TARGET_SIZE, "h": TARGET_SIZE},
            "path_points": session.last_completed_points,
            "success": True
        }
        
        # Check if swing just completed
        if session.state == SwingState.COMPLETED and len(session.swings) > 0:
            response["swing_completed"] = True
            response["path_points"] = session.last_completed_points
        
        # Add session summary
        response.update(session.get_session_summary())
        
        return response
        
    except Exception as e:
        logger.error(f"Frame processing error: {str(e)}")
        return {"error": str(e), "success": False}

# ===== API ENDPOINTS =====

@app.post("/swing/start")
async def swing_start(req: StartSwingSessionReq):
    """Start a new swing rhythm analysis session"""
    try:
        session = SwingSession(user_id=req.user_id)
        swing_sessions[session.session_id] = session
        logger.info(f"Started new swing session: {session.session_id}")
        return {"session_id": session.session_id, "state": session.state.value}
    except Exception as e:
        logger.error(f"Error starting swing session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start session")

@app.post("/swing/frame")
async def swing_frame(file: UploadFile = File(...), session_id: str = Form(...)):
    """Process a frame for swing rhythm analysis"""
    try:
        # Validate session exists
        session = swing_sessions.get(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return {"error": "not_found", "success": False}
        
        # Frame skipping for performance
        session.frame_count = getattr(session, 'frame_count', 0) + 1
        if session.frame_count > getattr(session, 'warmup_frames', 0) and (session.frame_count % FRAME_SKIP_RATIO != 0):
            return {
                "session_id": session_id,
                "state": session.state.value,
                "swings": [s.__dict__ for s in session.swings],
                "swing_completed": False,
                "analysis_size": {"w": TARGET_SIZE, "h": TARGET_SIZE},
                "path_points": [],
                "success": True,
                "skipped": True,
                **session.get_session_summary()
            }

        # Validate file
        if not file.content_type or not file.content_type.startswith('image/'):
            logger.warning(f"Invalid file type: {file.content_type}")
            return {"error": "Invalid file type", "success": False}

        content = await file.read()
        if len(content) == 0:
            logger.warning("Empty file received")
            return {"error": "Empty file", "success": False}

        # Process frame
        content_hash = hashlib.md5(content).hexdigest()
        
        # Check cache first
        if content_hash in frame_cache:
            cached_result = frame_cache[content_hash].copy()
            cached_result["session_id"] = session_id
            performance_stats['cache_hits'] += 1
            return cached_result

        # Decode image
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.warning("Failed to decode image")
            return {"error": "Failed to decode image", "success": False}

        # Process frame
        start_time = time.time()
        result = process_frame_swing(image, session_id)
        processing_time = time.time() - start_time
        
        # Update performance stats
        performance_stats['total_requests'] += 1
        if performance_stats['avg_latency'] == 0:
            performance_stats['avg_latency'] = processing_time
        else:
            performance_stats['avg_latency'] = 0.9 * performance_stats['avg_latency'] + 0.1 * processing_time
        
        performance_stats['min_latency'] = min(performance_stats['min_latency'], processing_time) if performance_stats['min_latency'] > 0 else processing_time
        performance_stats['max_latency'] = max(performance_stats['max_latency'], processing_time)
        
        # Cache result
        if len(frame_cache) < CACHE_SIZE:
            frame_cache[content_hash] = result.copy()
            cache_access_times[content_hash] = time.time()
        else:
            # LRU eviction
            oldest_key = min(cache_access_times.keys(), key=lambda k: cache_access_times[k])
            del frame_cache[oldest_key]
            del cache_access_times[oldest_key]
            frame_cache[content_hash] = result.copy()
            cache_access_times[content_hash] = time.time()

        return result

    except Exception as e:
        logger.error(f"Frame processing error: {str(e)}")
        return {"error": str(e), "success": False}

@app.get("/swing/status/{session_id}")
async def swing_status(session_id: str):
    """Get status of a swing session"""
    session = swing_sessions.get(session_id)
    if not session:
        return {"error": "Session not found", "success": False}
    
    return {
        "session_id": session_id,
        "state": session.state.value,
        "swings": [s.__dict__ for s in session.swings],
        "analysis_size": {"w": TARGET_SIZE, "h": TARGET_SIZE},
        "success": True,
        **session.get_session_summary()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if model is loaded
        model_status = "loaded" if model is not None else "not_loaded"
        
        # Check system resources
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        return JSONResponse(content={
            "status": "healthy",
            "model_status": model_status,
            "device": device,
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "active_sessions": len(swing_sessions),
            "performance_stats": performance_stats
        })
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.on_event("startup")
def load_model():
    """Load the YOLO model on startup"""
    global model
    global onnx_session, onnx_input_name, onnx_output_names
    try:
        # Try to initialize ONNX Runtime if best.onnx exists
        try:
            onnx_path = 'best.onnx'
            if os.path.exists(onnx_path):
                import onnxruntime as ort
                if device == 'cuda':
                    providers = [('CUDAExecutionProvider', {'do_copy_in_default_stream': True})]
                else:
                    providers = ['CPUExecutionProvider']
                onnx_session = ort.InferenceSession(onnx_path, providers=providers)
                onnx_input_name = onnx_session.get_inputs()[0].name
                onnx_output_names = [o.name for o in onnx_session.get_outputs()]
                print(f"Initialized ONNX Runtime session on {device} with providers={providers}")
        except Exception as e:
            print(f"ONNX Runtime initialization skipped/failed: {e}")

        # Try to load custom model
        try:
            model = YOLO('best.pt', task='detect')
            print("Loaded custom model: best.pt")
        except Exception as e1:
            print(f"First attempt failed: {e1}")
            try:
                model = YOLO('best.pt')
                print("Loaded custom model: best.pt (second attempt)")
            except Exception as e2:
                print(f"Second attempt failed: {e2}")
                try:
                    model = YOLO('best.pt')
                    print("Loaded custom model: best.pt (third attempt)")
                except Exception as e3:
                    print(f"All attempts failed. Last error: {e3}")
                    print("Falling back to standard YOLOv8n model...")
                    model = YOLO('yolov8n.pt')
                    print("Loaded standard YOLOv8n model")
        
        model.to(device)
        
        # Fuse layers for faster inference
        try:
            model.fuse()
            print("Model layers fused for faster inference")
        except Exception:
            print("Model fusion not supported; continuing without fusion")
        
        # Warmup
        dummy_img = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
        for _ in range(3):
            _ = model(dummy_img, conf=0.3, verbose=False)
        
        logger.info(f"Model loaded on {device} with optimizations enabled")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.info("Starting in test mode without model...")
        model = None

# ===== WebSocket endpoint for real-time communication =====

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time swing analysis"""
    await websocket.accept()
    client_id = None
    
    try:
        while True:
            # Receive message
            data = await websocket.receive()
            
            if data["type"] == "websocket.receive":
                if "bytes" in data:
                    # Binary data (image frame)
                    if client_id:
                        # Process frame
                        result = await _process_frame_bytes_ws(client_id, data["bytes"])
                        await websocket.send_text(json.dumps(result))
                elif "text" in data:
                    # Text data (JSON commands)
                    message = json.loads(data["text"])
                    
                    if message.get("type") == "start":
                        # Start new session
                        session = SwingSession(user_id=message.get("user_id", "anonymous"))
                        swing_sessions[session.session_id] = session
                        client_id = session.session_id
                        await websocket.send_text(json.dumps({
                            "type": "started",
                            "session_id": session.session_id,
                            "state": session.state.value
                        }))
                    elif message.get("type") == "ping":
                        # Ping/pong for RTT monitoring
                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "t": message.get("t")
                        }))
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        if client_id:
            # Clean up session
            swing_sessions.pop(client_id, None)

async def _process_frame_bytes_ws(session_id: str, content: bytes) -> dict:
    """Process frame bytes for WebSocket"""
    session = swing_sessions.get(session_id)
    if not session:
        return {"error": "not_found", "success": False}
    if not content:
        return {"error": "empty", "success": False}

    try:
        # Decode image
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"error": "Failed to decode image", "success": False}

        # Process frame
        result = process_frame_swing(image, session_id)
        return result

    except Exception as e:
        logger.error(f"WebSocket frame processing error: {str(e)}")
        return {"error": str(e), "success": False}

# ===== Additional endpoints =====

@app.get("/performance")
async def get_performance_stats():
    """Get performance statistics"""
    return {
        "performance_stats": performance_stats,
        "cache_size": len(frame_cache),
        "active_sessions": len(swing_sessions)
    }

@app.get("/debug_sessions")
async def debug_sessions():
    """Debug endpoint to see all active sessions"""
    return {
        "sessions": {
            sid: {
                "user_id": sess.user_id,
                "state": sess.state.value,
                "swing_count": len(sess.swings),
                "started_at": sess.started_at.isoformat()
            }
            for sid, sess in swing_sessions.items()
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
