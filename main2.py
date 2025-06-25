import cv2
import numpy as np
import math
from gaze_tracking import GazeTracking
import time

# Initialize gaze tracking
gaze = GazeTracking()

# Load face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Initialize facial landmark detector
LANDMARK_MODEL = "lbfmodel.yaml"
landmark_detector = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LANDMARK_MODEL)

# 3D model points for head pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

# Constants
HEAD_POSE_THRESHOLD = 25  # Degrees threshold for head orientation
MIN_FACE_SIZE = (50, 50)  # Minimum face size for detection
DETECTION_SCALE = 1.1     # Scale factor for face detection
MIN_NEIGHBORS = 5         # Minimum neighbors for face detection

def rotation_matrix_to_euler_angles(R):
    """Convert rotation matrix to Euler angles (pitch, yaw, roll)"""
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
        
    return np.array([math.degrees(x), math.degrees(y), math.degrees(z)])

def pose_estimate(img, landmarks):
    """Estimate head pose using facial landmarks"""
    size = img.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    
    dist_coeffs = np.zeros((4, 1))  # No lens distortion
    
    # Extract relevant landmarks with bounds checking
    landmark_indices = [30, 8, 36, 45, 48, 54]
    image_points = []
    
    for idx in landmark_indices:
        if idx < len(landmarks):
            image_points.append(landmarks[idx])
        else:
            # Use a default position if landmark is missing
            image_points.append((center[0], center[1]))
    
    image_points = np.array(image_points, dtype="double")

    # Solve PnP to get rotation and translation vectors
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, 
        image_points, 
        camera_matrix, 
        dist_coeffs, 
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return False, img, (0, 0, 0)
    
    # Convert rotation vector to matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Get Euler angles
    pitch, yaw, roll = rotation_matrix_to_euler_angles(rotation_matrix)
    
    # Determine if head is facing forward
    head_forward = abs(yaw) < HEAD_POSE_THRESHOLD and abs(pitch) < HEAD_POSE_THRESHOLD
    
    # Draw head pose direction (only if points are within image bounds)
    try:
        nose_end_point2D, _ = cv2.projectPoints(
            np.array([(0.0, 0.0, 1000.0)]), 
            rotation_vector, 
            translation_vector, 
            camera_matrix, 
            dist_coeffs
        )
        
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        
        # Check if points are within image boundaries
        height, width = img.shape[:2]
        if (0 <= p1[0] < width and 0 <= p1[1] < height and 
            0 <= p2[0] < width and 0 <= p2[1] < height):
            cv2.line(img, p1, p2, (0, 255, 255), 2)
    except:
        pass
    
    return head_forward, img, (pitch, yaw, roll)

def detect_faces(img):
    """Detect faces in the image with optimized parameters"""
    if img is None or img.size == 0:
        return [], None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve contrast
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=DETECTION_SCALE,
        minNeighbors=MIN_NEIGHBORS,
        minSize=MIN_FACE_SIZE,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    return faces, gray

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize variables for attention tracking
    attention_history = []
    attention_window = 10  # Track attention over last N frames
    
    while True:
        ret, img = cap.read()
        if not ret:
            print("Failed to capture frame")
            time.sleep(0.1)
            continue
            
        # Validate frame
        if img is None or img.size == 0:
            print("Invalid frame received")
            time.sleep(0.1)
            continue
            
        # Flip frame horizontally for more natural view
        img = cv2.flip(img, 1)
        
        # Process gaze tracking
        gaze.refresh(img)
        gaze_attention = gaze.is_center()
        
        # Detect faces
        faces, gray = detect_faces(img)
        head_forward = False
        head_pose = (0, 0, 0)
        landmarks_detected = False
        
        if len(faces) > 0:
            # Process only the largest face
            main_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = main_face
            
            # Ensure face coordinates are within image bounds
            height, width = img.shape[:2]
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = max(10, min(w, width - x))
            h = max(10, min(h, height - y))
            
            face_roi = gray[y:y+h, x:x+w]
            
            # Only process if ROI is valid
            if face_roi.size > 0:
                try:
                    # Detect facial landmarks
                    _, landmarks = landmark_detector.fit(face_roi, [np.array([x, y, w, h])])
                    if landmarks and len(landmarks) > 0 and len(landmarks[0]) > 0:
                        landmarks = landmarks[0][0]
                        landmarks_detected = True
                        
                        # Adjust landmarks to full image coordinates
                        landmarks = [(x + int(pt[0]), (y + int(pt[1]))) for pt in landmarks]
                        
                        # Estimate head pose
                        head_forward, img, head_pose = pose_estimate(img, landmarks)
                        
                        # Draw landmarks
                        for pt in landmarks:
                            # Check if point is within image boundaries
                            if 0 <= pt[0] < width and 0 <= pt[1] < height:
                                cv2.circle(img, pt, 2, (0, 255, 255), -1)
                except Exception as e:
                    print(f"Landmark detection error: {e}")
        
        # Determine attention level
        attention_level = "Distracted"
        if landmarks_detected:
            if head_forward and gaze_attention:
                attention_level = "Focused"
            elif head_forward or gaze_attention:
                attention_level = "Partially Focused"
        
        # Update attention history
        attention_history.append(1 if attention_level == "Focused" else 0)
        if len(attention_history) > attention_window:
            attention_history.pop(0)
        
        # Calculate attention score
        attention_score = sum(attention_history) / len(attention_history) if attention_history else 0
        
        # Draw face bounding box (with boundary checks)
        for (x, y, w, h) in faces:
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x, width - 1))
            y1 = max(0, min(y, height - 1))
            x2 = max(0, min(x + w, width - 1))
            y2 = max(0, min(y + h, height - 1))
            
            if x2 > x1 and y2 > y1:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, attention_level, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (0, 255, 0) if attention_level == "Focused" else (0, 0, 255), 2)
        
        # Display metrics
        pitch, yaw, roll = head_pose
        cv2.putText(img, f"Head Pose: Yaw:{yaw:.1f}, Pitch:{pitch:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        cv2.putText(img, f"Gaze: {'Center' if gaze_attention else 'Away'}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img, f"Attention Score: {attention_score:.1f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Validate image before display
        if img is not None and img.size > 0 and img.shape[0] > 0 and img.shape[1] > 0:
            try:
                cv2.imshow('Attention Monitor', img)
            except cv2.error as e:
                print(f"OpenCV display error: {e}")
                break
        
        # Exit on ESC
        key = cv2.waitKey(10)
        if key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()