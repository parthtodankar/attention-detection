import streamlit as st
import cv2
import numpy as np
import math
import time
from threading import Thread
import queue
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from gaze_tracking.gaze_tracking import GazeTracking


# Try to import gaze_tracking, handle if not available
try:
    from gaze_tracking.gaze_tracking import GazeTracking
    GAZE_TRACKING_AVAILABLE = True
except ImportError:
    GAZE_TRACKING_AVAILABLE = False
    st.warning("⚠️ gaze_tracking module not found. Make sure the local folder exists.")


# Page configuration
st.set_page_config(
    page_title="Attention Detection System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .status-card {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-weight: bold;
        font-size: 1.2em;
    }
    
    .focused {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    
    .somewhat-focused {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffeaa7;
    }
    
    .distracted {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class AttentionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize gaze tracking if available
        if GAZE_TRACKING_AVAILABLE:
            self.gaze = GazeTracking()
        
        # Try to load facial landmark model
        self.landmark_detector = None
        try:
            self.landmark_detector = cv2.face.createFacemarkLBF()
            # You'll need to provide the path to your model file
            # self.landmark_detector.loadModel("lbfmodel.yaml")
        except:
            st.warning("⚠️ Facial landmark model not loaded. Some features may not work.")
        
        # 3D model points for pose estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
        # Statistics tracking
        self.attention_history = []
        self.session_start = datetime.now()
    
    def detect_faces(self, img):
        """Detect faces in the image"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=2,
            minSize=(20, 20)
        )
        return faces, gray
    
    def pose_estimate(self, img, landmarks):
        """Estimate head pose from facial landmarks"""
        if self.landmark_detector is None or len(landmarks) == 0:
            return False
        
        try:
            size = img.shape
            focal_length = size[1]
            center = (size[1] / 2, size[0] / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")

            shape = np.array(landmarks, dtype=np.float32).astype(np.uint)

            image_points = np.array([
                shape[0][0][30],  # Nose tip
                shape[0][0][8],   # Chin
                shape[0][0][36],  # Left eye left corner
                shape[0][0][45],  # Right eye right corner
                shape[0][0][48],  # Left Mouth corner
                shape[0][0][54]   # Right mouth corner
            ], dtype="double")
            
            dist_coeffs = np.zeros((4, 1))
            (success, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points, image_points, camera_matrix, dist_coeffs, 
                flags=cv2.SOLVEPNP_UPNP
            )

            (nose_end_point2D, jacobian) = cv2.projectPoints(
                np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                translation_vector, camera_matrix, dist_coeffs
            )

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            try:
                m = (p2[1] - p1[1]) / (p2[0] - p1[0])
                ang = int(math.degrees(math.atan(m)))
            except:
                ang = 90

            # Draw pose line
            cv2.line(img, p1, p2, (0, 255, 255), 2)
            
            # Return True if head is facing forward (within 45 degrees)
            return -45 < ang < 45
            
        except Exception as e:
            st.error(f"Pose estimation error: {str(e)}")
            return False
    
    def analyze_attention(self, img):
        """Main function to analyze attention level"""
        faces, gray = self.detect_faces(img)
        
        if len(faces) == 0:
            return img, "No Face Detected", 0
        
        attention_score = 0
        gaze_focused = False
        pose_good = False
        
        # Gaze tracking
        if GAZE_TRACKING_AVAILABLE:
            self.gaze.refresh(img)
            img = self.gaze.annotated_frame()
            gaze_focused = self.gaze.is_center()
            if gaze_focused:
                attention_score += 50
        
        # Pose estimation
        try:
            if self.landmark_detector:
                _, landmarks = self.landmark_detector.fit(gray, faces)
                pose_good = self.pose_estimate(img, landmarks)
                if pose_good:
                    attention_score += 50
        except Exception as e:
            pass
        
        # Draw face rectangles and status
        for (x, y, w, h) in faces:
            if attention_score >= 80:
                color = (0, 255, 0)  # Green
                status = "Definitely Focused"
            elif attention_score >= 40:
                color = (0, 255, 255)  # Yellow
                status = "Somewhat Focused"
            else:
                color = (0, 0, 255)  # Red
                status = "Distracted"
            
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add timestamp to history
        self.attention_history.append({
            'timestamp': datetime.now(),
            'attention_score': attention_score,
            'status': status,
            'gaze_focused': gaze_focused,
            'pose_good': pose_good
        })
        
        # Keep only last 100 records
        if len(self.attention_history) > 100:
            self.attention_history = self.attention_history[-100:]
        
        return img, status, attention_score

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>👁️ Attention Detection System</h1>
        <p>Real-time monitoring of focus and attention levels</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize detector
    if 'detector' not in st.session_state:
        st.session_state.detector = AttentionDetector()
    
    detector = st.session_state.detector
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    # Camera settings
    camera_source = st.sidebar.selectbox("Camera Source", [0, 1, 2], index=0)
    
    # Detection settings
    st.sidebar.subheader("Detection Settings")
    show_landmarks = st.sidebar.checkbox("Show Facial Landmarks", value=False)
    show_gaze = st.sidebar.checkbox("Show Gaze Tracking", value=True)
    sensitivity = st.sidebar.slider("Detection Sensitivity", 0.1, 2.0, 1.0, 0.1)
    
    # Session controls
    st.sidebar.subheader("Session Controls")
    if st.sidebar.button("🔄 Reset Session"):
        detector.attention_history = []
        detector.session_start = datetime.now()
        st.sidebar.success("Session reset!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Feed")
        
        # Placeholders for video and status
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Start/Stop buttons
        start_col, stop_col = st.columns(2)
        
        with start_col:
            start_detection = st.button("▶️ Start Detection", type="primary")
        
        with stop_col:
            stop_detection = st.button("⏹️ Stop Detection")
        
        # Video capture logic
        if start_detection:
            st.session_state.running = True
        
        if stop_detection:
            st.session_state.running = False
        
        # Initialize running state
        if 'running' not in st.session_state:
            st.session_state.running = False
        
        # Main detection loop
        if st.session_state.running:
            try:
                cap = cv2.VideoCapture(camera_source)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                if not cap.isOpened():
                    st.error("❌ Could not open camera. Please check camera connection.")
                    st.session_state.running = False
                else:
                    frame_count = 0
                    
                    while st.session_state.running:
                        ret, frame = cap.read()
                        
                        if not ret:
                            st.error("❌ Failed to read from camera.")
                            break
                        
                        # Process every 3rd frame for performance
                        if frame_count % 3 == 0:
                            processed_frame, status, score = detector.analyze_attention(frame)
                            
                            # Convert BGR to RGB for display
                            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            
                            # Update video display
                            video_placeholder.image(processed_frame, channels="RGB", width=600)
                            
                            # Update status display
                            if score >= 80:
                                status_class = "focused"
                                emoji = "🎯"
                            elif score >= 40:
                                status_class = "somewhat-focused"
                                emoji = "⚠️"
                            else:
                                status_class = "distracted"
                                emoji = "❌"
                            
                            status_placeholder.markdown(f"""
                            <div class="status-card {status_class}">
                                {emoji} {status} (Score: {score}/100)
                            </div>
                            """, unsafe_allow_html=True)
                        
                        frame_count += 1
                        time.sleep(0.1)  # Control frame rate
                
                cap.release()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.running = False
    
    with col2:
        st.subheader("📊 Analytics")
        
        if detector.attention_history:
            # Calculate session statistics
            df = pd.DataFrame(detector.attention_history)
            
            # Current session duration
            session_duration = datetime.now() - detector.session_start
            
            # Statistics
            avg_attention = df['attention_score'].mean()
            focused_time = len(df[df['attention_score'] >= 80])
            total_readings = len(df)
            focused_percentage = (focused_time / total_readings * 100) if total_readings > 0 else 0
            
            # Display metrics
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Session Duration", f"{session_duration.seconds // 60}m {session_duration.seconds % 60}s")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Average Attention", f"{avg_attention:.1f}/100")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Focused Time", f"{focused_percentage:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Attention over time chart
            if len(df) > 1:
                st.subheader("📈 Attention Timeline")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['attention_score'],
                    mode='lines+markers',
                    name='Attention Score',
                    line=dict(color='blue', width=2),
                    fill='tonexty'
                ))
                
                fig.add_hline(y=80, line_dash="dash", line_color="green", 
                             annotation_text="Focused Threshold")
                fig.add_hline(y=40, line_dash="dash", line_color="orange", 
                             annotation_text="Somewhat Focused Threshold")
                
                fig.update_layout(
                    title="Attention Score Over Time",
                    xaxis_title="Time",
                    yaxis_title="Attention Score",
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Status distribution
            if len(df) > 5:
                st.subheader("📊 Status Distribution")
                status_counts = df['status'].value_counts()
                
                fig_pie = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    color_discrete_map={
                        'Definitely Focused': '#28a745',
                        'Somewhat Focused': '#ffc107',
                        'Distracted': '#dc3545'
                    }
                )
                fig_pie.update_layout(height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Export data
            if st.button("📥 Download Session Data"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"attention_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        else:
            st.info("📊 Start detection to see analytics")
            st.markdown("""
            **Features:**
            - Real-time attention scoring
            - Session duration tracking
            - Attention timeline charts
            - Status distribution analysis
            - Data export capabilities
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>💡 <strong>Tips:</strong> Ensure good lighting and position your face clearly in the camera frame for best results.</p>
        <p>🔧 Built with Streamlit • OpenCV • Computer Vision</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()