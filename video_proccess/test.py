import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

class EnhancedVideoProcessor:
    def __init__(self, yolo_model_path, width=640, detect_classes=None):
        self.width = width
        self.model = YOLO(yolo_model_path)
        # Default classes to detect (persons, cars, bikes, etc.)
        self.detect_classes = detect_classes or [0, 1, 2, 3, 5, 7]  
        
    def resize_frame(self, frame):
        """Resize frame maintaining aspect ratio"""
        height = int((self.width / frame.shape[1]) * frame.shape[0])
        return cv2.resize(frame, (self.width, height))
    
    def detect_objects(self, frame, conf_threshold=0.35):
        """Enhanced object detection with confidence tracking"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb, conf=conf_threshold)
        return results[0]
    
    def estimate_background(self, frames, conf_threshold=0.35):
        """Improved and memory-efficient background estimation"""
        print("Estimating clean background...")
        
        # Initialize accumulators for background estimation
        acc_frame = np.zeros_like(frames[0], dtype=np.float64)
        acc_weight = np.zeros_like(frames[0], dtype=np.float64)
        
        for frame in tqdm(frames, desc="Analyzing frames"):
            # Detect objects
            detections = self.detect_objects(frame, conf_threshold)
            
            # Create detection mask
            object_mask = self.create_object_mask(frame, detections)
            clean_mask = (1 - object_mask / 255.0)
            clean_mask = np.stack([clean_mask] * 3, axis=-1)
            
            # Accumulate clean areas
            acc_frame += frame.astype(np.float64) * clean_mask
            acc_weight += clean_mask
        
        # Avoid division by zero
        acc_weight[acc_weight == 0] = 1
        
        # Calculate background as weighted average
        background = (acc_frame / acc_weight).astype(np.uint8)
        
        return background

    def create_object_mask(self, frame, detections, mask_expansion=30):
        """Create enhanced mask for detected objects - Fixed version"""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        for box in detections.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            
            if cls in self.detect_classes and conf > 0.35:
                # Get coordinates and expand the box
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                
                # Calculate dynamic expansion based on object size
                width = x_max - x_min
                height = y_max - y_min
                expansion = int(mask_expansion * (width + height) / 200)
                
                # Apply expanded boundaries
                x_min = max(0, x_min - expansion)
                y_min = max(0, y_min - expansion)
                x_max = min(frame.shape[1], x_max + expansion)
                y_max = min(frame.shape[0], y_max + expansion)
                
                # Draw filled rectangle with soft edges
                cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)
        
        # Smooth the mask
        if np.any(mask > 0):  # Only process if mask contains objects
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return mask

    def create_gradient_mask(self, mask, kernel_size=21):
        """Create improved gradient mask - Fixed version"""
        if np.any(mask > 0):  # Only process if mask contains objects
            gradient_mask = cv2.GaussianBlur(mask.astype(np.float32), 
                                        (kernel_size, kernel_size), 0)
            gradient_mask = gradient_mask / 255.0  # Normalize to [0, 1]
        else:
            gradient_mask = np.zeros_like(mask, dtype=np.float32)
        
        return gradient_mask
    
    def blend_frames(self, frame, background, mask, edges=None):
        """Enhanced frame blending with edge preservation - Fixed version"""
        # Ensure mask is in correct format
        gradient_mask = self.create_gradient_mask(mask)
        
        # If we have edge information, use it to preserve static elements
        if edges is not None:
            edge_mask = (edges > 0).astype(np.float32)
            gradient_mask = gradient_mask * (1 - edge_mask)
        
        # Ensure the gradient mask is 3D for proper broadcasting
        gradient_mask_3d = np.stack([gradient_mask] * 3, axis=-1)
        
        # Perform blending using numpy operations
        blended = (1 - gradient_mask_3d) * frame + gradient_mask_3d * background
        
        return blended.astype(np.uint8)

    def result(self, output_dir, processed_frames, background_frame):
        """
        Analyze all processed frames and select the best result with final verification
        Returns the most clean frame with minimal detected objects
        """
        print("Performing final verification and selecting best result...")
        
        best_score = float('inf')
        best_frame = None
        best_verification = None
        
        # Check both processed frames and background
        all_frames = processed_frames + [background_frame]
        
        for frame in tqdm(all_frames, desc="Verifying results"):
            # Perform final YOLO detection with high confidence
            detections = self.detect_objects(frame, conf_threshold=0.4)
            
            # Create verification mask
            verification_mask = self.create_object_mask(frame, detections, mask_expansion=20)
            
            # Calculate score based on multiple factors
            num_detections = len(detections.boxes)
            detection_area = np.sum(verification_mask > 0) / verification_mask.size
            total_conf = sum(float(box.conf) for box in detections.boxes)
            
            # Combined score (lower is better)
            score = num_detections * detection_area * total_conf
            
            if score < best_score:
                best_score = score
                best_frame = frame
                best_verification = verification_mask
        
        if best_frame is not None:
            # Perform final cleanup on best frame
            edges = cv2.Canny(cv2.cvtColor(best_frame, cv2.COLOR_BGR2GRAY), 50, 150)
            
            # If objects still detected, try one final blend with background
            if np.any(best_verification > 0):
                best_frame = self.blend_frames(best_frame, background_frame, 
                                            best_verification, edges)
            
            # Save final result and verification images
            result_path = os.path.join(output_dir, 'result.jpg')
            cv2.imwrite(result_path, best_frame)
            
            # Create and save verification visualization
            verification_viz = np.hstack([
                best_frame,
                cv2.cvtColor(best_verification, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            ])
            cv2.imwrite(os.path.join(output_dir, 'result_verification.jpg'), verification_viz)
            
            # Save detection information
            with open(os.path.join(output_dir, 'result_info.txt'), 'w') as f:
                f.write(f"Final Detection Score: {best_score}\n")
                f.write(f"Number of Remaining Detections: {len(detections.boxes)}\n")
                f.write("Detected Objects:\n")
                for box in detections.boxes:
                    f.write(f"- Class: {int(box.cls)}, Confidence: {float(box.conf):.3f}\n")
            
            print(f"Final result saved with detection score: {best_score:.4f}")
            return best_frame
        else:
            print("Warning: Could not determine best result frame")
            return background_frame

def process_video(video_path, output_dir, yolo_model_path, interval_seconds=1):
    """Updated process_video function with result verification"""
    os.makedirs(output_dir, exist_ok=True)
    processor = EnhancedVideoProcessor(yolo_model_path)
    
    # First pass: extract frames and estimate background
    print("First pass: Extracting frames...")
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            frame = processor.resize_frame(frame)
            frames.append(frame)
            
        frame_count += 1
    
    cap.release()
    
    # Estimate background from collected frames
    background_frame = processor.estimate_background(frames)
    cv2.imwrite(os.path.join(output_dir, 'estimated_background.jpg'), background_frame)
    
    # Process frames with established background
    print("Second pass: Processing frames...")
    processed_frames = []
    output_video_path = os.path.join(output_dir, 'processed_video.mp4')
    height, width = frames[0].shape[:2]
    
    writer = cv2.VideoWriter(output_video_path,
                           cv2.VideoWriter_fourcc(*'mp4v'),
                           fps/frame_interval,
                           (width, height))
    
    for i, frame in enumerate(tqdm(frames, desc="Processing frames")):
        # Detect objects and create mask
        detections = processor.detect_objects(frame)
        object_mask = processor.create_object_mask(frame, detections)
        
        # Extract edges for static elements
        edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150)
        
        # Blend frames
        result = processor.blend_frames(frame, background_frame, object_mask, edges)
        processed_frames.append(result)
        
        # Save individual frame
        timestamp = datetime.fromtimestamp(i * interval_seconds).strftime('%H-%M-%S')
        frame_path = os.path.join(output_dir, f'frame_{i:04d}_{timestamp}.jpg')
        cv2.imwrite(frame_path, result)
        
        writer.write(result)
    
    writer.release()
    
    # Get final result with verification
    final_result = processor.result(output_dir, processed_frames, background_frame)
    
    # Save final comparison
    comparison = np.hstack([
        frames[0],
        final_result,
        frames[-1]
    ])
    cv2.imwrite(os.path.join(output_dir, 'final_comparison.jpg'), comparison)
    
    return final_result

if __name__ == "__main__":
    video_path = "images/752750716.004314.mp4"
    output_dir = "video_proccess/output"
    yolo_model_path = "yolov10x.pt"
    process_video(
        video_path,
        output_dir,
        yolo_model_path,
        interval_seconds=1
    )