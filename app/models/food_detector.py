from ultralytics import YOLO
import cv2
import numpy as np
import os

class FoodDetector:
    def __init__(self):
        self.model_path = "models/best.pt"
        self.confidence = 0.25
        self.model = self._load_model()
    
    def _load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                model = YOLO(self.model_path)
                print(f"✅ Model loaded: {self.model_path}")
                print(f"📊 Classes: {len(model.names)}")
                return model
            else:
                print("❌ Trained model not found, using default YOLO")
                return YOLO('yolov8n.pt')
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return YOLO('yolov8n.pt')
    
    def detect_food(self, image):
        """Detect food in image"""
        try:
            # Preprocess image
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            
            # Run detection with multiple confidence levels
            detections = []
            confidence_levels = [0.1, 0.15, 0.2, 0.25]
            
            for conf in confidence_levels:
                results = self.model(image, conf=conf, verbose=False)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            confidence = box.conf[0].item()
                            class_id = int(box.cls[0].item())
                            label = self.model.names[class_id]
                            
                            # Format label
                            formatted_label = self._format_label(label)
                            
                            # Check for duplicates
                            if not self._is_duplicate(detections, [x1, y1, x2, y2]):
                                detections.append({
                                    'bbox': [x1, y1, x2, y2],
                                    'label': formatted_label,
                                    'confidence': confidence,
                                    'class_id': class_id
                                })
                
                if detections:
                    break
            
            # Sort by confidence
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            return detections[:10]  # Limit to top 10
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return []
    
    def _is_duplicate(self, detections, bbox, threshold=0.7):
        """Check if detection is duplicate"""
        for det in detections:
            if self._calculate_iou(det['bbox'], bbox) > threshold:
                return True
        return False
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _format_label(self, label):
        """Format food label"""
        translations = {
            'nasi_goreng': 'Nasi Goreng',
            'fried_rice': 'Nasi Goreng',
            'bakso': 'Bakso',
            'mie_goreng': 'Mie Goreng',
            'martabak': 'Martabak',
            'sate': 'Sate',
            'rendang': 'Rendang',
            'pizza': 'Pizza',
            'hamburger': 'Hamburger',
            'sushi': 'Sushi',
            'ramen': 'Ramen'
        }
        return translations.get(label, label.replace('_', ' ').title())
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes on image"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            confidence = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label_text = f"{label} {confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Label background
            cv2.rectangle(image, (x1, y1 - text_height - 10),
                         (x1 + text_width, y1), (0, 255, 0), -1)
            
            # Label text
            cv2.putText(image, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
    
    def create_test_image(self):
        """Create test image for detection"""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # Draw food-like shapes
        cv2.circle(img, (160, 160), 80, (0, 165, 255), -1)  # Pizza
        cv2.rectangle(img, (400, 120), (560, 280), (139, 69, 19), -1)  # Burger
        cv2.rectangle(img, (300, 350), (380, 430), (0, 0, 255), -1)  # Sushi
        
        return img