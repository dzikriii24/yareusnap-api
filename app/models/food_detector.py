from ultralytics import YOLO
import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class FoodDetector:
    def __init__(self):
        self.model_paths = [
            "models/best.pt",
            "./models/best.pt",
            "/opt/render/models/best.pt"
        ]
        self.confidence = 0.25
        self.model = self._load_model()
    
    def _load_model(self):
        """Load model dengan multiple fallback paths untuk Render"""
        for model_path in self.model_paths:
            try:
                if os.path.exists(model_path):
                    logger.info(f"🔍 Loading model from: {model_path}")
                    model = YOLO(model_path)
                    
                    # Optimize untuk environment Render
                    model.overrides['verbose'] = False
                    model.overrides['device'] = 'cpu'
                    
                    logger.info(f"✅ Model loaded successfully: {model_path}")
                    logger.info(f"📊 Model classes: {len(model.names)}")
                    
                    # Test model dengan gambar kecil
                    test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
                    results = model(test_image, verbose=False)
                    logger.info("🧪 Model test passed")
                    
                    return model
                else:
                    logger.warning(f"⚠️ Model not found: {model_path}")
                    
            except Exception as e:
                logger.error(f"❌ Error loading model from {model_path}: {e}")
                continue
        
        # Fallback ke YOLO default
        logger.warning("🚨 No trained model found, using default YOLOv8n")
        try:
            model = YOLO('yolov8n.pt')
            logger.info("✅ Default YOLO model loaded as fallback")
            return model
        except Exception as e:
            logger.error(f"❌ Even default YOLO failed: {e}")
            raise e
    
    def detect_food(self, image):
        """Detect food in image dengan error handling"""
        try:
            # Validate input
            if image is None or image.size == 0:
                logger.error("❌ Invalid image input")
                return []
            
            # Preprocess image
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            
            logger.info(f"🖼️ Processing image: {image.shape}")
            
            # Run detection dengan confidence bertahap
            detections = []
            confidence_levels = [0.1, 0.15, 0.2, 0.25]
            
            for conf in confidence_levels:
                try:
                    results = self.model(image, conf=conf, verbose=False, device='cpu')
                    
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                confidence = float(box.conf[0])
                                class_id = int(box.cls[0])
                                label = self.model.names[class_id]
                                
                                formatted_label = self._format_label(label)
                                
                                # Check duplicate
                                if not self._is_duplicate(detections, [x1, y1, x2, y2]):
                                    detections.append({
                                        'bbox': [x1, y1, x2, y2],
                                        'label': formatted_label,
                                        'confidence': confidence,
                                        'class_id': class_id
                                    })
                    
                    if detections:
                        logger.info(f"🎯 Found {len(detections)} detections at confidence {conf}")
                        break
                        
                except Exception as e:
                    logger.warning(f"⚠️ Detection at confidence {conf} failed: {e}")
                    continue
            
            # Sort dan limit
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            logger.info(f"📊 Final detections: {len(detections)}")
            
            return detections[:10]
            
        except Exception as e:
            logger.error(f"❌ Detection error: {e}")
            return []
    
    def _is_duplicate(self, detections, bbox, threshold=0.7):
        """Check duplicate detection"""
        for det in detections:
            iou = self._calculate_iou(det['bbox'], bbox)
            if iou > threshold:
                return True
        return False
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _format_label(self, label):
        """Format food label"""
        translations = {
            'nasi_goreng': 'Nasi Goreng', 'fried_rice': 'Nasi Goreng',
            'bakso': 'Bakso', 'meatball': 'Bakso',
            'mie_goreng': 'Mie Goreng', 'noodle': 'Mie',
            'martabak': 'Martabak', 'sate': 'Sate',
            'rendang': 'Rendang', 'pizza': 'Pizza',
            'hamburger': 'Hamburger', 'sushi': 'Sushi',
            'ramen': 'Ramen', 'baklava': 'Baklava'
        }
        return translations.get(label, label.replace('_', ' ').title())
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes"""
        try:
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                label = det['label']
                confidence = det['confidence']
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label_text = f"{label} {confidence:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Label background
                cv2.rectangle(image, (x1, y1 - text_height - 10),
                            (x1 + text_width, y1), (0, 255, 0), -1)
                
                # Label text
                cv2.putText(image, label_text, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return image
        except Exception as e:
            logger.error(f"❌ Error drawing detections: {e}")
            return image