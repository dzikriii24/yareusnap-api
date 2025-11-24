from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import base64
import io
from PIL import Image
import numpy as np

router = APIRouter()

# Initialize components
from app.models.food_detector import FoodDetector
from app.models.nutrition_analyzer import NutritionAnalyzer

food_detector = FoodDetector()
nutrition_analyzer = NutritionAnalyzer()

@router.post("/analyze")
async def analyze_food(image: UploadFile = File(...)):
    """
    Analyze food image and provide nutrition analysis
    """
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        print(f"📨 Processing: {image.filename}")
        
        # Read and process image
        image_data = await image.read()
        image_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_np = np.array(image_pil)
        
        print(f"🖼️ Image size: {image_np.shape}")
        
        # Detect food
        detections = food_detector.detect_food(image_np)
        detected_foods = [det['label'] for det in detections]
        
        print(f"🎯 Detected foods: {detected_foods}")
        
        # Analyze nutrition
        nutrition_analysis = nutrition_analyzer.analyze_food_components(detected_foods)
        
        # Create annotated image
        annotated_image = food_detector.draw_detections(image_np.copy(), detections)
        
        # Convert to base64
        import cv2
        _, buffer = cv2.imencode('.jpg', annotated_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return JSONResponse({
            "success": True,
            "detected_foods": detected_foods,
            "detections": detections,
            "nutrition_analysis": nutrition_analysis,
            "annotated_image": image_base64,
            "message": f"Detected {len(detections)} food items"
        })
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test model
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        detections = food_detector.detect_food(test_image)
        
        # Test Mistral API
        mistral_success, mistral_message = nutrition_analyzer.test_api_connection()
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "food_classes": len(food_detector.model.names),
            "mistral_api": mistral_message,
            "service": "food-detection-api"
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "service": "food-detection-api"
        }

@router.get("/model-info")
async def model_info():
    """Get model information"""
    try:
        model = food_detector.model
        return {
            "model_path": "models/best.pt",
            "total_classes": len(model.names),
            "classes": list(model.names.values()),
            "confidence_threshold": food_detector.confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@router.post("/test-mistral")
async def test_mistral():
    """Test Mistral API connection"""
    try:
        success, message = nutrition_analyzer.test_api_connection()
        return {
            "success": success,
            "message": message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mistral test failed: {str(e)}")

@router.get("/test-detection")
async def test_detection():
    """Test detection with sample image"""
    try:
        # Create test image
        test_image = food_detector.create_test_image()
        detections = food_detector.detect_food(test_image)
        
        return {
            "success": True,
            "detections": detections,
            "test_image_created": True,
            "message": f"Test detection completed with {len(detections)} detections"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test detection failed: {str(e)}")