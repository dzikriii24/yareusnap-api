from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
import os

app = FastAPI(
    title="Food Detection API",
    description="AI-powered Food Detection and Nutrition Analysis API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("🚀 Food Detection API Starting...")
    
    # Check if model exists
    model_path = "models/best.pt"
    if os.path.exists(model_path):
        print(f"✅ Model found: {model_path}")
    else:
        print(f"❌ Model not found: {model_path}")
    
    # Initialize components
    try:
        from app.models.food_detector import FoodDetector
        detector = FoodDetector()
        print(f"🍽️ Food Detector ready with {len(detector.model.names)} classes")
    except Exception as e:
        print(f"⚠️ Food detector init warning: {e}")
    
    try:
        from app.models.nutrition_analyzer import NutritionAnalyzer
        analyzer = NutritionAnalyzer()
        success, message = analyzer.test_api_connection()
        print(f"🤖 {message}")
    except Exception as e:
        print(f"⚠️ Nutrition analyzer init warning: {e}")

@app.get("/")
async def root():
    return {
        "message": "Food Detection API is running!",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/v1/analyze": "Analyze food image",
            "GET /api/v1/health": "Health check",
            "GET /api/v1/model-info": "Model information"
        },
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "food-detection-api"}