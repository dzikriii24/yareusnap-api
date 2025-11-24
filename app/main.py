from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
import os
import time

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
    """Initialize on startup - OPTIMIZED FOR RENDER"""
    print("🚀 Food Detection API Starting on Render...")
    
    # Check environment
    print(f"🌍 Environment: {os.getenv('RENDER', 'Local')}")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📦 Contents: {os.listdir('.')}")
    
    # Check models directory
    models_dir = "models"
    if os.path.exists(models_dir):
        print(f"✅ Models directory exists")
        model_files = os.listdir(models_dir)
        print(f"📁 Model files: {model_files}")
    else:
        print(f"❌ Models directory not found")
        os.makedirs(models_dir, exist_ok=True)
        print(f"✅ Created models directory")
    
    # Initialize components with error handling
    try:
        print("🔄 Initializing Food Detector...")
        from app.models.food_detector import FoodDetector
        detector = FoodDetector()
        print(f"✅ Food Detector ready with {len(detector.model.names)} classes")
    except Exception as e:
        print(f"❌ Food Detector failed: {e}")
    
    try:
        print("🔄 Initializing Nutrition Analyzer...")
        from app.models.nutrition_analyzer import NutritionAnalyzer
        analyzer = NutritionAnalyzer()
        success, message = analyzer.test_api_connection()
        print(f"🤖 {message}")
    except Exception as e:
        print(f"⚠️ Nutrition Analyzer warning: {e}")
    
    print("✅ All systems initialized!")

@app.get("/")
async def root():
    return {
        "message": "Food Detection API is running on Render!",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "POST /api/v1/analyze": "Analyze food image",
            "GET /api/v1/health": "Health check", 
            "GET /api/v1/model-info": "Model information",
            "GET /docs": "API Documentation"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "service": "food-detection-api",
        "environment": "render",
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)