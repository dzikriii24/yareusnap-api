from mistralai import Mistral
import json
import re
import os
from dotenv import load_dotenv

load_dotenv()

class NutritionAnalyzer:
    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY", "ICH2b9048GrX2WXz8JsgBunZDfgMLo0G")
        self.model = "mistral-medium"
        if self.api_key:
            self.client = Mistral(api_key=self.api_key)
        else:
            self.client = None
            print("⚠️ Mistral API key not found")
    
    def test_api_connection(self):
        """Test Mistral API connection"""
        if not self.client:
            return False, "Mistral API key not configured"
        
        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": "Test connection"}],
                max_tokens=10
            )
            return True, "Mistral API connection successful"
        except Exception as e:
            return False, f"Mistral API connection failed: {str(e)}"
    
    def analyze_food_components(self, detected_foods):
        """Analyze food nutrition"""
        if not self.client:
            return self._create_fallback_response(detected_foods)
        
        try:
            prompt = self._create_analysis_prompt(detected_foods)
            
            response = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            return self._parse_response(content, detected_foods)
            
        except Exception as e:
            print(f"❌ Mistral analysis error: {e}")
            return self._create_fallback_response(detected_foods)
    
    def _create_analysis_prompt(self, detected_foods):
        return f"""
        ANALYZE FOOD NUTRITION - RESPOND WITH JSON ONLY:
        
        Detected Foods: {', '.join(detected_foods) if detected_foods else 'No foods detected'}
        
        Provide nutrition analysis in Indonesian with this JSON structure:
        {{
            "food_type": "string",
            "components": ["list", "of", "components"],
            "nutrition": {{
                "protein": "Tinggi/Sedang/Rendah",
                "carbs": "Tinggi/Sedang/Rendah",
                "fat": "Tinggi/Sedang/Rendah", 
                "fiber": "Tinggi/Sedang/Rendah",
                "vitamins": "Tinggi/Sedang/Rendah"
            }},
            "deficiencies": ["list", "of", "deficiencies"],
            "recommendations": ["list", "of", "recommendations"]
        }}
        
        Important: Respond with JSON only.
        """
    
    def _parse_response(self, content, detected_foods):
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return self._create_fallback_response(detected_foods)
    
    def _create_fallback_response(self, detected_foods):
        if not detected_foods:
            return {
                "food_type": "Tidak Terdeteksi",
                "components": [],
                "nutrition": {
                    "protein": "Tidak Diketahui", "carbs": "Tidak Diketahui",
                    "fat": "Tidak Diketahui", "fiber": "Tidak Diketahui",
                    "vitamins": "Tidak Diketahui"
                },
                "deficiencies": ["Tidak ada makanan terdeteksi"],
                "recommendations": [
                    "Pastikan makanan terlihat jelas",
                    "Gunakan pencahayaan yang baik",
                    "Fokus pada satu jenis makanan"
                ]
            }
        
        return {
            "food_type": "Makanan Terdeteksi",
            "components": detected_foods,
            "nutrition": {
                "protein": "Sedang", "carbs": "Tinggi",
                "fat": "Sedang", "fiber": "Rendah",
                "vitamins": "Sedang"
            },
            "deficiencies": ["Perlu analisis lebih detail"],
            "recommendations": [
                "Konsultasi dengan ahli gizi",
                "Variasi dengan buah dan sayuran",
                "Perhatikan porsi makan"
            ]
        }