"""
Multimedia Generation Routes

API endpoints for music and image generation using multiple AI services
"""

from flask import Blueprint, request, jsonify, render_template
from services.multimedia_generation_service import multimedia_service
import logging

logger = logging.getLogger(__name__)

multimedia_bp = Blueprint('multimedia', __name__)

@multimedia_bp.route('/multimedia')
def multimedia_dashboard():
    """Multimedia generation dashboard"""
    try:
        service_status = multimedia_service.get_service_status()
        return render_template('multimedia_dashboard.html', 
                             service_status=service_status)
    except Exception as e:
        logger.error(f"Error loading multimedia dashboard: {str(e)}")
        return render_template('multimedia_dashboard.html', 
                             service_status={"error": str(e)})

@multimedia_bp.route('/api/multimedia/status')
def get_multimedia_status():
    """Get status of all multimedia services"""
    try:
        status = multimedia_service.get_service_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting multimedia status: {str(e)}")
        return jsonify({"error": str(e)}), 500

@multimedia_bp.route('/api/multimedia/generate-music', methods=['POST'])
def generate_music():
    """Generate music using Suno AI"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({"error": "Prompt is required"}), 400
        
        result = multimedia_service.generate_music(
            prompt=data['prompt'],
            title=data.get('title'),
            make_instrumental=data.get('make_instrumental', False),
            model=data.get('model', 'chirp-v4'),
            tags=data.get('tags')
        )
        
        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Error generating music: {str(e)}")
        return jsonify({"error": str(e)}), 500

@multimedia_bp.route('/api/multimedia/music-status/<task_id>')
def get_music_status(task_id):
    """Get status of a music generation task"""
    try:
        result = multimedia_service.get_music_status(task_id)
        
        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Error getting music status: {str(e)}")
        return jsonify({"error": str(e)}), 500

@multimedia_bp.route('/api/multimedia/generate-image', methods=['POST'])
def generate_image():
    """Generate image using Ideogram AI or DALL-E 3"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({"error": "Prompt is required"}), 400
        
        use_ideogram = data.get('service', 'ideogram') == 'ideogram'
        
        if use_ideogram:
            result = multimedia_service.generate_image_ideogram(
                prompt=data['prompt'],
                model=data.get('model', 'V_3'),
                aspect_ratio=data.get('aspect_ratio', 'ASPECT_1_1'),
                style_type=data.get('style_type', 'AUTO'),
                negative_prompt=data.get('negative_prompt'),
                style_reference_url=data.get('style_reference_url'),
                style_reference_weight=data.get('style_reference_weight', 0.7)
            )
        else:
            result = multimedia_service.generate_image_dalle(
                prompt=data['prompt'],
                size=data.get('size', '1024x1024'),
                quality=data.get('quality', 'standard'),
                style=data.get('style', 'vivid')
            )
        
        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@multimedia_bp.route('/api/multimedia/generate-content', methods=['POST'])
def generate_multimedia_content():
    """Generate both music and image content simultaneously"""
    try:
        data = request.get_json()
        
        if not data or 'music_prompt' not in data or 'image_prompt' not in data:
            return jsonify({"error": "Both music_prompt and image_prompt are required"}), 400
        
        result = multimedia_service.generate_multimedia_content(
            music_prompt=data['music_prompt'],
            image_prompt=data['image_prompt'],
            title=data.get('title'),
            use_ideogram=data.get('use_ideogram', True),
            music_tags=data.get('music_tags'),
            image_style=data.get('image_style', 'AUTO')
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error generating multimedia content: {str(e)}")
        return jsonify({"error": str(e)}), 500

@multimedia_bp.route('/api/multimedia/test-apis')
def test_apis():
    """Test all multimedia APIs with simple requests"""
    try:
        availability = multimedia_service.check_service_availability()
        test_results = {
            "availability": availability,
            "tests": {},
            "overall_status": "success"
        }
        
        # Test Suno API if available
        if availability.get("suno"):
            try:
                suno_test = multimedia_service.generate_music(
                    prompt="A short 10-second test jingle, upbeat and cheerful",
                    title="API Test",
                    tags="test, short"
                )
                test_results["tests"]["suno"] = {
                    "status": "success" if suno_test.get("success") else "failed",
                    "result": suno_test
                }
            except Exception as e:
                test_results["tests"]["suno"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            test_results["tests"]["suno"] = {
                "status": "unavailable",
                "reason": "API key not configured"
            }
        
        # Test Ideogram API if available
        if availability.get("ideogram"):
            try:
                ideogram_test = multimedia_service.generate_image_ideogram(
                    prompt="A simple test image: blue circle on white background",
                    model="V_2_TURBO"  # Use faster/cheaper model for testing
                )
                test_results["tests"]["ideogram"] = {
                    "status": "success" if ideogram_test.get("success") else "failed",
                    "result": ideogram_test
                }
            except Exception as e:
                test_results["tests"]["ideogram"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            test_results["tests"]["ideogram"] = {
                "status": "unavailable",
                "reason": "API key not configured"
            }
        
        # Test DALL-E 3 API if available
        if availability.get("openai_dalle"):
            try:
                dalle_test = multimedia_service.generate_image_dalle(
                    prompt="A simple test image: red square on white background",
                    size="1024x1024",
                    quality="standard"
                )
                test_results["tests"]["dalle3"] = {
                    "status": "success" if dalle_test.get("success") else "failed",
                    "result": dalle_test
                }
            except Exception as e:
                test_results["tests"]["dalle3"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            test_results["tests"]["dalle3"] = {
                "status": "unavailable",
                "reason": "API key not configured"
            }
        
        # Check overall status
        failed_tests = [k for k, v in test_results["tests"].items() 
                       if v["status"] in ["failed", "error"]]
        if failed_tests:
            test_results["overall_status"] = "partial_failure"
            if len(failed_tests) == len(test_results["tests"]):
                test_results["overall_status"] = "failure"
        
        return jsonify(test_results)
        
    except Exception as e:
        logger.error(f"Error testing multimedia APIs: {str(e)}")
        return jsonify({"error": str(e), "overall_status": "error"}), 500