"""
Multimedia Generation Service

Integrates with:
- Suno AI (via SunoAPI.org) for music generation
- Ideogram AI (official API) for image generation
- OpenAI DALL-E 3 for additional image generation (Playground alternative)
"""

import os
import requests
import json
import time
import logging
from typing import Dict, List, Optional, Union, Any
from openai import OpenAI
from google import genai
from google.genai import types

# the newest OpenAI model is "gpt-5" which was released August 7, 2025.
# do not change this unless explicitly requested by the user

logger = logging.getLogger(__name__)

class MultimediaGenerationService:
    """Service for generating multimedia content using multiple AI providers"""
    
    def __init__(self):
        self.suno_api_key = os.environ.get("SUNO_API_KEY")
        self.ideogram_api_key = os.environ.get("IDEOGRAM_API_KEY")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        
        # Initialize OpenAI client
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            
        # Initialize Gemini client for video generation
        if self.gemini_api_key:
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
        else:
            self.gemini_client = None
            
        # API endpoints
        self.suno_endpoint = "https://api.sunoapi.com/v1/suno/create"
        self.ideogram_endpoint = "https://api.ideogram.ai/generate"
        
        logger.info("Multimedia Generation Service initialized")

    def check_service_availability(self) -> Dict[str, bool]:
        """Check which services are available based on API keys"""
        availability = {
            "suno": bool(self.suno_api_key),
            "ideogram": bool(self.ideogram_api_key),
            "openai_dalle": bool(self.openai_api_key),
            "openai_storytelling": bool(self.openai_api_key),
            "gemini_veo3": bool(self.gemini_api_key),
        }
        
        missing_keys = [service for service, available in availability.items() if not available]
        if missing_keys:
            logger.warning(f"Missing API keys for: {', '.join(missing_keys)}")
            
        return availability

    # Suno AI Music Generation
    def generate_music(
        self,
        prompt: str,
        title: Optional[str] = None,
        make_instrumental: bool = False,
        model: str = "chirp-v4",
        tags: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate music using Suno AI
        
        Args:
            prompt: Description of the music to generate
            title: Optional title for the track
            make_instrumental: Whether to generate instrumental-only music
            model: Suno model to use (chirp-v4, chirp-v3-5, etc.)
            tags: Genre/style tags (e.g., "pop, electronic, upbeat")
            
        Returns:
            Dictionary containing generation result and metadata
        """
        if not self.suno_api_key:
            return {"error": "Suno API key not configured", "success": False}
            
        headers = {
            "Authorization": f"Bearer {self.suno_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "custom_mode": True,
            "gpt_description_prompt": prompt,
            "make_instrumental": make_instrumental,
            "mv": model
        }
        
        if title:
            payload["title"] = title
        if tags:
            payload["tags"] = tags
            
        try:
            response = requests.post(self.suno_endpoint, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Suno music generation initiated: {result.get('task_id', 'Unknown ID')}")
            
            return {
                "success": True,
                "service": "suno",
                "task_id": result.get("task_id"),
                "data": result,
                "estimated_cost": 0.01,  # Approximate cost per generation
                "model_used": model
            }
            
        except requests.RequestException as e:
            logger.error(f"Suno API request failed: {str(e)}")
            return {"error": f"Suno API request failed: {str(e)}", "success": False}
        except Exception as e:
            logger.error(f"Unexpected error in Suno generation: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}", "success": False}

    def get_music_status(self, task_id: str) -> Dict[str, Any]:
        """Check the status of a Suno music generation task"""
        if not self.suno_api_key:
            return {"error": "Suno API key not configured", "success": False}
            
        headers = {
            "Authorization": f"Bearer {self.suno_api_key}",
        }
        
        try:
            response = requests.get(f"https://api.sunoapi.com/v1/suno/get/{task_id}", headers=headers)
            response.raise_for_status()
            
            return {
                "success": True,
                "service": "suno",
                "data": response.json()
            }
            
        except Exception as e:
            logger.error(f"Error checking Suno task status: {str(e)}")
            return {"error": str(e), "success": False}

    # Ideogram AI Image Generation
    def generate_image_ideogram(
        self,
        prompt: str,
        model: str = "V_3",
        aspect_ratio: str = "ASPECT_1_1",
        style_type: str = "AUTO",
        negative_prompt: Optional[str] = None,
        style_reference_url: Optional[str] = None,
        style_reference_weight: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate image using Ideogram AI
        
        Args:
            prompt: Description of the image to generate
            model: Ideogram model (V_3, V_2, V_2_TURBO)
            aspect_ratio: Image aspect ratio (ASPECT_1_1, ASPECT_16_9, etc.)
            style_type: Style approach (AUTO, REALISTIC, DESIGN, etc.)
            negative_prompt: Things to avoid in the image
            style_reference_url: URL of reference image for style
            style_reference_weight: Influence strength of style reference (0-1)
            
        Returns:
            Dictionary containing generation result and metadata
        """
        if not self.ideogram_api_key:
            return {"error": "Ideogram API key not configured", "success": False}
            
        headers = {
            "Authorization": f"Bearer {self.ideogram_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "prompt": prompt,
            "model": model,
            "aspect_ratio": aspect_ratio,
            "style_type": style_type,
            "magic_prompt_option": "AUTO"
        }
        
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
            
        if style_reference_url:
            # Add style reference as dict
            payload["style_reference"] = {
                "image_url": style_reference_url,
                "weight": float(style_reference_weight)
            }
            
        try:
            response = requests.post(self.ideogram_endpoint, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Ideogram image generation successful")
            
            # Calculate estimated cost based on model
            cost_map = {
                "V_2_TURBO": 0.025,
                "V_2": 0.04,
                "V_3": 0.07,
                "V_3_TURBO": 0.04,
                "V_3_QUALITY": 0.10
            }
            
            return {
                "success": True,
                "service": "ideogram",
                "data": result,
                "estimated_cost": cost_map.get(model, 0.07),
                "model_used": model,
                "image_urls": result.get("data", [])
            }
            
        except requests.RequestException as e:
            logger.error(f"Ideogram API request failed: {str(e)}")
            return {"error": f"Ideogram API request failed: {str(e)}", "success": False}
        except Exception as e:
            logger.error(f"Unexpected error in Ideogram generation: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}", "success": False}

    # OpenAI DALL-E 3 Image Generation
    def generate_image_dalle(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid"
    ) -> Dict[str, Any]:
        """
        Generate image using OpenAI DALL-E 3
        
        Args:
            prompt: Description of the image to generate
            size: Image size (256x256, 512x512, 1024x1024, 1024x1792, 1792x1024)
            quality: Image quality (standard, hd)
            style: Image style (vivid, natural)
            
        Returns:
            Dictionary containing generation result and metadata
        """
        if not self.openai_client:
            return {"error": "OpenAI API key not configured", "success": False}
            
        try:
            from typing import cast, Literal
            
            # Validate parameters for OpenAI API and cast to proper types
            valid_sizes = ["256x256", "512x512", "1024x1024", "1024x1792", "1792x1024"]
            valid_qualities = ["standard", "hd"]
            valid_styles = ["vivid", "natural"]
            
            # Use defaults if invalid values provided
            validated_size = size if size in valid_sizes else "1024x1024"
            validated_quality = quality if quality in valid_qualities else "standard"
            validated_style = style if style in valid_styles else "vivid"
            
            # Cast to the specific literal types that OpenAI expects
            size_param = cast(Literal["256x256", "512x512", "1024x1024", "1024x1792", "1792x1024"], validated_size)
            quality_param = cast(Literal["standard", "hd"], validated_quality)
            style_param = cast(Literal["vivid", "natural"], validated_style)
                
            response = self.openai_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size_param,
                quality=quality_param,
                style=style_param,
                n=1
            )
            
            logger.info("DALL-E 3 image generation successful")
            
            # Calculate estimated cost based on size and quality
            if size == "1024x1024":
                estimated_cost = 0.08 if quality == "hd" else 0.04
            elif size in ["1024x1792", "1792x1024"]:
                estimated_cost = 0.12 if quality == "hd" else 0.08
            else:
                estimated_cost = 0.04
            
            image_data = response.data[0] if response.data else None
            if not image_data:
                return {"error": "No image data returned from API", "success": False}
                
            return {
                "success": True,
                "service": "dalle3",
                "data": {
                    "url": image_data.url,
                    "revised_prompt": getattr(image_data, 'revised_prompt', prompt)
                },
                "estimated_cost": estimated_cost,
                "model_used": "dall-e-3",
                "image_url": image_data.url
            }
            
        except Exception as e:
            logger.error(f"DALL-E 3 generation failed: {str(e)}")
            return {"error": f"DALL-E 3 generation failed: {str(e)}", "success": False}

    # OpenAI Storytelling and Script Generation
    def generate_script(
        self,
        concept: str,
        script_type: str = "social_media",
        duration: int = 30,
        tone: str = "engaging",
        target_audience: str = "general",
        model: str = "gpt-5"
    ) -> Dict[str, Any]:
        """
        Generate scripts and narratives using OpenAI GPT models
        
        Args:
            concept: The core idea or message for the script
            script_type: Type of script (social_media, commercial, explainer, story, etc.)
            duration: Target duration in seconds
            tone: Tone of the script (engaging, professional, casual, dramatic, etc.)
            target_audience: Target audience description
            model: OpenAI model to use (gpt-5, gpt-4o, etc.)
            
        Returns:
            Dictionary containing generated script and metadata
        """
        if not self.openai_client:
            return {"error": "OpenAI API key not configured", "success": False}
            
        try:
            # Craft specialized prompt based on script type
            prompts = {
                "social_media": f"Create an engaging {duration}-second social media script about: {concept}\n\nRequirements:\n- Hook viewers in the first 3 seconds\n- Clear, concise messaging\n- Call-to-action\n- Tone: {tone}\n- Audience: {target_audience}\n\nFormat: Include scene descriptions, dialogue, and timing cues.",
                "commercial": f"Write a professional {duration}-second commercial script for: {concept}\n\nRequirements:\n- Strong opening hook\n- Clear value proposition\n- Compelling call-to-action\n- Tone: {tone}\n- Target: {target_audience}\n\nInclude: Visual cues, voiceover, and transition notes.",
                "explainer": f"Create an educational {duration}-second explainer script about: {concept}\n\nRequirements:\n- Clear, simple explanations\n- Logical flow of information\n- Engaging presentation\n- Tone: {tone}\n- Audience: {target_audience}\n\nInclude: Key points, visual suggestions, and pacing.",
                "story": f"Write a compelling {duration}-second narrative story based on: {concept}\n\nRequirements:\n- Strong character development\n- Clear story arc\n- Emotional engagement\n- Tone: {tone}\n- Audience: {target_audience}\n\nInclude: Character descriptions, plot points, and scene details."
            }
            
            prompt = prompts.get(script_type, prompts["social_media"])
            
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert scriptwriter and storyteller with extensive experience in creating compelling content for various media formats. Your scripts are known for their engaging narratives, clear structure, and audience appeal."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=1500
            )
            
            script_content = response.choices[0].message.content
            
            logger.info(f"Script generation successful: {script_type}")
            
            # Calculate cost based on token usage
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            
            # GPT-5 pricing (approximate)
            cost_per_prompt_token = 0.005 / 1000  # $5 per 1M input tokens
            cost_per_completion_token = 0.015 / 1000  # $15 per 1M output tokens
            estimated_cost = (prompt_tokens * cost_per_prompt_token) + (completion_tokens * cost_per_completion_token)
            
            return {
                "success": True,
                "service": "openai_storytelling",
                "script_content": script_content,
                "script_type": script_type,
                "duration": duration,
                "tone": tone,
                "target_audience": target_audience,
                "model_used": model,
                "estimated_cost": estimated_cost,
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Script generation failed: {str(e)}")
            return {"error": f"Script generation failed: {str(e)}", "success": False}
    
    def generate_storyboard_descriptions(
        self,
        script_content: str,
        num_scenes: int = 6,
        visual_style: str = "cinematic",
        model: str = "gpt-5"
    ) -> Dict[str, Any]:
        """
        Generate detailed storyboard scene descriptions from a script
        
        Args:
            script_content: The script to create storyboard for
            num_scenes: Number of storyboard scenes to generate
            visual_style: Visual style (cinematic, minimalist, vibrant, corporate, etc.)
            model: OpenAI model to use
            
        Returns:
            Dictionary containing storyboard descriptions and metadata
        """
        if not self.openai_client:
            return {"error": "OpenAI API key not configured", "success": False}
            
        try:
            prompt = f"""Create detailed storyboard descriptions for the following script:

{script_content}

Requirements:
- Generate exactly {num_scenes} distinct scenes
- Visual style: {visual_style}
- Each scene should include:
  * Scene number and title
  * Detailed visual description
  * Camera angle and composition
  * Mood and lighting
  * Key visual elements
  * Any text/graphics overlays

Format each scene as:
SCENE X: [Title]
- Visual: [Detailed description]
- Camera: [Angle/composition]
- Mood: [Lighting/atmosphere]
- Elements: [Key objects/characters]
- Text: [Any overlays/graphics]

Make each scene visually distinct and create a cohesive visual narrative."""
            
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a professional storyboard artist and visual director with expertise in creating detailed scene descriptions for multimedia content. Your descriptions are vivid, technically precise, and optimized for visual content generation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            storyboard_content = response.choices[0].message.content
            
            logger.info(f"Storyboard generation successful: {num_scenes} scenes")
            
            # Calculate cost
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            estimated_cost = (prompt_tokens * 0.005 / 1000) + (completion_tokens * 0.015 / 1000)
            
            return {
                "success": True,
                "service": "openai_storytelling",
                "storyboard_content": storyboard_content,
                "num_scenes": num_scenes,
                "visual_style": visual_style,
                "model_used": model,
                "estimated_cost": estimated_cost,
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Storyboard generation failed: {str(e)}")
            return {"error": f"Storyboard generation failed: {str(e)}", "success": False}
    
    def generate_character_profiles(
        self,
        story_concept: str,
        num_characters: int = 3,
        character_types: Optional[List[str]] = None,
        model: str = "gpt-5"
    ) -> Dict[str, Any]:
        """
        Generate detailed character profiles for storytelling
        
        Args:
            story_concept: The overall story or project concept
            num_characters: Number of characters to create
            character_types: Specific character types (protagonist, antagonist, mentor, etc.)
            model: OpenAI model to use
            
        Returns:
            Dictionary containing character profiles and metadata
        """
        if not self.openai_client:
            return {"error": "OpenAI API key not configured", "success": False}
            
        try:
            character_type_text = ""
            if character_types:
                character_type_text = f"\nCharacter types needed: {', '.join(character_types)}"
            
            prompt = f"""Create {num_characters} detailed character profiles for this story concept:

{story_concept}{character_type_text}

For each character, provide:

1. **Name & Role**
   - Full name
   - Role in the story
   - Age and background

2. **Physical Description**
   - Appearance details
   - Distinctive features
   - Style/clothing preferences

3. **Personality**
   - Core personality traits
   - Motivations and goals
   - Fears or challenges

4. **Background**
   - Brief history
   - Skills and abilities
   - Relationships to other characters

5. **Visual Notes**
   - Key visual elements for image generation
   - Mood and expression
   - Signature look or props

Make each character unique, memorable, and relevant to the story concept."""
            
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a professional character designer and storyteller who creates compelling, well-rounded characters for various media. Your character profiles are detailed, consistent, and optimized for visual representation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=2500
            )
            
            character_content = response.choices[0].message.content
            
            logger.info(f"Character generation successful: {num_characters} characters")
            
            # Calculate cost
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            estimated_cost = (prompt_tokens * 0.005 / 1000) + (completion_tokens * 0.015 / 1000)
            
            return {
                "success": True,
                "service": "openai_storytelling",
                "character_profiles": character_content,
                "num_characters": num_characters,
                "character_types": character_types,
                "model_used": model,
                "estimated_cost": estimated_cost,
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Character generation failed: {str(e)}")
            return {"error": f"Character generation failed: {str(e)}", "success": False}
    
    def enhance_prompts_for_generation(
        self,
        basic_prompts: List[str],
        content_type: str = "image",
        style_preferences: Optional[str] = None,
        model: str = "gpt-5"
    ) -> Dict[str, Any]:
        """
        Enhance basic prompts with detailed descriptions for better AI generation
        
        Args:
            basic_prompts: List of basic prompt descriptions
            content_type: Type of content (image, music, video)
            style_preferences: Specific style preferences or requirements
            model: OpenAI model to use
            
        Returns:
            Dictionary containing enhanced prompts and metadata
        """
        if not self.openai_client:
            return {"error": "OpenAI API key not configured", "success": False}
            
        try:
            style_text = f"\nStyle requirements: {style_preferences}" if style_preferences else ""
            
            prompt_enhancement_guides = {
                "image": "Enhance these image generation prompts with detailed visual descriptions, artistic style notes, composition guidance, lighting details, color palette suggestions, and technical photography terms that will produce high-quality, professional images.",
                "music": "Enhance these music generation prompts with detailed genre specifications, instrument descriptions, tempo and mood indicators, arrangement suggestions, and musical style references that will produce engaging, professional audio.",
                "video": "Enhance these video generation prompts with detailed scene descriptions, camera movement notes, visual effects suggestions, pacing guidance, and cinematic style references that will produce compelling, professional video content."
            }
            
            base_guide = prompt_enhancement_guides.get(content_type, prompt_enhancement_guides["image"])
            
            prompt = f"""{base_guide}{style_text}

Original prompts to enhance:
{chr(10).join([f"{i+1}. {prompt}" for i, prompt in enumerate(basic_prompts)])}

For each prompt, provide:
- **Enhanced Prompt**: Detailed, optimized version
- **Key Elements**: Important visual/audio elements to emphasize
- **Technical Notes**: Specific parameters or style guidance

Make the enhanced prompts significantly more detailed and specific while maintaining the original intent."""
            
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"You are an expert prompt engineer specializing in AI {content_type} generation. You understand how to craft detailed, effective prompts that produce high-quality results from various AI generation services."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=2000
            )
            
            enhanced_content = response.choices[0].message.content
            
            logger.info(f"Prompt enhancement successful: {len(basic_prompts)} prompts")
            
            # Calculate cost
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            estimated_cost = (prompt_tokens * 0.005 / 1000) + (completion_tokens * 0.015 / 1000)
            
            return {
                "success": True,
                "service": "openai_storytelling",
                "enhanced_prompts": enhanced_content,
                "original_prompts": basic_prompts,
                "content_type": content_type,
                "style_preferences": style_preferences,
                "model_used": model,
                "estimated_cost": estimated_cost,
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Prompt enhancement failed: {str(e)}")
            return {"error": f"Prompt enhancement failed: {str(e)}", "success": False}

    # Advanced Storytelling Workflows
    def create_complete_story_package(
        self,
        story_concept: str,
        duration: int = 60,
        num_scenes: int = 6,
        visual_style: str = "cinematic",
        music_style: str = "emotional, cinematic",
        target_audience: str = "general"
    ) -> Dict[str, Any]:
        """
        Generate a complete storytelling package: script, storyboard, characters, and multimedia
        
        Args:
            story_concept: Core story idea
            duration: Target duration in seconds
            num_scenes: Number of storyboard scenes
            visual_style: Visual style for imagery
            music_style: Music style/tags
            target_audience: Target audience
            
        Returns:
            Dictionary containing complete story package
        """
        package = {
            "success": True,
            "story_concept": story_concept,
            "script": None,
            "storyboard": None,
            "characters": None,
            "enhanced_prompts": None,
            "total_estimated_cost": 0,
            "errors": []
        }
        
        try:
            # 1. Generate script
            script_result = self.generate_script(
                concept=story_concept,
                script_type="story",
                duration=duration,
                target_audience=target_audience
            )
            
            if script_result.get("success"):
                package["script"] = script_result
                package["total_estimated_cost"] += script_result.get("estimated_cost", 0)
            else:
                package["errors"].append(f"Script generation failed: {script_result.get('error')}")
                package["success"] = False
                return package
            
            # 2. Generate character profiles
            character_result = self.generate_character_profiles(
                story_concept=story_concept,
                num_characters=3
            )
            
            if character_result.get("success"):
                package["characters"] = character_result
                package["total_estimated_cost"] += character_result.get("estimated_cost", 0)
            else:
                package["errors"].append(f"Character generation failed: {character_result.get('error')}")
            
            # 3. Generate storyboard descriptions
            storyboard_result = self.generate_storyboard_descriptions(
                script_content=script_result.get("script_content", ""),
                num_scenes=num_scenes,
                visual_style=visual_style
            )
            
            if storyboard_result.get("success"):
                package["storyboard"] = storyboard_result
                package["total_estimated_cost"] += storyboard_result.get("estimated_cost", 0)
            else:
                package["errors"].append(f"Storyboard generation failed: {storyboard_result.get('error')}")
            
            # 4. Create enhanced prompts for multimedia generation
            if storyboard_result.get("success"):
                # Extract basic prompts from storyboard
                basic_prompts = [f"Scene from story: {story_concept}, visual style: {visual_style}"]
                
                enhancement_result = self.enhance_prompts_for_generation(
                    basic_prompts=basic_prompts,
                    content_type="image",
                    style_preferences=visual_style
                )
                
                if enhancement_result.get("success"):
                    package["enhanced_prompts"] = enhancement_result
                    package["total_estimated_cost"] += enhancement_result.get("estimated_cost", 0)
                else:
                    package["errors"].append(f"Prompt enhancement failed: {enhancement_result.get('error')}")
            
            logger.info(f"Complete story package generated. Total cost: ${package['total_estimated_cost']:.4f}")
            return package
            
        except Exception as e:
            logger.error(f"Story package generation failed: {str(e)}")
            package["errors"].append(f"Unexpected error: {str(e)}")
            package["success"] = False
            return package

    # Unified Generation Methods
    def generate_multimedia_content(
        self,
        music_prompt: str,
        image_prompt: str,
        title: Optional[str] = None,
        use_ideogram: bool = True,
        music_tags: Optional[str] = None,
        image_style: str = "AUTO"
    ) -> Dict[str, Any]:
        """
        Generate both music and image content simultaneously
        
        Args:
            music_prompt: Description for music generation
            image_prompt: Description for image generation
            title: Optional title for both assets
            use_ideogram: Whether to use Ideogram (True) or DALL-E 3 (False)
            music_tags: Genre/style tags for music
            image_style: Style for image generation
            
        Returns:
            Dictionary containing results from both services
        """
        results = {
            "success": True,
            "music": None,
            "image": None,
            "total_estimated_cost": 0,
            "errors": []
        }
        
        # Generate music
        try:
            music_result = self.generate_music(
                prompt=music_prompt,
                title=title,
                tags=music_tags
            )
            
            if music_result.get("success"):
                results["music"] = music_result
                results["total_estimated_cost"] += music_result.get("estimated_cost", 0)
            else:
                results["errors"].append(f"Music generation failed: {music_result.get('error')}")
                results["success"] = False
                
        except Exception as e:
            results["errors"].append(f"Music generation error: {str(e)}")
            results["success"] = False
        
        # Generate image
        try:
            if use_ideogram:
                image_result = self.generate_image_ideogram(
                    prompt=image_prompt,
                    style_type=image_style
                )
            else:
                image_result = self.generate_image_dalle(
                    prompt=image_prompt
                )
                
            if image_result.get("success"):
                results["image"] = image_result
                results["total_estimated_cost"] += image_result.get("estimated_cost", 0)
            else:
                results["errors"].append(f"Image generation failed: {image_result.get('error')}")
                results["success"] = False
                
        except Exception as e:
            results["errors"].append(f"Image generation error: {str(e)}")
            results["success"] = False
        
        logger.info(f"Multimedia generation completed. Cost: ${results['total_estimated_cost']:.4f}")
        return results

    # Gemini VEO3 Video Generation
    def generate_video_veo3(
        self,
        prompt: str,
        duration: int = 5,
        aspect_ratio: str = "16:9",
        style: str = "realistic"
    ) -> Dict[str, Any]:
        """
        Generate video using Gemini VEO3
        
        Args:
            prompt: Description of the video to generate
            duration: Video duration in seconds (max 60)
            aspect_ratio: Video aspect ratio (16:9, 9:16, 1:1)
            style: Video style (realistic, cinematic, artistic, etc.)
            
        Returns:
            Dictionary containing generation result and metadata
        """
        if not self.gemini_client:
            return {"error": "Gemini API key not configured", "success": False}
            
        try:
            # Construct enhanced prompt for better video generation
            enhanced_prompt = f"{prompt}. Style: {style}. Duration: {duration} seconds. High quality, detailed, smooth motion."
            
            # Generate video using Gemini VEO3
            response = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash-preview-video-generation",
                contents=enhanced_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=['VIDEO'],
                    temperature=0.7
                )
            )
            
            if not response.candidates or not response.candidates[0].content:
                return {"error": "No video content generated", "success": False}
                
            content = response.candidates[0].content
            video_data = None
            
            # Extract video data from response
            for part in content.parts:
                if part.inline_data and part.inline_data.mime_type and part.inline_data.mime_type.startswith('video/'):
                    video_data = part.inline_data.data
                    break
            
            if not video_data:
                return {"error": "No video data found in response", "success": False}
            
            logger.info("Gemini VEO3 video generation successful")
            
            # Estimate cost (approximate pricing for Gemini video generation)
            estimated_cost = duration * 0.05  # ~$0.05 per second
            
            return {
                "success": True,
                "service": "gemini_veo3",
                "video_data": video_data,
                "estimated_cost": estimated_cost,
                "model_used": "gemini-2.0-flash-preview-video-generation",
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "style": style,
                "prompt": prompt
            }
            
        except Exception as e:
            logger.error(f"Gemini VEO3 video generation failed: {str(e)}")
            return {"error": f"Video generation failed: {str(e)}", "success": False}

    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all multimedia services"""
        availability = self.check_service_availability()
        
        status = {
            "services": {
                "suno": {
                    "available": availability["suno"],
                    "name": "Suno AI Music Generation",
                    "endpoint": self.suno_endpoint if availability["suno"] else None,
                    "models": ["chirp-v4", "chirp-v3-5", "chirp-v3-0"] if availability["suno"] else [],
                    "estimated_cost": "$0.01 per generation"
                },
                "ideogram": {
                    "available": availability["ideogram"],
                    "name": "Ideogram AI Image Generation",
                    "endpoint": self.ideogram_endpoint if availability["ideogram"] else None,
                    "models": ["V_3", "V_2", "V_2_TURBO"] if availability["ideogram"] else [],
                    "estimated_cost": "$0.025-0.10 per image"
                },
                "dalle3": {
                    "available": availability["openai_dalle"],
                    "name": "OpenAI DALL-E 3",
                    "endpoint": "OpenAI API" if availability["openai_dalle"] else None,
                    "models": ["dall-e-3"] if availability["openai_dalle"] else [],
                    "estimated_cost": "$0.04-0.12 per image"
                },
                "storytelling": {
                    "available": availability.get("openai_storytelling", False),
                    "name": "OpenAI Storytelling & Script Generation",
                    "endpoint": "OpenAI API" if availability.get("openai_storytelling") else None,
                    "models": ["gpt-5", "gpt-4o", "gpt-4"] if availability.get("openai_storytelling") else [],
                    "estimated_cost": "$0.005-0.015 per 1K tokens",
                    "capabilities": ["Script Generation", "Storyboard Descriptions", "Character Profiles", "Prompt Enhancement"] if availability.get("openai_storytelling") else []
                }
            },
            "total_available": sum(availability.values()),
            "missing_keys": [k for k, v in availability.items() if not v]
        }
        
        # Add Gemini VEO3 service info if available
        if availability.get("gemini_veo3"):
            status["services"]["gemini_veo3"] = {
                "available": True,
                "name": "Gemini VEO3 Video Generation", 
                "models": ["gemini-2.0-flash-preview-video-generation"],
                "estimated_cost": "~$0.05 per second",
                "max_duration": "60 seconds",
                "supported_formats": ["MP4", "WebM"],
                "aspect_ratios": ["16:9", "9:16", "1:1"]
            }
        
        return status

# Global service instance
multimedia_service = MultimediaGenerationService()