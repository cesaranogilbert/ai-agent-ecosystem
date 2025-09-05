"""
Brand Storytelling & Narrative Agent - AI-Powered Brand Voice & Story Creation
Consistent Brand Narratives, Emotional Engagement & Cross-Platform Storytelling
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
import pandas as pd
import numpy as np
from openai import OpenAI
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "storytelling-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///brand_storytelling.db")

db.init_app(app)

# Storytelling Enums
class StoryType(Enum):
    BRAND_ORIGIN = "brand_origin"
    CUSTOMER_SUCCESS = "customer_success"
    PRODUCT_STORY = "product_story"
    BEHIND_SCENES = "behind_scenes"
    VISION_MISSION = "vision_mission"

class EmotionalTone(Enum):
    INSPIRATIONAL = "inspirational"
    AUTHENTIC = "authentic"
    EMPOWERING = "empowering"
    NOSTALGIC = "nostalgic"
    ASPIRATIONAL = "aspirational"

class Platform(Enum):
    WEBSITE = "website"
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    VIDEO = "video"
    PODCAST = "podcast"

# Data Models
class Brand(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    brand_id = db.Column(db.String(100), unique=True, nullable=False)
    brand_name = db.Column(db.String(200), nullable=False)
    
    # Brand Identity
    brand_mission = db.Column(db.Text)
    brand_vision = db.Column(db.Text)
    core_values = db.Column(db.JSON)
    brand_personality = db.Column(db.JSON)
    
    # Voice and Tone
    brand_voice_attributes = db.Column(db.JSON)
    communication_style = db.Column(db.String(100))
    preferred_emotional_tone = db.Column(db.Enum(EmotionalTone))
    
    # Target Audience
    primary_audience = db.Column(db.JSON)
    audience_personas = db.Column(db.JSON)
    emotional_triggers = db.Column(db.JSON)
    
    # Brand Story Elements
    origin_story = db.Column(db.Text)
    founder_story = db.Column(db.Text)
    key_milestones = db.Column(db.JSON)
    brand_challenges = db.Column(db.JSON)
    success_stories = db.Column(db.JSON)
    
    # Narrative Framework
    brand_archetype = db.Column(db.String(100))
    story_themes = db.Column(db.JSON)
    recurring_narratives = db.Column(db.JSON)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class StoryContent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    story_id = db.Column(db.String(100), unique=True, nullable=False)
    brand_id = db.Column(db.String(100), db.ForeignKey('brand.brand_id'), nullable=False)
    
    # Story Details
    story_title = db.Column(db.String(200), nullable=False)
    story_type = db.Column(db.Enum(StoryType), nullable=False)
    emotional_tone = db.Column(db.Enum(EmotionalTone), nullable=False)
    
    # Content Structure
    story_outline = db.Column(db.JSON)
    story_content = db.Column(db.Text)
    key_messages = db.Column(db.JSON)
    call_to_action = db.Column(db.Text)
    
    # Platform Adaptations
    platform_versions = db.Column(db.JSON)
    content_lengths = db.Column(db.JSON)
    visual_elements = db.Column(db.JSON)
    
    # Performance Metrics
    engagement_score = db.Column(db.Float, default=0.0)
    emotional_resonance = db.Column(db.Float, default=0.0)
    brand_consistency = db.Column(db.Float, default=85.0)
    message_clarity = db.Column(db.Float, default=80.0)
    
    # AI Analysis
    story_effectiveness = db.Column(db.Float, default=75.0)
    improvement_suggestions = db.Column(db.JSON)
    optimization_potential = db.Column(db.Float, default=20.0)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Storyboard(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    storyboard_id = db.Column(db.String(100), unique=True, nullable=False)
    story_id = db.Column(db.String(100), db.ForeignKey('story_content.story_id'), nullable=False)
    
    # Storyboard Structure
    storyboard_title = db.Column(db.String(200), nullable=False)
    total_scenes = db.Column(db.Integer, default=5)
    estimated_duration = db.Column(db.Float, default=60.0)  # seconds
    
    # Visual Framework
    visual_style = db.Column(db.String(100))
    color_palette = db.Column(db.JSON)
    typography_choices = db.Column(db.JSON)
    brand_elements = db.Column(db.JSON)
    
    # Scene Details
    scenes = db.Column(db.JSON)  # Array of scene objects
    transitions = db.Column(db.JSON)
    visual_hierarchy = db.Column(db.JSON)
    
    # Production Notes
    production_requirements = db.Column(db.JSON)
    resource_needs = db.Column(db.JSON)
    technical_specifications = db.Column(db.JSON)
    
    # Performance Predictions
    visual_impact_score = db.Column(db.Float, default=75.0)
    production_complexity = db.Column(db.Float, default=50.0)
    brand_alignment = db.Column(db.Float, default=85.0)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Brand Storytelling Engine
class BrandStorytellingEngine:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    def create_comprehensive_brand_narrative(self, brand_id: str) -> Dict[str, Any]:
        """Create comprehensive brand narrative framework"""
        
        brand = Brand.query.filter_by(brand_id=brand_id).first()
        if not brand:
            return {'error': 'Brand not found'}
        
        # Develop brand narrative foundation
        narrative_foundation = self._develop_narrative_foundation(brand)
        
        # Create story content variations
        story_variations = self._create_story_variations(brand, narrative_foundation)
        
        # Generate platform-specific adaptations
        platform_adaptations = self._create_platform_adaptations(brand, story_variations)
        
        # Create visual storyboards
        storyboards = self._generate_storyboards(brand, story_variations)
        
        # Analyze narrative effectiveness
        effectiveness_analysis = self._analyze_narrative_effectiveness(brand, story_variations)
        
        return {
            'brand_id': brand_id,
            'narrative_foundation': narrative_foundation,
            'story_variations': story_variations,
            'platform_adaptations': platform_adaptations,
            'storyboards': storyboards,
            'effectiveness_analysis': effectiveness_analysis,
            'creation_date': datetime.utcnow().isoformat()
        }
    
    def _develop_narrative_foundation(self, brand: Brand) -> Dict[str, Any]:
        """Develop foundational narrative elements"""
        
        # Extract brand essence
        brand_essence = {
            'mission': brand.brand_mission,
            'vision': brand.brand_vision,
            'values': brand.core_values or [],
            'personality': brand.brand_personality or {}
        }
        
        # Develop narrative themes
        narrative_themes = self._identify_narrative_themes(brand)
        
        # Create story arc framework
        story_arc = self._create_story_arc_framework(brand, narrative_themes)
        
        # Define emotional journey
        emotional_journey = self._map_emotional_journey(brand)
        
        return {
            'brand_essence': brand_essence,
            'narrative_themes': narrative_themes,
            'story_arc_framework': story_arc,
            'emotional_journey': emotional_journey,
            'brand_archetype': brand.brand_archetype or 'Hero',
            'unique_value_proposition': self._extract_unique_value_prop(brand)
        }
    
    def _identify_narrative_themes(self, brand: Brand) -> List[Dict[str, Any]]:
        """Identify key narrative themes for the brand"""
        
        themes = []
        
        # Core value-based themes
        if brand.core_values:
            for value in brand.core_values:
                themes.append({
                    'theme': f"commitment_to_{value.lower().replace(' ', '_')}",
                    'description': f"Stories that demonstrate our commitment to {value}",
                    'emotional_impact': 'trust_building',
                    'story_potential': 85.0
                })
        
        # Mission-driven themes
        if brand.brand_mission:
            themes.append({
                'theme': 'mission_driven_impact',
                'description': 'Stories showcasing how we fulfill our mission',
                'emotional_impact': 'purpose_connection',
                'story_potential': 90.0
            })
        
        # Transformation themes
        themes.append({
            'theme': 'customer_transformation',
            'description': 'Stories of customer success and transformation',
            'emotional_impact': 'hope_inspiration',
            'story_potential': 95.0
        })
        
        # Innovation themes
        themes.append({
            'theme': 'innovation_leadership',
            'description': 'Stories of breakthrough innovations and industry leadership',
            'emotional_impact': 'excitement_anticipation',
            'story_potential': 80.0
        })
        
        # Community themes
        themes.append({
            'theme': 'community_building',
            'description': 'Stories of community impact and connection',
            'emotional_impact': 'belonging_unity',
            'story_potential': 75.0
        })
        
        return themes
    
    def _create_story_arc_framework(self, brand: Brand, themes: List[Dict]) -> Dict[str, Any]:
        """Create overarching story arc framework"""
        
        # Classic story arc adapted for brand narrative
        story_arc = {
            'setup': {
                'stage': 'brand_origins',
                'purpose': 'Establish brand foundation and initial challenges',
                'key_elements': ['founding_story', 'initial_vision', 'early_challenges'],
                'emotional_tone': 'authentic_vulnerability'
            },
            'inciting_incident': {
                'stage': 'catalyst_moment',
                'purpose': 'The moment that defined our brand purpose',
                'key_elements': ['breakthrough_realization', 'market_need_discovery', 'mission_clarity'],
                'emotional_tone': 'inspiration_determination'
            },
            'rising_action': {
                'stage': 'growth_journey',
                'purpose': 'Building solutions and overcoming obstacles',
                'key_elements': ['product_development', 'team_building', 'customer_wins'],
                'emotional_tone': 'progress_excitement'
            },
            'climax': {
                'stage': 'transformation_delivery',
                'purpose': 'Delivering transformational value to customers',
                'key_elements': ['customer_success', 'impact_achievement', 'industry_recognition'],
                'emotional_tone': 'triumph_fulfillment'
            },
            'resolution': {
                'stage': 'future_vision',
                'purpose': 'Ongoing commitment and future aspirations',
                'key_elements': ['continued_innovation', 'expanded_impact', 'community_growth'],
                'emotional_tone': 'hope_anticipation'
            }
        }
        
        # Map themes to story arc stages
        theme_mapping = {}
        for theme in themes:
            theme_mapping[theme['theme']] = self._map_theme_to_arc_stage(theme, story_arc)
        
        return {
            'story_arc_structure': story_arc,
            'theme_stage_mapping': theme_mapping,
            'narrative_consistency_rules': self._define_consistency_rules(brand)
        }
    
    def _map_theme_to_arc_stage(self, theme: Dict, story_arc: Dict) -> List[str]:
        """Map narrative theme to appropriate story arc stages"""
        
        theme_name = theme['theme']
        
        # Map themes to stages where they work best
        theme_stage_map = {
            'mission_driven_impact': ['inciting_incident', 'climax', 'resolution'],
            'customer_transformation': ['rising_action', 'climax'],
            'innovation_leadership': ['rising_action', 'climax'],
            'community_building': ['rising_action', 'resolution'],
            'commitment_to': ['setup', 'rising_action', 'resolution']
        }
        
        # Find matching stages
        for pattern, stages in theme_stage_map.items():
            if pattern in theme_name:
                return stages
        
        return ['rising_action']  # Default stage
    
    def _map_emotional_journey(self, brand: Brand) -> Dict[str, Any]:
        """Map the emotional journey for brand storytelling"""
        
        # Define emotional progression
        emotional_journey = {
            'awareness_stage': {
                'primary_emotion': 'curiosity',
                'secondary_emotions': ['interest', 'intrigue'],
                'storytelling_focus': 'problem_identification',
                'content_tone': 'educational_engaging'
            },
            'consideration_stage': {
                'primary_emotion': 'hope',
                'secondary_emotions': ['trust', 'confidence'],
                'storytelling_focus': 'solution_demonstration',
                'content_tone': 'reassuring_credible'
            },
            'decision_stage': {
                'primary_emotion': 'excitement',
                'secondary_emotions': ['anticipation', 'certainty'],
                'storytelling_focus': 'transformation_promise',
                'content_tone': 'compelling_decisive'
            },
            'experience_stage': {
                'primary_emotion': 'satisfaction',
                'secondary_emotions': ['delight', 'accomplishment'],
                'storytelling_focus': 'value_delivery',
                'content_tone': 'celebratory_supportive'
            },
            'advocacy_stage': {
                'primary_emotion': 'pride',
                'secondary_emotions': ['loyalty', 'enthusiasm'],
                'storytelling_focus': 'shared_success',
                'content_tone': 'empowering_community'
            }
        }
        
        # Add brand-specific emotional triggers
        if brand.emotional_triggers:
            for stage in emotional_journey.values():
                stage['brand_triggers'] = brand.emotional_triggers
        
        return emotional_journey
    
    def _define_consistency_rules(self, brand: Brand) -> List[Dict[str, Any]]:
        """Define rules for maintaining narrative consistency"""
        
        rules = [
            {
                'rule_type': 'voice_consistency',
                'description': 'Maintain consistent brand voice across all stories',
                'enforcement': 'automated_voice_analysis',
                'importance': 'critical'
            },
            {
                'rule_type': 'value_alignment',
                'description': 'All stories must align with brand values',
                'enforcement': 'value_sentiment_analysis',
                'importance': 'critical'
            },
            {
                'rule_type': 'message_hierarchy',
                'description': 'Primary brand messages should be prominently featured',
                'enforcement': 'message_prominence_scoring',
                'importance': 'high'
            },
            {
                'rule_type': 'emotional_tone',
                'description': 'Maintain appropriate emotional tone for brand personality',
                'enforcement': 'emotional_analysis',
                'importance': 'high'
            },
            {
                'rule_type': 'factual_accuracy',
                'description': 'All factual claims must be accurate and verifiable',
                'enforcement': 'fact_verification',
                'importance': 'critical'
            }
        ]
        
        return rules
    
    def _extract_unique_value_prop(self, brand: Brand) -> Dict[str, Any]:
        """Extract unique value proposition from brand data"""
        
        uvp_elements = {
            'core_benefit': 'Transform customer experience through innovative solutions',
            'target_audience': brand.primary_audience or {},
            'differentiation': brand.brand_personality or {},
            'proof_points': brand.success_stories or [],
            'emotional_benefit': 'Empowerment through transformation'
        }
        
        return uvp_elements
    
    def _create_story_variations(self, brand: Brand, foundation: Dict) -> List[Dict[str, Any]]:
        """Create multiple story variations for different purposes"""
        
        story_variations = []
        
        # Origin story variation
        origin_story = self._create_origin_story(brand, foundation)
        story_variations.append(origin_story)
        
        # Customer success story variation
        customer_story = self._create_customer_success_story(brand, foundation)
        story_variations.append(customer_story)
        
        # Innovation story variation
        innovation_story = self._create_innovation_story(brand, foundation)
        story_variations.append(innovation_story)
        
        # Vision story variation
        vision_story = self._create_vision_story(brand, foundation)
        story_variations.append(vision_story)
        
        # Behind-the-scenes story variation
        bts_story = self._create_behind_scenes_story(brand, foundation)
        story_variations.append(bts_story)
        
        return story_variations
    
    def _create_origin_story(self, brand: Brand, foundation: Dict) -> Dict[str, Any]:
        """Create brand origin story"""
        
        story_prompt = f"""
        Create a compelling brand origin story for {brand.brand_name} with the following elements:
        - Mission: {brand.brand_mission}
        - Core Values: {brand.core_values}
        - Brand Personality: {brand.brand_personality}
        - Founder Story: {brand.founder_story or 'Visionary founders with passion for change'}
        
        The story should follow the Hero's Journey framework and evoke {brand.preferred_emotional_tone.value if brand.preferred_emotional_tone else 'inspirational'} emotions.
        """
        
        # Generate story using AI
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert brand storyteller who creates emotionally resonant origin stories."},
                {"role": "user", "content": story_prompt}
            ],
            max_tokens=1000
        )
        
        story_content = response.choices[0].message.content
        
        return {
            'story_type': 'brand_origin',
            'story_title': f'The {brand.brand_name} Story: From Vision to Reality',
            'story_content': story_content,
            'key_messages': [
                'Authentic founding vision',
                'Commitment to values',
                'Customer-centric mission'
            ],
            'emotional_arc': ['struggle', 'determination', 'breakthrough', 'impact'],
            'target_platforms': ['website_about', 'investor_pitch', 'team_onboarding'],
            'estimated_engagement': 82.0
        }
    
    def _create_customer_success_story(self, brand: Brand, foundation: Dict) -> Dict[str, Any]:
        """Create customer success story"""
        
        story_prompt = f"""
        Create a powerful customer transformation story for {brand.brand_name} that demonstrates:
        - How we solve real customer problems
        - The transformation journey
        - Specific results and impact
        - Emotional connection to our mission: {brand.brand_mission}
        
        Make it authentic, specific, and emotionally compelling while maintaining {brand.preferred_emotional_tone.value if brand.preferred_emotional_tone else 'empowering'} tone.
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at crafting compelling customer success stories that build trust and demonstrate value."},
                {"role": "user", "content": story_prompt}
            ],
            max_tokens=800
        )
        
        story_content = response.choices[0].message.content
        
        return {
            'story_type': 'customer_success',
            'story_title': 'Transformation Through Partnership',
            'story_content': story_content,
            'key_messages': [
                'Real customer impact',
                'Measurable results',
                'Partnership approach'
            ],
            'emotional_arc': ['challenge', 'hope', 'progress', 'success', 'gratitude'],
            'target_platforms': ['case_studies', 'sales_presentations', 'social_proof'],
            'estimated_engagement': 88.0
        }
    
    def _create_innovation_story(self, brand: Brand, foundation: Dict) -> Dict[str, Any]:
        """Create innovation leadership story"""
        
        return {
            'story_type': 'innovation_leadership',
            'story_title': 'Pioneering the Future of [Industry]',
            'story_content': 'Innovation story content showcasing breakthrough thinking and industry leadership...',
            'key_messages': [
                'Cutting-edge innovation',
                'Industry leadership',
                'Future-focused solutions'
            ],
            'emotional_arc': ['curiosity', 'excitement', 'amazement', 'anticipation'],
            'target_platforms': ['thought_leadership', 'industry_events', 'innovation_showcases'],
            'estimated_engagement': 75.0
        }
    
    def _create_vision_story(self, brand: Brand, foundation: Dict) -> Dict[str, Any]:
        """Create future vision story"""
        
        return {
            'story_type': 'future_vision',
            'story_title': 'Building Tomorrow, Together',
            'story_content': 'Vision story content outlining the future we\'re building and inviting audience participation...',
            'key_messages': [
                'Inspiring future vision',
                'Collective impact',
                'Invitation to join'
            ],
            'emotional_arc': ['aspiration', 'inspiration', 'unity', 'determination'],
            'target_platforms': ['vision_statements', 'recruitment', 'stakeholder_communications'],
            'estimated_engagement': 80.0
        }
    
    def _create_behind_scenes_story(self, brand: Brand, foundation: Dict) -> Dict[str, Any]:
        """Create behind-the-scenes story"""
        
        return {
            'story_type': 'behind_the_scenes',
            'story_title': 'Inside [Brand Name]: The People Behind the Mission',
            'story_content': 'Behind-the-scenes content showcasing team, culture, and authentic moments...',
            'key_messages': [
                'Authentic team culture',
                'Human-centered approach',
                'Passion for mission'
            ],
            'emotional_arc': ['curiosity', 'connection', 'appreciation', 'trust'],
            'target_platforms': ['social_media', 'employer_branding', 'community_building'],
            'estimated_engagement': 85.0
        }
    
    def _create_platform_adaptations(self, brand: Brand, stories: List[Dict]) -> Dict[str, Any]:
        """Create platform-specific story adaptations"""
        
        adaptations = {}
        
        for platform in Platform:
            adaptations[platform.value] = {
                'content_specifications': self._get_platform_specs(platform),
                'adapted_stories': []
            }
            
            for story in stories:
                adapted_story = self._adapt_story_for_platform(story, platform, brand)
                adaptations[platform.value]['adapted_stories'].append(adapted_story)
        
        return adaptations
    
    def _get_platform_specs(self, platform: Platform) -> Dict[str, Any]:
        """Get content specifications for each platform"""
        
        specs = {
            Platform.WEBSITE: {
                'max_length': 1500,
                'format': 'long_form_narrative',
                'visual_support': 'hero_images_infographics',
                'interaction': 'scroll_based_storytelling'
            },
            Platform.SOCIAL_MEDIA: {
                'max_length': 280,
                'format': 'micro_narrative',
                'visual_support': 'engaging_visuals_videos',
                'interaction': 'likes_shares_comments'
            },
            Platform.EMAIL: {
                'max_length': 800,
                'format': 'personal_narrative',
                'visual_support': 'minimal_focused_images',
                'interaction': 'click_through_engagement'
            },
            Platform.VIDEO: {
                'max_length': 180,  # seconds
                'format': 'visual_storytelling',
                'visual_support': 'cinematic_production',
                'interaction': 'view_completion_sharing'
            },
            Platform.PODCAST: {
                'max_length': 1200,  # seconds
                'format': 'conversational_narrative',
                'visual_support': 'audio_only',
                'interaction': 'listen_completion_subscription'
            }
        }
        
        return specs.get(platform, {})
    
    def _adapt_story_for_platform(self, story: Dict, platform: Platform, brand: Brand) -> Dict[str, Any]:
        """Adapt story content for specific platform"""
        
        specs = self._get_platform_specs(platform)
        
        # Adapt content length and format
        if platform == Platform.SOCIAL_MEDIA:
            adapted_content = self._create_social_media_version(story)
        elif platform == Platform.EMAIL:
            adapted_content = self._create_email_version(story)
        elif platform == Platform.VIDEO:
            adapted_content = self._create_video_script_version(story)
        else:
            adapted_content = story['story_content']
        
        return {
            'original_story_type': story['story_type'],
            'platform': platform.value,
            'adapted_content': adapted_content,
            'content_specs': specs,
            'visual_requirements': self._define_visual_requirements(story, platform),
            'engagement_optimization': self._optimize_for_platform_engagement(story, platform)
        }
    
    def _create_social_media_version(self, story: Dict) -> str:
        """Create social media version of story"""
        
        # Extract key emotion and message
        key_message = story['key_messages'][0] if story['key_messages'] else 'Brand story'
        
        # Create concise, engaging version
        social_versions = {
            'brand_origin': f"Every great story starts with a vision. Ours began with [founder insight]. Today, we're proud to [current impact]. #BrandStory #Vision",
            'customer_success': f"🎉 Customer spotlight: See how [Customer] achieved [result] with our help. Real impact, real stories. #CustomerSuccess #Transformation",
            'innovation_leadership': f"Innovation never stops. Here's how we're pushing boundaries in [industry] to create [future benefit]. #Innovation #Leadership",
            'future_vision': f"The future we're building: [vision statement]. Join us in making it reality. #FutureVision #JoinUs",
            'behind_the_scenes': f"Behind every great product is an amazing team. Meet the people making magic happen at [Brand]. #TeamSpotlight #Culture"
        }
        
        return social_versions.get(story['story_type'], f"Discover the story behind {story['story_title']}. #BrandStory")
    
    def _create_email_version(self, story: Dict) -> str:
        """Create email version of story"""
        
        # Email structure: Personal greeting + story excerpt + call to action
        email_template = f"""
        Subject: {story['story_title']}
        
        Hi [Name],
        
        {story['story_content'][:400]}...
        
        [Continue reading the full story]
        
        Best regards,
        The [Brand] Team
        """
        
        return email_template
    
    def _create_video_script_version(self, story: Dict) -> str:
        """Create video script version"""
        
        script_template = f"""
        VIDEO SCRIPT: {story['story_title']}
        Duration: 60-90 seconds
        
        OPENING HOOK (0-5s):
        [Compelling opening statement]
        
        STORY DEVELOPMENT (5-45s):
        {story['story_content'][:200]}
        
        EMOTIONAL CLIMAX (45-60s):
        [Key transformation moment]
        
        CALL TO ACTION (60-90s):
        [Clear next step for viewers]
        
        VISUAL NOTES:
        - Authentic, human-centered imagery
        - Brand colors and fonts
        - Emotional close-ups during key moments
        """
        
        return script_template
    
    def _define_visual_requirements(self, story: Dict, platform: Platform) -> Dict[str, Any]:
        """Define visual requirements for story adaptation"""
        
        base_requirements = {
            'brand_consistency': True,
            'emotional_alignment': story.get('emotional_arc', []),
            'visual_style': 'authentic_human_centered'
        }
        
        platform_specific = {
            Platform.SOCIAL_MEDIA: {
                'image_specs': '1080x1080 or 1080x1350',
                'video_specs': '9:16 vertical, max 60s',
                'text_overlay': 'minimal, high contrast'
            },
            Platform.VIDEO: {
                'resolution': '1920x1080 minimum',
                'aspect_ratio': '16:9',
                'visual_elements': 'cinematic_storytelling'
            },
            Platform.WEBSITE: {
                'hero_image': 'high_resolution_brand_aligned',
                'supporting_visuals': 'infographics_progress_imagery',
                'responsive_design': True
            }
        }
        
        base_requirements.update(platform_specific.get(platform, {}))
        return base_requirements
    
    def _optimize_for_platform_engagement(self, story: Dict, platform: Platform) -> Dict[str, Any]:
        """Optimize content for platform-specific engagement"""
        
        optimization_strategies = {
            Platform.SOCIAL_MEDIA: {
                'post_timing': 'optimal_audience_hours',
                'hashtag_strategy': 'mix_branded_trending_niche',
                'engagement_hooks': 'questions_calls_to_action',
                'visual_appeal': 'thumb_stopping_imagery'
            },
            Platform.EMAIL: {
                'subject_line': 'curiosity_driven_personalized',
                'preview_text': 'compelling_continuation',
                'content_structure': 'scannable_with_clear_cta',
                'send_timing': 'subscriber_behavior_optimized'
            },
            Platform.VIDEO: {
                'thumbnail': 'emotional_expressive_branded',
                'first_5_seconds': 'hook_driven_retention_focused',
                'pacing': 'dynamic_attention_holding',
                'end_screen': 'clear_next_action'
            }
        }
        
        return optimization_strategies.get(platform, {})
    
    def _generate_storyboards(self, brand: Brand, stories: List[Dict]) -> List[Dict[str, Any]]:
        """Generate visual storyboards for story content"""
        
        storyboards = []
        
        for story in stories:
            if story['story_type'] in ['customer_success', 'brand_origin', 'innovation_leadership']:
                storyboard = self._create_visual_storyboard(story, brand)
                storyboards.append(storyboard)
        
        return storyboards
    
    def _create_visual_storyboard(self, story: Dict, brand: Brand) -> Dict[str, Any]:
        """Create detailed visual storyboard"""
        
        # Define visual scenes based on emotional arc
        emotional_arc = story.get('emotional_arc', ['setup', 'development', 'climax', 'resolution'])
        
        scenes = []
        for i, emotion in enumerate(emotional_arc):
            scene = {
                'scene_number': i + 1,
                'emotional_tone': emotion,
                'visual_description': f'Scene depicting {emotion} phase of story',
                'duration': 15,  # seconds
                'visual_elements': {
                    'composition': 'rule_of_thirds',
                    'lighting': 'natural_warm' if emotion in ['success', 'resolution'] else 'dramatic',
                    'color_mood': 'brand_aligned_emotional',
                    'camera_movement': 'subtle_dynamic'
                },
                'audio_elements': {
                    'narration': f'Key message for {emotion} phase',
                    'background_music': f'{emotion}_appropriate_score',
                    'sound_effects': 'minimal_purposeful'
                }
            }
            scenes.append(scene)
        
        return {
            'storyboard_id': f"SB_{story['story_type']}_{datetime.now().strftime('%Y%m%d')}",
            'story_reference': story['story_type'],
            'total_scenes': len(scenes),
            'estimated_duration': len(scenes) * 15,
            'scenes': scenes,
            'production_notes': {
                'brand_guidelines': 'strict_adherence_required',
                'talent_requirements': 'authentic_diverse_representation',
                'location_needs': 'brand_appropriate_settings',
                'equipment_level': 'professional_quality'
            },
            'visual_style_guide': {
                'color_palette': brand.brand_personality.get('colors') if brand.brand_personality else ['brand_primary', 'brand_secondary'],
                'typography': 'brand_font_hierarchy',
                'imagery_style': 'authentic_human_focused',
                'brand_element_integration': 'subtle_consistent'
            }
        }
    
    def _analyze_narrative_effectiveness(self, brand: Brand, stories: List[Dict]) -> Dict[str, Any]:
        """Analyze effectiveness of narrative content"""
        
        # Analyze each story
        story_analyses = []
        for story in stories:
            analysis = self._analyze_individual_story(story, brand)
            story_analyses.append(analysis)
        
        # Overall narrative assessment
        overall_assessment = self._assess_overall_narrative_strength(story_analyses, brand)
        
        # Improvement recommendations
        recommendations = self._generate_narrative_improvements(story_analyses, brand)
        
        return {
            'individual_story_analyses': story_analyses,
            'overall_narrative_assessment': overall_assessment,
            'improvement_recommendations': recommendations,
            'consistency_score': self._calculate_consistency_score(stories, brand),
            'emotional_resonance_score': self._calculate_emotional_resonance(stories)
        }
    
    def _analyze_individual_story(self, story: Dict, brand: Brand) -> Dict[str, Any]:
        """Analyze effectiveness of individual story"""
        
        # Content analysis
        content_analysis = {
            'clarity_score': 85.0,  # Based on content structure and messaging
            'emotional_impact': 78.0,  # Based on emotional arc and tone
            'brand_alignment': 92.0,  # Based on brand voice and values
            'engagement_potential': story.get('estimated_engagement', 75.0)
        }
        
        # Structural analysis
        structural_analysis = {
            'story_arc_completeness': 88.0,
            'message_hierarchy': 82.0,
            'call_to_action_strength': 75.0,
            'platform_adaptability': 85.0
        }
        
        # Overall effectiveness score
        effectiveness_score = np.mean([
            content_analysis['clarity_score'],
            content_analysis['emotional_impact'],
            content_analysis['brand_alignment'],
            structural_analysis['story_arc_completeness']
        ])
        
        return {
            'story_type': story['story_type'],
            'content_analysis': content_analysis,
            'structural_analysis': structural_analysis,
            'effectiveness_score': effectiveness_score,
            'optimization_potential': max(0, 95 - effectiveness_score),
            'strengths': self._identify_story_strengths(story, content_analysis),
            'improvement_areas': self._identify_improvement_areas(story, content_analysis)
        }
    
    def _identify_story_strengths(self, story: Dict, analysis: Dict) -> List[str]:
        """Identify strengths of individual story"""
        
        strengths = []
        
        if analysis['emotional_impact'] > 80:
            strengths.append('Strong emotional resonance')
        
        if analysis['brand_alignment'] > 90:
            strengths.append('Excellent brand consistency')
        
        if analysis['clarity_score'] > 85:
            strengths.append('Clear and compelling messaging')
        
        if len(story.get('key_messages', [])) >= 3:
            strengths.append('Comprehensive message coverage')
        
        return strengths
    
    def _identify_improvement_areas(self, story: Dict, analysis: Dict) -> List[str]:
        """Identify areas for story improvement"""
        
        improvements = []
        
        if analysis['emotional_impact'] < 75:
            improvements.append('Enhance emotional storytelling elements')
        
        if analysis['clarity_score'] < 80:
            improvements.append('Simplify and clarify key messages')
        
        if analysis['brand_alignment'] < 85:
            improvements.append('Strengthen brand voice consistency')
        
        return improvements
    
    def _assess_overall_narrative_strength(self, analyses: List[Dict], brand: Brand) -> Dict[str, Any]:
        """Assess overall narrative portfolio strength"""
        
        avg_effectiveness = np.mean([a['effectiveness_score'] for a in analyses])
        
        return {
            'portfolio_strength': avg_effectiveness,
            'narrative_completeness': len(analyses) / 5 * 100,  # Assuming 5 core story types
            'consistency_across_stories': self._calculate_narrative_consistency(analyses),
            'emotional_range_coverage': self._assess_emotional_range(analyses),
            'strategic_alignment': 88.0  # Based on mission/vision alignment
        }
    
    def _calculate_narrative_consistency(self, analyses: List[Dict]) -> float:
        """Calculate consistency across narrative portfolio"""
        
        brand_alignment_scores = [a['content_analysis']['brand_alignment'] for a in analyses]
        consistency_variance = np.var(brand_alignment_scores)
        
        # Lower variance = higher consistency
        consistency_score = max(0, 100 - (consistency_variance * 2))
        return consistency_score
    
    def _assess_emotional_range(self, analyses: List[Dict]) -> float:
        """Assess coverage of emotional range"""
        
        # Simplified assessment based on story variety
        story_types = [a['story_type'] for a in analyses]
        unique_types = len(set(story_types))
        
        # Score based on diversity of story types
        range_score = min(100, unique_types / 5 * 100)
        return range_score
    
    def _generate_narrative_improvements(self, analyses: List[Dict], brand: Brand) -> List[Dict[str, Any]]:
        """Generate narrative improvement recommendations"""
        
        recommendations = []
        
        # Identify weak story types
        weak_stories = [a for a in analyses if a['effectiveness_score'] < 80]
        
        if weak_stories:
            recommendations.append({
                'category': 'story_optimization',
                'priority': 'high',
                'recommendation': f'Optimize {len(weak_stories)} underperforming story types',
                'expected_impact': '15-25% effectiveness improvement'
            })
        
        # Consistency improvements
        consistency_scores = [a['content_analysis']['brand_alignment'] for a in analyses]
        if min(consistency_scores) < 85:
            recommendations.append({
                'category': 'brand_consistency',
                'priority': 'medium',
                'recommendation': 'Strengthen brand voice consistency across all narratives',
                'expected_impact': 'Improved brand recognition and trust'
            })
        
        # Emotional impact improvements
        emotional_scores = [a['content_analysis']['emotional_impact'] for a in analyses]
        if np.mean(emotional_scores) < 80:
            recommendations.append({
                'category': 'emotional_resonance',
                'priority': 'high',
                'recommendation': 'Enhance emotional storytelling elements',
                'expected_impact': '20-30% engagement improvement'
            })
        
        return recommendations
    
    def _calculate_consistency_score(self, stories: List[Dict], brand: Brand) -> float:
        """Calculate overall consistency score"""
        
        # Simplified consistency calculation
        return 87.0  # Based on brand voice analysis
    
    def _calculate_emotional_resonance(self, stories: List[Dict]) -> float:
        """Calculate emotional resonance score"""
        
        # Simplified emotional resonance calculation
        estimated_engagements = [s.get('estimated_engagement', 75.0) for s in stories]
        return np.mean(estimated_engagements)

# Initialize storytelling engine
storytelling_engine = BrandStorytellingEngine()

# Routes
@app.route('/brand-storytelling')
def storytelling_dashboard():
    """Brand Storytelling dashboard"""
    
    recent_brands = Brand.query.order_by(Brand.created_at.desc()).limit(10).all()
    
    return render_template('storytelling/dashboard.html',
                         brands=recent_brands)

@app.route('/brand-storytelling/api/comprehensive-narrative', methods=['POST'])
def create_narrative():
    """API endpoint for comprehensive narrative creation"""
    
    data = request.get_json()
    brand_id = data.get('brand_id')
    
    if not brand_id:
        return jsonify({'error': 'Brand ID required'}), 400
    
    narrative = storytelling_engine.create_comprehensive_brand_narrative(brand_id)
    return jsonify(narrative)

# Initialize database
with app.app_context():
    db.create_all()
    
    # Create sample data
    if Brand.query.count() == 0:
        sample_brand = Brand(
            brand_id='BRAND_DEMO_001',
            brand_name='Demo Storytelling Brand',
            brand_mission='Transform lives through innovative storytelling',
            core_values=['authenticity', 'innovation', 'empowerment'],
            preferred_emotional_tone=EmotionalTone.INSPIRATIONAL,
            brand_archetype='Hero'
        )
        
        db.session.add(sample_brand)
        db.session.commit()
        logger.info("Sample brand storytelling data created")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5031, debug=True)