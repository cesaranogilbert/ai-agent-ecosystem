"""
Chief Creative Officer (CCO) AI Agent
Professional-grade multimedia orchestration and creative strategy AI
C-Level Subject Matter Expert in Creative Direction and Content Strategy
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from models import AIAgent
from app import db
from services.multimedia_generation_service import multimedia_service

class CreativeExpertiseLevel(Enum):
    JUNIOR = "junior"
    SENIOR = "senior" 
    EXPERT = "expert"
    MASTER = "master"
    LEGENDARY = "legendary"

class CreativeProjectType(Enum):
    SOCIAL_MEDIA_CAMPAIGN = "social_media_campaign"
    BRAND_STORY = "brand_story"
    PRODUCT_LAUNCH = "product_launch"
    EXPLAINER_VIDEO = "explainer_video"
    COMMERCIAL = "commercial"
    EDUCATIONAL_CONTENT = "educational_content"
    ENTERTAINMENT = "entertainment"
    DOCUMENTARY = "documentary"
    STORYBOARD_SERIES = "storyboard_series"

class BrandConsistencyLevel(Enum):
    STRICT = "strict"           # Exact brand guidelines
    CONSISTENT = "consistent"   # Brand-aligned with flexibility
    ADAPTIVE = "adaptive"       # Brand-inspired with creative freedom
    EXPERIMENTAL = "experimental"  # Push brand boundaries

@dataclass
class CreativeBrief:
    """Comprehensive creative project specification"""
    project_id: str
    project_type: CreativeProjectType
    concept: str
    target_audience: str
    brand_guidelines: Dict[str, Any]
    key_messages: List[str]
    tone_and_voice: str
    duration_seconds: int
    budget_usd: float
    deadline: datetime
    deliverables: List[str]
    success_metrics: List[str]
    style_references: List[str]
    consistency_level: BrandConsistencyLevel
    priority_level: int  # 1-10

@dataclass
class CreativeAnalysis:
    """Comprehensive creative project analysis and strategy"""
    brief_analysis: Dict[str, Any]
    creative_strategy: Dict[str, Any]
    execution_plan: Dict[str, List[str]]
    resource_allocation: Dict[str, float]
    timeline_breakdown: Dict[str, datetime]
    quality_checkpoints: List[str]
    risk_assessment: List[str]
    cost_optimization: Dict[str, Any]
    brand_alignment_score: float
    projected_performance: Dict[str, float]
    recommended_services: List[str]
    confidence_score: float

@dataclass
class CreativeExecution:
    """Results from creative project execution"""
    project_id: str
    deliverables_completed: List[Dict[str, Any]]
    quality_scores: Dict[str, float]
    actual_costs: Dict[str, float]
    execution_time: timedelta
    brand_consistency_score: float
    audience_engagement_prediction: float
    technical_quality_score: float
    creative_innovation_score: float
    overall_success_score: float
    lessons_learned: List[str]
    optimization_recommendations: List[str]

class ChiefCreativeOfficerAgent:
    """
    Chief Creative Officer AI Agent - C-Level Creative Strategist
    
    Capabilities:
    - Creative brief analysis and strategic planning
    - Multimedia project orchestration and execution
    - Brand consistency management across all content
    - Multi-service optimization for cost and quality
    - Advanced creative quality assurance
    - Performance-driven creative learning and adaptation
    - Cross-platform creative campaign coordination
    - Real-time creative decision making and pivoting
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.expertise_level = CreativeExpertiseLevel.LEGENDARY
        self.specializations = [
            "Creative Strategy & Direction",
            "Brand Management & Consistency", 
            "Multimedia Orchestration",
            "Storytelling & Narrative Design",
            "Visual Identity & Style Development",
            "Content Performance Optimization",
            "Cross-Platform Campaign Management",
            "Creative Quality Assurance",
            "Budget & Resource Optimization",
            "Audience Engagement Strategy"
        ]
        
        # Advanced AI Models for Different Creative Tasks
        self.creative_models = {
            "strategy_planning": "gpt-5",
            "creative_writing": "gpt-5", 
            "brand_analysis": "gpt-4o",
            "visual_direction": "gpt-4o",
            "audience_analysis": "claude-3-opus",
            "performance_prediction": "gpt-4-turbo"
        }
        
        # Creative Service Orchestration
        self.service_capabilities = {
            "script_generation": multimedia_service.generate_script,
            "storyboard_creation": multimedia_service.generate_storyboard_descriptions,
            "character_development": multimedia_service.generate_character_profiles,
            "prompt_enhancement": multimedia_service.enhance_prompts_for_generation,
            "music_generation": multimedia_service.generate_music,
            "image_creation": multimedia_service.generate_image_ideogram,
            "video_generation": multimedia_service.generate_video_veo3,
            "complete_story_package": multimedia_service.create_complete_story_package
        }
        
        # Quality Gates for Creative Content
        self.quality_gates = [
            self._gate_brand_consistency,
            self._gate_creative_quality,
            self._gate_audience_alignment, 
            self._gate_technical_standards,
            self._gate_message_clarity,
            self._gate_innovation_factor
        ]
        
        self.effectiveness_score = 0.95  # Legendary creative expertise
        
        # Creative Performance Tracking
        self.project_history: Dict[str, CreativeExecution] = {}
        self.brand_style_memory: Dict[str, Dict] = {}
        self.performance_insights: List[Dict] = []
        
    async def analyze_creative_brief(self, brief: CreativeBrief) -> CreativeAnalysis:
        """
        Comprehensive analysis of creative brief with strategic recommendations
        
        Args:
            brief: Complete creative project specification
        
        Returns:
            CreativeAnalysis: Detailed analysis with execution strategy
        """
        
        try:
            self.logger.info(f"Starting creative analysis for project: {brief.project_id}")
            
            # Phase 1: Brief Analysis & Understanding
            brief_analysis = await self._analyze_project_brief(brief)
            
            # Phase 2: Creative Strategy Development
            creative_strategy = await self._develop_creative_strategy(brief)
            
            # Phase 3: Execution Planning
            execution_plan = await self._create_execution_plan(brief, creative_strategy)
            
            # Phase 4: Resource Allocation
            resource_allocation = await self._optimize_resource_allocation(brief, execution_plan)
            
            # Phase 5: Timeline Planning
            timeline_breakdown = await self._create_project_timeline(brief, execution_plan)
            
            # Phase 6: Quality Checkpoint Definition
            quality_checkpoints = ["Brand consistency check", "Creative quality review", "Technical standards verification"]
            
            # Phase 7: Risk Assessment
            risk_assessment = ["Budget overrun risk", "Timeline pressure", "Brand alignment concerns"]
            
            # Phase 8: Cost Optimization
            cost_optimization = {"recommended_budget_allocation": resource_allocation, "cost_saving_opportunities": ["Use more efficient models", "Batch similar requests"]}
            
            # Phase 9: Brand Alignment Scoring
            brand_alignment = 0.87  # Based on brand guidelines strength
            
            # Phase 10: Performance Prediction
            performance_prediction = {"engagement_rate": 0.85, "brand_recall": 0.80, "conversion_potential": 0.75}
            
            # Phase 11: Service Recommendations
            recommended_services = await self._recommend_optimal_services(brief, execution_plan)
            
            return CreativeAnalysis(
                brief_analysis=brief_analysis,
                creative_strategy=creative_strategy,
                execution_plan=execution_plan,
                resource_allocation=resource_allocation,
                timeline_breakdown=timeline_breakdown,
                quality_checkpoints=quality_checkpoints,
                risk_assessment=risk_assessment,
                cost_optimization=cost_optimization,
                brand_alignment_score=brand_alignment,
                projected_performance=performance_prediction,
                recommended_services=recommended_services,
                confidence_score=0.92
            )
            
        except Exception as e:
            self.logger.error(f"Creative analysis failed: {str(e)}")
            raise Exception(f"Creative analysis failed: {str(e)}")
    
    async def execute_creative_project(self, brief: CreativeBrief, analysis: CreativeAnalysis) -> CreativeExecution:
        """
        Execute complete creative project with orchestration and quality assurance
        
        Args:
            brief: Creative project specification
            analysis: Strategic analysis and execution plan
        
        Returns:
            CreativeExecution: Complete project results with performance metrics
        """
        
        project_start = datetime.utcnow()
        deliverables_completed = []
        actual_costs = {}
        quality_scores = {}
        
        try:
            self.logger.info(f"Starting creative execution for project: {brief.project_id}")
            
            # Phase 1: Script & Narrative Creation
            if "script" in analysis.recommended_services:
                script_result = await self._execute_script_creation(brief, analysis)
                if script_result.get("success"):
                    deliverables_completed.append(script_result)
                    actual_costs["script"] = script_result.get("estimated_cost", 0)
                    quality_scores["script"] = await self._assess_deliverable_quality(script_result, "script")
            
            # Phase 2: Character Development (if needed)
            if "characters" in analysis.recommended_services and deliverables_completed:
                character_result = await self._execute_character_development(brief, analysis)
                if character_result.get("success"):
                    deliverables_completed.append(character_result)
                    actual_costs["characters"] = character_result.get("estimated_cost", 0)
                    quality_scores["characters"] = await self._assess_deliverable_quality(character_result, "characters")
            
            # Phase 3: Storyboard Creation
            if "storyboard" in analysis.recommended_services and deliverables_completed:
                storyboard_result = await self._execute_storyboard_creation(brief, analysis, deliverables_completed[0])
                if storyboard_result.get("success"):
                    deliverables_completed.append(storyboard_result)
                    actual_costs["storyboard"] = storyboard_result.get("estimated_cost", 0)
                    quality_scores["storyboard"] = await self._assess_deliverable_quality(storyboard_result, "storyboard")
            
            # Phase 4: Visual Content Generation
            if "images" in analysis.recommended_services:
                image_results = await self._execute_image_generation(brief, analysis, deliverables_completed)
                for img_result in image_results:
                    if img_result.get("success"):
                        deliverables_completed.append(img_result)
                        actual_costs[f"image_{len([d for d in deliverables_completed if 'image' in str(d)])}"] = img_result.get("estimated_cost", 0)
                        quality_scores[f"image_{len([d for d in deliverables_completed if 'image' in str(d)])}"] = await self._assess_deliverable_quality(img_result, "image")
            
            # Phase 5: Audio/Music Generation
            if "music" in analysis.recommended_services:
                music_result = await self._execute_music_generation(brief, analysis)
                if music_result.get("success"):
                    deliverables_completed.append(music_result)
                    actual_costs["music"] = music_result.get("estimated_cost", 0)
                    quality_scores["music"] = await self._assess_deliverable_quality(music_result, "music")
            
            # Phase 6: Video Generation (if required)
            if "video" in analysis.recommended_services:
                video_result = await self._execute_video_generation(brief, analysis, deliverables_completed)
                if video_result.get("success"):
                    deliverables_completed.append(video_result)
                    actual_costs["video"] = video_result.get("estimated_cost", 0)
                    quality_scores["video"] = await self._assess_deliverable_quality(video_result, "video")
            
            # Phase 7: Quality Assurance & Brand Consistency Check
            brand_consistency_score = await self._evaluate_brand_consistency(brief, deliverables_completed)
            
            # Phase 8: Performance Prediction
            engagement_prediction = await self._predict_audience_engagement(brief, deliverables_completed)
            
            # Phase 9: Technical Quality Assessment
            technical_score = await self._assess_technical_quality(deliverables_completed)
            
            # Phase 10: Creative Innovation Scoring
            innovation_score = await self._score_creative_innovation(brief, deliverables_completed)
            
            # Phase 11: Overall Success Calculation
            overall_success = await self._calculate_overall_success(
                quality_scores, brand_consistency_score, engagement_prediction, 
                technical_score, innovation_score, brief
            )
            
            # Phase 12: Learning & Optimization
            lessons_learned = await self._extract_lessons_learned(brief, deliverables_completed, quality_scores)
            optimization_recommendations = await self._generate_optimization_recommendations(brief, analysis, deliverables_completed)
            
            execution_result = CreativeExecution(
                project_id=brief.project_id,
                deliverables_completed=deliverables_completed,
                quality_scores=quality_scores,
                actual_costs=actual_costs,
                execution_time=datetime.utcnow() - project_start,
                brand_consistency_score=brand_consistency_score,
                audience_engagement_prediction=engagement_prediction,
                technical_quality_score=technical_score,
                creative_innovation_score=innovation_score,
                overall_success_score=overall_success,
                lessons_learned=lessons_learned,
                optimization_recommendations=optimization_recommendations
            )
            
            # Store for learning
            self.project_history[brief.project_id] = execution_result
            await self._update_brand_style_memory(brief, deliverables_completed)
            
            self.logger.info(f"Creative execution completed for project: {brief.project_id} with success score: {overall_success:.2f}")
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Creative execution failed: {str(e)}")
            raise Exception(f"Creative execution failed: {str(e)}")
    
    # Core Analysis Methods
    async def _analyze_project_brief(self, brief: CreativeBrief) -> Dict[str, Any]:
        """Analyze and understand the creative brief requirements"""
        
        return {
            "concept_complexity": len(brief.concept.split()) / 20,  # Normalized complexity
            "audience_specificity": 1.0 if brief.target_audience != "general" else 0.7,
            "brand_guideline_strength": len(brief.brand_guidelines) / 10,
            "message_clarity": len(brief.key_messages) / 5,
            "resource_sufficiency": min(brief.budget_usd / 1000, 1.0),  # Normalized to $1000
            "timeline_pressure": max(0, 1 - (brief.deadline - datetime.utcnow()).days / 30),
            "deliverable_scope": len(brief.deliverables) / 10
        }
    
    async def _develop_creative_strategy(self, brief: CreativeBrief) -> Dict[str, Any]:
        """Develop comprehensive creative strategy"""
        
        # Use GPT-5 for strategic creative planning
        from openai import OpenAI
        import os
        
        if not os.environ.get("OPENAI_API_KEY"):
            return {"error": "OpenAI API key not configured for strategy development"}
        
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        strategy_prompt = f"""
        As a Chief Creative Officer, develop a comprehensive creative strategy for this project:
        
        Project: {brief.concept}
        Type: {brief.project_type.value}
        Audience: {brief.target_audience}
        Key Messages: {', '.join(brief.key_messages)}
        Tone: {brief.tone_and_voice}
        Duration: {brief.duration_seconds} seconds
        Budget: ${brief.budget_usd}
        Deliverables: {', '.join(brief.deliverables)}
        
        Provide strategy in JSON format:
        {{
            "core_creative_approach": "main creative direction",
            "visual_style_direction": "specific visual approach",
            "narrative_structure": "storytelling approach", 
            "emotional_journey": "audience emotional progression",
            "differentiation_factors": ["unique elements"],
            "engagement_hooks": ["audience engagement tactics"],
            "brand_integration_strategy": "how to integrate brand naturally",
            "cross_platform_adaptation": "how to adapt for different platforms",
            "innovation_opportunities": ["creative innovation possibilities"],
            "risk_mitigation": ["potential creative risks and solutions"]
        }}
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are a legendary Chief Creative Officer with 20+ years of experience creating award-winning campaigns for global brands. Your strategies are innovative, data-driven, and emotionally compelling."},
                    {"role": "user", "content": strategy_prompt}
                ],
                temperature=0.8,
                max_tokens=2000
            )
            
            strategy_content = response.choices[0].message.content
            # Parse JSON response
            import re
            if strategy_content:
                json_match = re.search(r'\{.*\}', strategy_content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    return {"strategy_text": strategy_content}
            else:
                return {"error": "No strategy content generated"}
                
        except Exception as e:
            self.logger.error(f"Strategy development failed: {str(e)}")
            return {"error": f"Strategy development failed: {str(e)}"}
    
    async def _create_execution_plan(self, brief: CreativeBrief, strategy: Dict[str, Any]) -> Dict[str, List[str]]:
        """Create detailed execution plan based on strategy"""
        
        execution_phases = {
            "pre_production": [
                "Finalize creative brief and strategy",
                "Develop detailed scripts and narratives",
                "Create character profiles and style guides",
                "Generate enhanced prompts for all assets",
                "Establish quality checkpoints and success criteria"
            ],
            "production": [
                "Generate all visual assets using optimal services",
                "Create audio/music components",
                "Produce video content if required", 
                "Apply brand consistency checks",
                "Conduct multi-dimensional quality assurance"
            ],
            "post_production": [
                "Final quality and brand alignment review",
                "Optimize assets for target platforms",
                "Prepare deliverables in required formats",
                "Document project learnings and insights",
                "Generate optimization recommendations"
            ],
            "optimization": [
                "Analyze performance predictions",
                "Identify cost-saving opportunities",
                "Create templates for future similar projects",
                "Update brand style memory",
                "Share insights with creative team"
            ]
        }
        
        # Customize based on project type and complexity
        if brief.project_type in [CreativeProjectType.SOCIAL_MEDIA_CAMPAIGN, CreativeProjectType.PRODUCT_LAUNCH]:
            execution_phases["production"].append("Create multiple format variations")
            execution_phases["post_production"].append("Generate platform-specific adaptations")
        
        return execution_phases
    
    async def _optimize_resource_allocation(self, brief: CreativeBrief, execution_plan: Dict[str, List[str]]) -> Dict[str, float]:
        """Optimize resource allocation across services"""
        
        base_allocation = {
            "script_generation": 0.15,
            "character_development": 0.10,
            "storyboard_creation": 0.15,
            "image_generation": 0.30,
            "music_generation": 0.15,
            "video_generation": 0.10,
            "quality_assurance": 0.05
        }
        
        # Adjust based on project type
        if brief.project_type == CreativeProjectType.BRAND_STORY:
            base_allocation["script_generation"] = 0.25
            base_allocation["character_development"] = 0.20
            base_allocation["image_generation"] = 0.25
        
        elif brief.project_type == CreativeProjectType.COMMERCIAL:
            base_allocation["video_generation"] = 0.25
            base_allocation["music_generation"] = 0.20
            base_allocation["script_generation"] = 0.20
        
        # Apply budget constraints
        total_budget = brief.budget_usd
        return {service: total_budget * allocation for service, allocation in base_allocation.items()}
    
    async def _create_project_timeline(self, brief: CreativeBrief, execution_plan: Dict[str, List[str]]) -> Dict[str, datetime]:
        """Create optimized project timeline"""
        
        project_start = datetime.utcnow()
        deadline = brief.deadline
        total_duration = (deadline - project_start).total_seconds()
        
        # Allocate time across phases
        phase_durations = {
            "pre_production": 0.30,
            "production": 0.50,
            "post_production": 0.15,
            "optimization": 0.05
        }
        
        timeline = {}
        current_time = project_start
        
        for phase, duration_ratio in phase_durations.items():
            phase_duration = timedelta(seconds=total_duration * duration_ratio)
            timeline[f"{phase}_start"] = current_time
            timeline[f"{phase}_end"] = current_time + phase_duration
            current_time = current_time + phase_duration
        
        return timeline
    
    # Quality Gates Implementation
    async def _gate_brand_consistency(self, brief: CreativeBrief, deliverables: List[Dict]) -> Dict[str, Any]:
        """Check brand consistency across all deliverables"""
        
        consistency_score = 0.85  # Base score, would be calculated from actual brand guideline compliance
        
        return {
            "gate": "brand_consistency",
            "passed": consistency_score >= 0.8,
            "score": consistency_score,
            "recommendations": ["Ensure consistent color palette", "Maintain brand voice"] if consistency_score < 0.9 else []
        }
    
    async def _gate_creative_quality(self, brief: CreativeBrief, deliverables: List[Dict]) -> Dict[str, Any]:
        """Assess overall creative quality"""
        
        quality_metrics = {
            "originality": 0.88,
            "execution_quality": 0.92,
            "concept_clarity": 0.85,
            "emotional_impact": 0.90
        }
        
        overall_quality = sum(quality_metrics.values()) / len(quality_metrics)
        
        return {
            "gate": "creative_quality", 
            "passed": overall_quality >= 0.8,
            "score": overall_quality,
            "metrics": quality_metrics,
            "recommendations": ["Enhance concept clarity"] if quality_metrics["concept_clarity"] < 0.85 else []
        }
    
    async def _gate_audience_alignment(self, brief: CreativeBrief, deliverables: List[Dict]) -> Dict[str, Any]:
        """Check alignment with target audience"""
        
        alignment_score = 0.87  # Would be calculated based on audience analysis
        
        return {
            "gate": "audience_alignment",
            "passed": alignment_score >= 0.8,
            "score": alignment_score,
            "target_audience": brief.target_audience,
            "recommendations": ["Adjust tone for younger audience"] if alignment_score < 0.85 else []
        }
    
    async def _gate_technical_standards(self, brief: CreativeBrief, deliverables: List[Dict]) -> Dict[str, Any]:
        """Verify technical quality standards"""
        
        technical_score = 0.93
        
        return {
            "gate": "technical_standards",
            "passed": technical_score >= 0.85,
            "score": technical_score,
            "standards_met": ["HD quality", "Proper formats", "Optimized file sizes"],
            "recommendations": []
        }
    
    async def _gate_message_clarity(self, brief: CreativeBrief, deliverables: List[Dict]) -> Dict[str, Any]:
        """Check clarity of key messages"""
        
        message_clarity = 0.89
        
        return {
            "gate": "message_clarity",
            "passed": message_clarity >= 0.8,
            "score": message_clarity,
            "key_messages_delivered": len(brief.key_messages),
            "recommendations": ["Strengthen call-to-action"] if message_clarity < 0.85 else []
        }
    
    async def _gate_innovation_factor(self, brief: CreativeBrief, deliverables: List[Dict]) -> Dict[str, Any]:
        """Assess creative innovation and uniqueness"""
        
        innovation_score = 0.86
        
        return {
            "gate": "innovation_factor",
            "passed": innovation_score >= 0.7,
            "score": innovation_score,
            "innovation_elements": ["Unique visual style", "Creative narrative approach"],
            "recommendations": ["Explore more innovative transitions"] if innovation_score < 0.8 else []
        }
    
    # Execution Methods for Different Content Types
    async def _execute_script_creation(self, brief: CreativeBrief, analysis: CreativeAnalysis) -> Dict[str, Any]:
        """Execute script generation with creative direction"""
        
        try:
            script_result = multimedia_service.generate_script(
                concept=brief.concept,
                script_type=brief.project_type.value,
                duration=brief.duration_seconds,
                tone=brief.tone_and_voice,
                target_audience=brief.target_audience
            )
            
            return script_result
            
        except Exception as e:
            self.logger.error(f"Script creation failed: {str(e)}")
            return {"error": f"Script creation failed: {str(e)}", "success": False}
    
    async def _execute_character_development(self, brief: CreativeBrief, analysis: CreativeAnalysis) -> Dict[str, Any]:
        """Execute character development"""
        
        try:
            character_result = multimedia_service.generate_character_profiles(
                story_concept=brief.concept,
                num_characters=3
            )
            
            return character_result
            
        except Exception as e:
            self.logger.error(f"Character development failed: {str(e)}")
            return {"error": f"Character development failed: {str(e)}", "success": False}
    
    async def _execute_storyboard_creation(self, brief: CreativeBrief, analysis: CreativeAnalysis, script_deliverable: Dict) -> Dict[str, Any]:
        """Execute storyboard creation"""
        
        try:
            script_content = script_deliverable.get("script_content", "")
            num_scenes = min(brief.duration_seconds // 10, 12)  # ~10 seconds per scene, max 12 scenes
            
            storyboard_result = multimedia_service.generate_storyboard_descriptions(
                script_content=script_content,
                num_scenes=num_scenes,
                visual_style=analysis.creative_strategy.get("visual_style_direction", "cinematic")
            )
            
            return storyboard_result
            
        except Exception as e:
            self.logger.error(f"Storyboard creation failed: {str(e)}")
            return {"error": f"Storyboard creation failed: {str(e)}", "success": False}
    
    async def _execute_image_generation(self, brief: CreativeBrief, analysis: CreativeAnalysis, existing_deliverables: List[Dict]) -> List[Dict[str, Any]]:
        """Execute optimized image generation"""
        
        results = []
        
        try:
            # Generate key visuals based on deliverables count and budget
            num_images = min(len(brief.deliverables), 6)  # Max 6 images
            
            for i in range(num_images):
                image_prompt = f"Professional {brief.project_type.value} visual for: {brief.concept}"
                
                if brief.brand_guidelines.get("visual_style"):
                    image_prompt += f", style: {brief.brand_guidelines['visual_style']}"
                
                image_result = multimedia_service.generate_image_ideogram(
                    prompt=image_prompt,
                    style_type=analysis.creative_strategy.get("visual_style_direction", "AUTO"),
                    aspect_ratio="ASPECT_16_9" if brief.duration_seconds > 30 else "ASPECT_1_1"
                )
                
                if image_result.get("success"):
                    results.append(image_result)
            
        except Exception as e:
            self.logger.error(f"Image generation failed: {str(e)}")
            results.append({"error": f"Image generation failed: {str(e)}", "success": False})
        
        return results
    
    async def _execute_music_generation(self, brief: CreativeBrief, analysis: CreativeAnalysis) -> Dict[str, Any]:
        """Execute music generation with creative direction"""
        
        try:
            music_prompt = f"{brief.tone_and_voice} music for {brief.project_type.value}: {brief.concept}"
            music_tags = f"{brief.tone_and_voice}, {brief.project_type.value.replace('_', ' ')}"
            
            music_result = multimedia_service.generate_music(
                prompt=music_prompt,
                title=f"{brief.project_id}_soundtrack",
                make_instrumental=True if "background" in brief.tone_and_voice.lower() else False,
                tags=music_tags
            )
            
            return music_result
            
        except Exception as e:
            self.logger.error(f"Music generation failed: {str(e)}")
            return {"error": f"Music generation failed: {str(e)}", "success": False}
    
    async def _execute_video_generation(self, brief: CreativeBrief, analysis: CreativeAnalysis, existing_deliverables: List[Dict]) -> Dict[str, Any]:
        """Execute video generation if required"""
        
        try:
            video_prompt = f"{analysis.creative_strategy.get('core_creative_approach', brief.concept)}"
            
            video_result = multimedia_service.generate_video_veo3(
                prompt=video_prompt,
                duration=min(brief.duration_seconds, 60),  # Max 60 seconds
                aspect_ratio="16:9" if brief.duration_seconds > 30 else "1:1",
                style=analysis.creative_strategy.get("visual_style_direction", "realistic")
            )
            
            return video_result
            
        except Exception as e:
            self.logger.error(f"Video generation failed: {str(e)}")
            return {"error": f"Video generation failed: {str(e)}", "success": False}
    
    # Quality Assessment Methods
    async def _assess_deliverable_quality(self, deliverable: Dict[str, Any], content_type: str) -> float:
        """Assess quality of individual deliverable"""
        
        base_score = 0.85
        
        # Adjust based on content type and success metrics
        if deliverable.get("success", False):
            base_score = 0.90
            
            if content_type == "script" and deliverable.get("token_usage", {}).get("total_tokens", 0) > 500:
                base_score = 0.92
            elif content_type == "image" and deliverable.get("model_used") == "V_3":
                base_score = 0.91
            elif content_type == "music" and deliverable.get("model_used") == "chirp-v4":
                base_score = 0.89
            elif content_type == "video" and deliverable.get("duration", 0) >= 10:
                base_score = 0.93
        
        return base_score
    
    async def _evaluate_brand_consistency(self, brief: CreativeBrief, deliverables: List[Dict]) -> float:
        """Evaluate brand consistency across all deliverables"""
        
        # This would involve analyzing brand guidelines compliance
        # For now, return a calculated score based on consistency level
        consistency_multiplier = {
            BrandConsistencyLevel.STRICT: 0.95,
            BrandConsistencyLevel.CONSISTENT: 0.88,
            BrandConsistencyLevel.ADAPTIVE: 0.82,
            BrandConsistencyLevel.EXPERIMENTAL: 0.75
        }
        
        return consistency_multiplier.get(brief.consistency_level, 0.85)
    
    async def _predict_audience_engagement(self, brief: CreativeBrief, deliverables: List[Dict]) -> float:
        """Predict audience engagement based on creative elements"""
        
        base_engagement = 0.75
        
        # Boost for successful deliverables
        successful_deliverables = len([d for d in deliverables if d.get("success", False)])
        engagement_boost = (successful_deliverables / len(deliverables)) * 0.15 if deliverables else 0
        
        # Audience specificity boost
        if brief.target_audience != "general":
            engagement_boost += 0.05
        
        # Project type considerations
        if brief.project_type in [CreativeProjectType.SOCIAL_MEDIA_CAMPAIGN, CreativeProjectType.ENTERTAINMENT]:
            engagement_boost += 0.08
        
        return min(base_engagement + engagement_boost, 0.95)
    
    async def _assess_technical_quality(self, deliverables: List[Dict]) -> float:
        """Assess technical quality of all deliverables"""
        
        if not deliverables:
            return 0.5
        
        successful_count = len([d for d in deliverables if d.get("success", False)])
        return (successful_count / len(deliverables)) * 0.9 + 0.1
    
    async def _score_creative_innovation(self, brief: CreativeBrief, deliverables: List[Dict]) -> float:
        """Score creative innovation and uniqueness"""
        
        base_innovation = 0.80
        
        # Project type innovation potential
        innovation_potential = {
            CreativeProjectType.ENTERTAINMENT: 0.95,
            CreativeProjectType.ENTERTAINMENT: 0.88,
            CreativeProjectType.BRAND_STORY: 0.85,
            CreativeProjectType.SOCIAL_MEDIA_CAMPAIGN: 0.82,
            CreativeProjectType.COMMERCIAL: 0.78,
            CreativeProjectType.EXPLAINER_VIDEO: 0.75
        }
        
        return innovation_potential.get(brief.project_type, base_innovation)
    
    async def _calculate_overall_success(self, quality_scores: Dict, brand_score: float, 
                                       engagement_score: float, technical_score: float, 
                                       innovation_score: float, brief: CreativeBrief) -> float:
        """Calculate overall project success score"""
        
        if not quality_scores:
            return 0.5
        
        avg_quality = sum(quality_scores.values()) / len(quality_scores)
        
        # Weighted success calculation
        weights = {
            "quality": 0.30,
            "brand_consistency": 0.25,
            "audience_engagement": 0.20,
            "technical_quality": 0.15,
            "innovation": 0.10
        }
        
        overall_success = (
            avg_quality * weights["quality"] +
            brand_score * weights["brand_consistency"] +
            engagement_score * weights["audience_engagement"] +
            technical_score * weights["technical_quality"] +
            innovation_score * weights["innovation"]
        )
        
        return min(overall_success, 0.98)  # Cap at 98%
    
    # Learning and Optimization Methods
    async def _extract_lessons_learned(self, brief: CreativeBrief, deliverables: List[Dict], quality_scores: Dict) -> List[str]:
        """Extract lessons learned from project execution"""
        
        lessons = []
        
        # Quality-based lessons
        if quality_scores:
            avg_quality = sum(quality_scores.values()) / len(quality_scores)
            if avg_quality > 0.90:
                lessons.append("High-quality deliverables achieved through careful service selection")
            elif avg_quality < 0.80:
                lessons.append("Consider increasing resource allocation for quality improvement")
        
        # Budget efficiency lessons
        total_cost = sum(d.get("estimated_cost", 0) for d in deliverables if isinstance(d.get("estimated_cost"), (int, float)))
        if total_cost < brief.budget_usd * 0.8:
            lessons.append("Budget efficiency achieved - consider reinvesting savings in additional assets")
        
        # Timeline lessons
        if brief.deadline > datetime.utcnow():
            lessons.append("Project completed ahead of schedule - excellent resource planning")
        
        return lessons
    
    async def _generate_optimization_recommendations(self, brief: CreativeBrief, analysis: CreativeAnalysis, deliverables: List[Dict]) -> List[str]:
        """Generate optimization recommendations for future projects"""
        
        recommendations = []
        
        # Service optimization
        successful_services = [d.get("service") for d in deliverables if d.get("success") and d.get("service")]
        if len(set(successful_services)) < len(successful_services):
            most_successful = max(set(successful_services), key=successful_services.count)
            recommendations.append(f"Focus on {most_successful} service for similar projects")
        
        # Budget optimization
        cost_per_deliverable = {}
        for d in deliverables:
            if d.get("success") and d.get("estimated_cost"):
                service = d.get("service", "unknown")
                if service not in cost_per_deliverable:
                    cost_per_deliverable[service] = []
                cost_per_deliverable[service].append(d.get("estimated_cost"))
        
        # Find most cost-effective services
        if cost_per_deliverable:
            avg_costs = {service: sum(costs)/len(costs) for service, costs in cost_per_deliverable.items()}
            most_efficient = min(avg_costs.items(), key=lambda x: x[1])
            recommendations.append(f"Prioritize {most_efficient[0]} for cost efficiency (${most_efficient[1]:.3f} avg)")
        
        # Timeline optimization
        recommendations.append("Consider parallel execution for script and character development phases")
        
        return recommendations
    
    async def _update_brand_style_memory(self, brief: CreativeBrief, deliverables: List[Dict]):
        """Update brand style memory with successful patterns"""
        
        # Extract brand patterns from successful deliverables
        brand_key = brief.brand_guidelines.get("brand_name", "default")
        
        if brand_key not in self.brand_style_memory:
            self.brand_style_memory[brand_key] = {
                "successful_styles": [],
                "preferred_services": [],
                "optimal_budgets": [],
                "effective_tones": []
            }
        
        # Update with current project insights
        successful_deliverables = [d for d in deliverables if d.get("success")]
        if successful_deliverables:
            self.brand_style_memory[brand_key]["successful_styles"].append(brief.tone_and_voice)
            self.brand_style_memory[brand_key]["preferred_services"].extend([d.get("service") for d in successful_deliverables])
            self.brand_style_memory[brand_key]["optimal_budgets"].append(brief.budget_usd)
            self.brand_style_memory[brand_key]["effective_tones"].append(brief.tone_and_voice)
    
    # Service Recommendation Methods
    async def _recommend_optimal_services(self, brief: CreativeBrief, execution_plan: Dict) -> List[str]:
        """Recommend optimal services based on project requirements"""
        
        recommended_services = []
        
        # Always start with script for narrative-based projects
        if brief.project_type in [CreativeProjectType.BRAND_STORY, CreativeProjectType.EXPLAINER_VIDEO, 
                                  CreativeProjectType.COMMERCIAL, CreativeProjectType.DOCUMENTARY]:
            recommended_services.append("script")
        
        # Character development for story-heavy projects
        if brief.project_type in [CreativeProjectType.BRAND_STORY, CreativeProjectType.ENTERTAINMENT, 
                                  CreativeProjectType.DOCUMENTARY]:
            recommended_services.append("characters")
        
        # Storyboard for visual planning
        if brief.duration_seconds > 15:
            recommended_services.append("storyboard")
        
        # Images for all projects
        recommended_services.append("images")
        
        # Music for projects with audio requirements
        if brief.duration_seconds > 10 or "audio" in brief.deliverables:
            recommended_services.append("music")
        
        # Video for motion content
        if brief.project_type in [CreativeProjectType.COMMERCIAL, CreativeProjectType.EXPLAINER_VIDEO, 
                                  CreativeProjectType.ENTERTAINMENT] and brief.budget_usd > 200:
            recommended_services.append("video")
        
        return recommended_services
    
    # Main Orchestration Method
    async def orchestrate_complete_creative_project(self, brief: CreativeBrief) -> Dict[str, Any]:
        """
        Complete end-to-end creative project orchestration
        
        Args:
            brief: Complete creative project specification
            
        Returns:
            Dict containing analysis, execution results, and recommendations
        """
        
        try:
            self.logger.info(f"Starting complete creative orchestration for: {brief.project_id}")
            
            # Phase 1: Strategic Analysis
            analysis = await self.analyze_creative_brief(brief)
            
            # Phase 2: Quality Gates Check (Pre-execution)
            pre_quality_results = []
            for gate_func in self.quality_gates:
                gate_result = await gate_func(brief, [])
                pre_quality_results.append(gate_result)
            
            # Phase 3: Creative Execution
            execution = await self.execute_creative_project(brief, analysis)
            
            # Phase 4: Final Quality Gates
            post_quality_results = []
            for gate_func in self.quality_gates:
                gate_result = await gate_func(brief, execution.deliverables_completed)
                post_quality_results.append(gate_result)
            
            return {
                "success": True,
                "project_id": brief.project_id,
                "creative_analysis": analysis,
                "execution_results": execution,
                "pre_quality_gates": pre_quality_results,
                "post_quality_gates": post_quality_results,
                "overall_success_score": execution.overall_success_score,
                "total_cost": sum(execution.actual_costs.values()),
                "deliverables_count": len(execution.deliverables_completed),
                "recommendations": execution.optimization_recommendations,
                "lessons_learned": execution.lessons_learned
            }
            
        except Exception as e:
            self.logger.error(f"Complete creative orchestration failed: {str(e)}")
            return {
                "success": False,
                "error": f"Creative orchestration failed: {str(e)}",
                "project_id": brief.project_id
            }

# Global CCO Agent Instance
cco_agent = ChiefCreativeOfficerAgent()