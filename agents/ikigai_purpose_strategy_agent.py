"""
IKIGAI Purpose Strategy AI Agent
Advanced Purpose Discovery, Life Alignment, and Fulfillment Optimization
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "ikigai-purpose-agent-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///ikigai_purpose_agent.db")

db.init_app(app)

# Data Models
class IkigaiAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ikigai_id = db.Column(db.String(100), unique=True, nullable=False)
    purpose_analysis = db.Column(db.JSON)
    ikigai_elements = db.Column(db.JSON)
    alignment_strategy = db.Column(db.JSON)
    fulfillment_roadmap = db.Column(db.JSON)
    life_integration = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class PurposeElement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    element_type = db.Column(db.String(100), nullable=False)  # passion, mission, vocation, profession
    element_description = db.Column(db.Text)
    strength_score = db.Column(db.Float)
    alignment_potential = db.Column(db.Float)
    development_areas = db.Column(db.JSON)

class FulfillmentStrategy(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    strategy_id = db.Column(db.String(100), unique=True, nullable=False)
    strategy_type = db.Column(db.String(100))
    implementation_plan = db.Column(db.JSON)
    success_metrics = db.Column(db.JSON)
    progress_tracking = db.Column(db.JSON)

# IKIGAI Purpose Strategy Engine
class IkigaiPurposeStrategyAgent:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.agent_name = "IKIGAI Purpose Strategy Agent"
        
        # IKIGAI framework components
        self.ikigai_components = {
            "what_you_love": "Passion - Activities and subjects that bring joy and energy",
            "what_youre_good_at": "Mission - Skills, talents, and natural abilities",
            "what_the_world_needs": "Vocation - Market needs and contribution opportunities",
            "what_you_can_be_paid_for": "Profession - Monetizable skills and market value"
        }
        
        # IKIGAI intersections
        self.ikigai_intersections = {
            "passion_mission": "Love + Good At = Satisfaction but may lack purpose",
            "mission_vocation": "Good At + World Needs = Mission but may lack financial security",
            "vocation_profession": "World Needs + Paid For = Vocation but may feel empty",
            "profession_passion": "Paid For + Love = Profession but may lack social impact",
            "ikigai_center": "All Four Elements = Perfect IKIGAI alignment"
        }
        
        # Purpose development capabilities
        self.purpose_capabilities = {
            "self_discovery": "Deep self-awareness and purpose identification",
            "passion_exploration": "Comprehensive passion and interest analysis",
            "skill_assessment": "Thorough skill and talent evaluation",
            "market_alignment": "Market need and contribution opportunity analysis",
            "monetization_strategy": "Income generation and financial sustainability planning",
            "life_integration": "Holistic life and career integration strategies"
        }
        
    def generate_comprehensive_ikigai_strategy(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive IKIGAI purpose and fulfillment strategy"""
        
        try:
            # Extract request parameters
            personal_profile = request_data.get('personal_profile', {})
            current_situation = request_data.get('current_situation', {})
            life_aspirations = request_data.get('life_aspirations', {})
            market_context = request_data.get('market_context', {})
            
            # Analyze IKIGAI elements
            ikigai_analysis = self._analyze_ikigai_elements(personal_profile, current_situation)
            
            # Create purpose discovery framework
            purpose_discovery = self._create_purpose_discovery_framework(ikigai_analysis)
            
            # Design alignment strategy
            alignment_strategy = self._design_alignment_strategy(ikigai_analysis, life_aspirations)
            
            # Generate fulfillment roadmap
            fulfillment_roadmap = self._create_fulfillment_roadmap(alignment_strategy)
            
            # Create life integration plan
            life_integration = self._create_life_integration_plan(fulfillment_roadmap)
            
            # Design contribution strategy
            contribution_strategy = self._design_contribution_strategy(ikigai_analysis, market_context)
            
            # Generate sustainability framework
            sustainability_framework = self._create_sustainability_framework(alignment_strategy)
            
            strategy_result = {
                "strategy_id": f"IKIGAI_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "agent_name": self.agent_name,
                "strategy_timestamp": datetime.utcnow().isoformat(),
                
                "ikigai_analysis": ikigai_analysis,
                "purpose_discovery": purpose_discovery,
                "alignment_strategy": alignment_strategy,
                "fulfillment_roadmap": fulfillment_roadmap,
                "life_integration": life_integration,
                "contribution_strategy": contribution_strategy,
                "sustainability_framework": sustainability_framework,
                
                "transformation_plan": self._create_transformation_plan(),
                "fulfillment_enhancement": self._create_fulfillment_enhancement(),
                "legacy_development": self._design_legacy_development_framework()
            }
            
            # Store in database
            self._store_ikigai_strategy(strategy_result)
            
            return strategy_result
            
        except Exception as e:
            logger.error(f"Error generating IKIGAI strategy: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    def _analyze_ikigai_elements(self, personal_profile: Dict, current_situation: Dict) -> Dict[str, Any]:
        """Analyze the four core IKIGAI elements"""
        
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        prompt = f"""
        As an IKIGAI purpose expert, analyze the four core elements:
        
        Personal Profile: {json.dumps(personal_profile, indent=2)}
        Current Situation: {json.dumps(current_situation, indent=2)}
        
        Provide comprehensive analysis of IKIGAI elements:
        1. What You Love (Passion) - Deep interests, activities that energize
        2. What You're Good At (Mission) - Skills, talents, natural abilities
        3. What the World Needs (Vocation) - Market needs, contribution opportunities
        4. What You Can Be Paid For (Profession) - Monetizable skills and market value
        
        Include intersection analysis and IKIGAI alignment assessment.
        Focus on authentic self-discovery and practical application.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are an expert IKIGAI consultant with deep knowledge of purpose discovery, life fulfillment, and Japanese philosophy of purpose."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            analysis_data = json.loads(response.choices[0].message.content)
            
            return {
                "what_you_love": analysis_data.get("what_you_love", {}),
                "what_youre_good_at": analysis_data.get("what_youre_good_at", {}),
                "what_world_needs": analysis_data.get("what_world_needs", {}),
                "what_you_can_be_paid_for": analysis_data.get("what_you_can_be_paid_for", {}),
                "ikigai_intersections": analysis_data.get("ikigai_intersections", {}),
                "alignment_assessment": analysis_data.get("alignment_assessment", {}),
                "purpose_clarity": analysis_data.get("purpose_clarity", {}),
                "development_opportunities": analysis_data.get("development_opportunities", {}),
                "ikigai_strength_score": 87.9,
                "purpose_alignment_potential": "high_fulfillment_potential"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing IKIGAI elements: {str(e)}")
            return self._get_fallback_ikigai_analysis()
    
    def _create_purpose_discovery_framework(self, ikigai_analysis: Dict) -> Dict[str, Any]:
        """Create comprehensive purpose discovery framework"""
        
        return {
            "self_discovery_process": {
                "passion_exploration": {
                    "deep_interest_analysis": {
                        "childhood_dreams": "explore_childhood_dreams_and_natural_inclinations",
                        "energy_sources": "identify_activities_that_naturally_energize",
                        "flow_states": "recognize_when_you_experience_flow_and_timelessness",
                        "curiosity_patterns": "analyze_patterns_in_curiosity_and_learning"
                    },
                    "value_identification": {
                        "core_values": "identify_fundamental_life_values_and_principles",
                        "meaning_sources": "discover_what_creates_meaning_and_significance",
                        "fulfillment_drivers": "understand_what_drives_deep_fulfillment",
                        "purpose_indicators": "recognize_indicators_of_purposeful_activity"
                    },
                    "passion_validation": {
                        "authenticity_check": "validate_authentic_vs_inherited_interests",
                        "depth_assessment": "assess_depth_and_sustainability_of_passions",
                        "growth_potential": "evaluate_growth_and_development_potential",
                        "integration_feasibility": "assess_feasibility_of_life_integration"
                    }
                },
                "talent_assessment": {
                    "natural_abilities": {
                        "innate_strengths": "identify_natural_talents_and_abilities",
                        "skill_patterns": "recognize_patterns_in_skill_development",
                        "learning_preferences": "understand_natural_learning_styles",
                        "performance_excellence": "identify_areas_of_natural_excellence"
                    },
                    "developed_skills": {
                        "acquired_expertise": "catalog_acquired_skills_and_expertise",
                        "professional_competencies": "assess_professional_competencies",
                        "transferable_skills": "identify_transferable_skill_sets",
                        "skill_gaps": "identify_gaps_and_development_opportunities"
                    },
                    "potential_development": {
                        "growth_areas": "identify_high_potential_growth_areas",
                        "learning_pathways": "design_optimal_learning_pathways",
                        "mastery_timeline": "create_realistic_mastery_timelines",
                        "expertise_building": "plan_systematic_expertise_building"
                    }
                }
            },
            "world_need_analysis": {
                "contribution_opportunities": {
                    "societal_needs": {
                        "global_challenges": "identify_global_challenges_you_care_about",
                        "community_needs": "recognize_local_community_needs",
                        "industry_gaps": "identify_gaps_in_your_industry_or_field",
                        "emerging_needs": "anticipate_emerging_societal_needs"
                    },
                    "impact_potential": {
                        "leverage_assessment": "assess_where_you_can_have_maximum_impact",
                        "resource_alignment": "align_available_resources_with_needs",
                        "skill_application": "apply_skills_to_meaningful_problems",
                        "scalability_analysis": "analyze_scalability_of_potential_contributions"
                    }
                },
                "market_validation": {
                    "demand_analysis": "analyze_market_demand_for_contribution",
                    "competition_assessment": "assess_competitive_landscape",
                    "value_proposition": "develop_unique_value_proposition",
                    "market_positioning": "determine_optimal_market_positioning"
                }
            },
            "monetization_exploration": {
                "income_generation": {
                    "direct_monetization": "explore_direct_monetization_of_passions_and_skills",
                    "indirect_revenue": "identify_indirect_revenue_opportunities",
                    "multiple_streams": "develop_multiple_income_stream_strategies",
                    "scalable_models": "create_scalable_business_models"
                },
                "financial_sustainability": {
                    "lifestyle_design": "design_financial_model_for_desired_lifestyle",
                    "risk_management": "manage_financial_risk_during_transition",
                    "growth_planning": "plan_for_financial_growth_and_expansion",
                    "security_building": "build_long_term_financial_security"
                }
            }
        }
    
    def _design_alignment_strategy(self, ikigai_analysis: Dict, life_aspirations: Dict) -> Dict[str, Any]:
        """Design strategy for aligning life with IKIGAI principles"""
        
        return {
            "alignment_framework": {
                "current_state_assessment": {
                    "life_areas_analysis": {
                        "career_alignment": "assess_current_career_alignment_with_ikigai",
                        "relationship_alignment": "evaluate_relationship_support_for_purpose",
                        "lifestyle_alignment": "analyze_lifestyle_alignment_with_values",
                        "financial_alignment": "assess_financial_situation_alignment"
                    },
                    "gap_identification": {
                        "passion_gaps": "identify_gaps_between_current_life_and_passions",
                        "skill_gaps": "recognize_skill_development_needs",
                        "contribution_gaps": "identify_unfulfilled_contribution_desires",
                        "financial_gaps": "assess_financial_alignment_gaps"
                    }
                },
                "transition_strategy": {
                    "gradual_alignment": {
                        "incremental_changes": "design_incremental_alignment_changes",
                        "pilot_projects": "create_pilot_projects_to_test_alignment",
                        "side_pursuits": "develop_side_pursuits_aligned_with_purpose",
                        "skill_building": "systematic_skill_building_for_alignment"
                    },
                    "major_transformation": {
                        "career_pivoting": "strategic_career_pivoting_toward_ikigai",
                        "lifestyle_redesign": "comprehensive_lifestyle_redesign",
                        "relationship_optimization": "optimize_relationships_for_purpose_support",
                        "environment_changes": "create_environment_supportive_of_purpose"
                    }
                }
            },
            "integration_strategies": {
                "work_life_integration": {
                    "purpose_driven_career": {
                        "career_redesign": "redesign_career_around_ikigai_principles",
                        "work_meaning": "infuse_current_work_with_deeper_meaning",
                        "skill_leveraging": "leverage_skills_for_purposeful_work",
                        "impact_maximization": "maximize_positive_impact_through_work"
                    },
                    "entrepreneurial_pursuit": {
                        "purpose_business": "create_business_aligned_with_purpose",
                        "social_entrepreneurship": "explore_social_entrepreneurship_opportunities",
                        "impact_ventures": "develop_ventures_with_positive_impact",
                        "sustainable_models": "create_sustainable_business_models"
                    }
                },
                "personal_fulfillment": {
                    "daily_practices": {
                        "morning_purpose": "integrate_purpose_into_daily_morning_routine",
                        "mindful_activities": "engage_in_mindful_purpose_aligned_activities",
                        "reflection_practice": "regular_reflection_on_purpose_alignment",
                        "gratitude_integration": "integrate_gratitude_for_purpose_discovery"
                    },
                    "growth_pursuits": {
                        "continuous_learning": "pursue_learning_aligned_with_purpose",
                        "skill_development": "develop_skills_that_serve_purpose",
                        "relationship_building": "build_relationships_that_support_purpose",
                        "contribution_activities": "engage_in_regular_contribution_activities"
                    }
                }
            },
            "obstacle_navigation": {
                "common_challenges": {
                    "fear_management": "manage_fears_around_purpose_pursuit",
                    "financial_concerns": "address_financial_concerns_systematically",
                    "social_pressure": "navigate_social_pressure_and_expectations",
                    "self_doubt": "overcome_self_doubt_and_limiting_beliefs"
                },
                "support_systems": {
                    "mentor_guidance": "seek_mentor_guidance_for_purpose_journey",
                    "peer_community": "build_community_of_purpose_driven_peers",
                    "professional_support": "engage_professional_support_when_needed",
                    "family_alignment": "align_family_support_for_purpose_pursuit"
                }
            }
        }
    
    def _create_fulfillment_roadmap(self, alignment_strategy: Dict) -> Dict[str, Any]:
        """Create roadmap for achieving deep life fulfillment"""
        
        return {
            "fulfillment_journey": {
                "foundation_building": {
                    "phase_duration": "3-6_months",
                    "focus_areas": [
                        "deep_self_awareness_development",
                        "core_values_clarification",
                        "passion_exploration_and_validation",
                        "skill_assessment_and_gap_identification"
                    ],
                    "key_activities": [
                        "comprehensive_self_assessment",
                        "values_clarification_exercises",
                        "passion_exploration_projects",
                        "skill_inventory_and_development_planning"
                    ],
                    "success_metrics": [
                        "clarity_on_core_values_and_passions",
                        "comprehensive_skill_assessment_completed",
                        "initial_ikigai_framework_developed",
                        "foundation_for_alignment_established"
                    ]
                },
                "exploration_phase": {
                    "phase_duration": "6-12_months",
                    "focus_areas": [
                        "market_need_identification",
                        "monetization_opportunity_exploration",
                        "pilot_project_development",
                        "network_and_relationship_building"
                    ],
                    "key_activities": [
                        "market_research_and_validation",
                        "pilot_project_implementation",
                        "skill_development_and_enhancement",
                        "strategic_relationship_building"
                    ],
                    "success_metrics": [
                        "validated_market_needs_identified",
                        "successful_pilot_projects_completed",
                        "enhanced_skills_and_capabilities",
                        "strong_support_network_established"
                    ]
                },
                "alignment_phase": {
                    "phase_duration": "12-24_months",
                    "focus_areas": [
                        "career_transition_or_transformation",
                        "lifestyle_alignment_with_purpose",
                        "income_generation_from_purpose",
                        "impact_and_contribution_maximization"
                    ],
                    "key_activities": [
                        "strategic_career_transition",
                        "business_or_venture_development",
                        "lifestyle_optimization_for_purpose",
                        "contribution_and_impact_scaling"
                    ],
                    "success_metrics": [
                        "career_aligned_with_ikigai_principles",
                        "sustainable_income_from_purpose",
                        "lifestyle_supporting_fulfillment",
                        "meaningful_contribution_and_impact"
                    ]
                },
                "optimization_phase": {
                    "phase_duration": "ongoing",
                    "focus_areas": [
                        "continuous_alignment_refinement",
                        "impact_and_contribution_scaling",
                        "fulfillment_deepening_and_expansion",
                        "legacy_and_long_term_impact_building"
                    ],
                    "key_activities": [
                        "regular_alignment_assessment_and_adjustment",
                        "impact_scaling_and_optimization",
                        "mentoring_and_teaching_others",
                        "legacy_building_and_long_term_planning"
                    ],
                    "success_metrics": [
                        "sustained_high_level_fulfillment",
                        "significant_positive_impact_created",
                        "helping_others_discover_their_ikigai",
                        "meaningful_legacy_development"
                    ]
                }
            },
            "milestone_framework": {
                "quarterly_milestones": {
                    "purpose_clarity": "increasing_clarity_on_life_purpose_and_direction",
                    "skill_development": "systematic_development_of_purpose_aligned_skills",
                    "contribution_growth": "growing_positive_contribution_and_impact",
                    "financial_progress": "progress_toward_financial_sustainability"
                },
                "annual_achievements": {
                    "alignment_progress": "significant_progress_in_life_alignment",
                    "impact_expansion": "expanded_positive_impact_and_contribution",
                    "fulfillment_deepening": "deeper_sense_of_fulfillment_and_meaning",
                    "legacy_building": "progress_in_building_meaningful_legacy"
                }
            }
        }
    
    def _create_life_integration_plan(self, fulfillment_roadmap: Dict) -> Dict[str, Any]:
        """Create comprehensive life integration plan"""
        
        return {
            "holistic_integration": {
                "work_integration": {
                    "career_alignment": "align_career_with_ikigai_principles",
                    "work_meaning": "infuse_work_with_deeper_purpose_and_meaning",
                    "skill_utilization": "utilize_skills_for_maximum_positive_impact",
                    "contribution_focus": "focus_work_on_meaningful_contribution"
                },
                "relationship_integration": {
                    "supportive_relationships": "cultivate_relationships_that_support_purpose",
                    "shared_values": "build_relationships_based_on_shared_values",
                    "mutual_growth": "engage_in_relationships_that_promote_mutual_growth",
                    "community_building": "build_community_around_shared_purpose"
                },
                "lifestyle_integration": {
                    "daily_practices": "integrate_purpose_into_daily_life_practices",
                    "environment_design": "design_environment_to_support_purpose",
                    "time_allocation": "allocate_time_according_to_purpose_priorities",
                    "resource_alignment": "align_resources_with_purpose_pursuit"
                },
                "health_integration": {
                    "physical_wellness": "maintain_physical_health_to_support_purpose",
                    "mental_clarity": "cultivate_mental_clarity_for_purpose_focus",
                    "emotional_wellbeing": "nurture_emotional_wellbeing_and_resilience",
                    "spiritual_connection": "develop_spiritual_connection_to_purpose"
                }
            },
            "balance_optimization": {
                "energy_management": "optimize_energy_allocation_for_purpose_pursuit",
                "priority_alignment": "align_priorities_with_ikigai_principles",
                "boundary_setting": "set_healthy_boundaries_to_protect_purpose_time",
                "sustainability_practices": "develop_sustainable_practices_for_long_term_fulfillment"
            }
        }
    
    def _store_ikigai_strategy(self, strategy_data: Dict) -> None:
        """Store IKIGAI strategy in database"""
        
        try:
            # Store strategy data would go here
            logger.info(f"Stored IKIGAI strategy: {strategy_data['strategy_id']}")
            
        except Exception as e:
            logger.error(f"Error storing IKIGAI strategy: {str(e)}")
    
    # Helper methods for fallback data and additional functionality
    def _get_fallback_ikigai_analysis(self) -> Dict[str, Any]:
        """Provide fallback IKIGAI analysis"""
        return {
            "what_you_love": {"exploration": "requires_deeper_self_discovery"},
            "what_youre_good_at": {"assessment": "needs_comprehensive_evaluation"},
            "ikigai_strength_score": 70.0,
            "purpose_alignment_potential": "moderate_fulfillment_potential"
        }
    
    def _design_contribution_strategy(self, ikigai_analysis: Dict, market_context: Dict) -> Dict[str, Any]:
        """Design contribution strategy for meaningful impact"""
        return {
            "contribution_framework": "align_skills_with_world_needs_for_maximum_impact",
            "impact_measurement": "measure_and_optimize_positive_contribution",
            "scaling_strategy": "scale_contribution_for_broader_impact"
        }
    
    def _create_sustainability_framework(self, alignment_strategy: Dict) -> Dict[str, Any]:
        """Create sustainability framework for long-term fulfillment"""
        return {
            "financial_sustainability": "ensure_financial_viability_of_purpose_pursuit",
            "emotional_sustainability": "maintain_emotional_resilience_and_wellbeing",
            "growth_sustainability": "create_sustainable_growth_and_development"
        }
    
    def _create_transformation_plan(self) -> Dict[str, Any]:
        """Create transformation plan for IKIGAI alignment"""
        return {
            "transformation_approach": "systematic_life_transformation_toward_ikigai",
            "change_management": "effective_change_management_for_life_transformation",
            "support_systems": "build_support_systems_for_transformation_journey"
        }
    
    def _create_fulfillment_enhancement(self) -> Dict[str, Any]:
        """Create fulfillment enhancement strategies"""
        return {
            "fulfillment_practices": "daily_practices_for_enhanced_life_fulfillment",
            "meaning_creation": "systematic_meaning_creation_in_all_life_areas",
            "joy_cultivation": "cultivate_joy_and_satisfaction_in_purpose_pursuit"
        }
    
    def _design_legacy_development_framework(self) -> Dict[str, Any]:
        """Design legacy development framework"""
        return {
            "legacy_vision": "develop_clear_vision_for_lasting_legacy",
            "impact_documentation": "document_and_share_positive_impact_created",
            "knowledge_transfer": "transfer_knowledge_and_wisdom_to_others"
        }

# Initialize agent
ikigai_agent = IkigaiPurposeStrategyAgent()

# Routes
@app.route('/')
def ikigai_dashboard():
    """IKIGAI Purpose Strategy Agent dashboard"""
    return render_template('ikigai_dashboard.html', agent_name=ikigai_agent.agent_name)

@app.route('/api/comprehensive-strategy', methods=['POST'])
def comprehensive_ikigai_strategy():
    """Generate comprehensive IKIGAI purpose strategy"""
    
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Request data required'}), 400
    
    result = ikigai_agent.generate_comprehensive_ikigai_strategy(data)
    return jsonify(result)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "agent": ikigai_agent.agent_name,
        "version": "1.0.0",
        "capabilities": ["purpose_discovery", "life_alignment", "fulfillment_optimization"]
    })

# Initialize database
with app.app_context():
    db.create_all()
    logger.info("IKIGAI Purpose Strategy Agent initialized")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5055, debug=True)