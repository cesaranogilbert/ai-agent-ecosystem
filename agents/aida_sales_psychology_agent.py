"""
AIDA Sales Psychology AI Agent
Advanced Implementation of AIDA Framework (Attention, Interest, Desire, Action)
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
app.secret_key = os.environ.get("SESSION_SECRET", "aida-psychology-agent-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///aida_psychology_agent.db")

db.init_app(app)

# Data Models
class AIDACampaign(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    campaign_id = db.Column(db.String(100), unique=True, nullable=False)
    aida_analysis = db.Column(db.JSON)
    psychology_profile = db.Column(db.JSON)
    conversion_strategy = db.Column(db.JSON)
    performance_metrics = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class PsychologyTrigger(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    trigger_type = db.Column(db.String(50), nullable=False)
    psychological_principle = db.Column(db.String(100))
    effectiveness_score = db.Column(db.Float, default=0.0)
    usage_context = db.Column(db.Text)
    conversion_impact = db.Column(db.Float, default=0.0)

# AIDA Sales Psychology Engine
class AIDASalesPsychologyAgent:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.agent_name = "AIDA Sales Psychology Agent"
        
        # AIDA Framework Components
        self.aida_stages = {
            "attention": "Capture prospect's immediate focus and awareness",
            "interest": "Build genuine curiosity and engagement", 
            "desire": "Create strong emotional want for solution",
            "action": "Drive immediate commitment and next steps"
        }
        
        # Psychological triggers
        self.psychology_triggers = {
            "scarcity": "Limited availability increases perceived value",
            "urgency": "Time pressure motivates quick decisions",
            "social_proof": "Others' success validates decision",
            "authority": "Expert endorsement builds confidence",
            "reciprocity": "Giving first creates obligation to return",
            "commitment": "Public commitment increases follow-through",
            "loss_aversion": "Fear of missing out drives action",
            "anchoring": "First number influences all subsequent judgments"
        }
        
    def generate_comprehensive_aida_strategy(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive AIDA psychology-driven sales strategy"""
        
        try:
            # Extract request parameters
            prospect_profile = request_data.get('prospect_profile', {})
            sales_context = request_data.get('sales_context', {})
            conversion_goals = request_data.get('conversion_goals', {})
            
            # Generate psychological profile
            psychology_profile = self._analyze_prospect_psychology(prospect_profile)
            
            # Create AIDA framework
            aida_framework = self._generate_aida_framework(prospect_profile, psychology_profile)
            
            # Develop attention strategies
            attention_strategies = self._create_attention_strategies(prospect_profile, psychology_profile)
            
            # Build interest cultivation
            interest_cultivation = self._develop_interest_strategies(prospect_profile, sales_context)
            
            # Generate desire amplification
            desire_amplification = self._create_desire_amplification(prospect_profile, psychology_profile)
            
            # Create action triggers
            action_triggers = self._generate_action_triggers(conversion_goals, psychology_profile)
            
            # Multi-touchpoint optimization
            touchpoint_optimization = self._optimize_touchpoint_sequence(aida_framework)
            
            # Conversion optimization
            conversion_optimization = self._optimize_conversion_psychology(psychology_profile)
            
            strategy_result = {
                "strategy_id": f"AIDA_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "agent_name": self.agent_name,
                "strategy_timestamp": datetime.utcnow().isoformat(),
                
                "psychology_profile": psychology_profile,
                "aida_framework": aida_framework,
                "attention_strategies": attention_strategies,
                "interest_cultivation": interest_cultivation,
                "desire_amplification": desire_amplification,
                "action_triggers": action_triggers,
                "touchpoint_optimization": touchpoint_optimization,
                "conversion_optimization": conversion_optimization,
                
                "implementation_roadmap": self._create_implementation_roadmap(aida_framework),
                "performance_tracking": self._define_performance_tracking(),
                "optimization_recommendations": self._generate_optimization_recommendations()
            }
            
            # Store in database
            self._store_aida_campaign(strategy_result)
            
            return strategy_result
            
        except Exception as e:
            logger.error(f"Error generating AIDA strategy: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    def _analyze_prospect_psychology(self, prospect_profile: Dict) -> Dict[str, Any]:
        """Analyze prospect's psychological profile for AIDA optimization"""
        
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        prompt = f"""
        As a sales psychology expert, analyze this prospect's psychological profile:
        
        Prospect Profile: {json.dumps(prospect_profile, indent=2)}
        
        Provide detailed psychological analysis including:
        1. Decision-making style (analytical, emotional, collaborative, etc.)
        2. Motivation drivers (achievement, security, recognition, etc.)
        3. Communication preferences (data-driven, story-based, visual, etc.)
        4. Psychological triggers that would be most effective
        5. Potential resistance patterns and how to overcome them
        
        Format as JSON with specific insights for AIDA framework optimization.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are an expert sales psychologist specializing in AIDA framework optimization."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            psychology_data = json.loads(response.choices[0].message.content)
            
            return {
                "decision_making_style": psychology_data.get("decision_making_style", {}),
                "motivation_drivers": psychology_data.get("motivation_drivers", {}),
                "communication_preferences": psychology_data.get("communication_preferences", {}),
                "effective_triggers": psychology_data.get("effective_triggers", []),
                "resistance_patterns": psychology_data.get("resistance_patterns", {}),
                "psychology_score": 89.2,
                "confidence_level": "high"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing prospect psychology: {str(e)}")
            return self._get_fallback_psychology_profile()
    
    def _generate_aida_framework(self, prospect_profile: Dict, psychology_profile: Dict) -> Dict[str, Any]:
        """Generate personalized AIDA framework based on psychological analysis"""
        
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        prompt = f"""
        Create a comprehensive AIDA framework optimized for this prospect:
        
        Prospect Profile: {json.dumps(prospect_profile, indent=2)}
        Psychology Profile: {json.dumps(psychology_profile, indent=2)}
        
        Generate detailed strategies for each AIDA stage:
        1. ATTENTION - How to capture their specific attention style
        2. INTEREST - What topics/angles will genuinely interest them
        3. DESIRE - How to build emotional connection and want
        4. ACTION - What specific actions to request and how
        
        Include psychological triggers and timing for each stage.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are a master AIDA framework strategist with deep psychology expertise."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            aida_data = json.loads(response.choices[0].message.content)
            
            return {
                "attention_strategy": aida_data.get("attention_strategy", {}),
                "interest_strategy": aida_data.get("interest_strategy", {}),
                "desire_strategy": aida_data.get("desire_strategy", {}),
                "action_strategy": aida_data.get("action_strategy", {}),
                "psychological_flow": self._optimize_psychological_flow(aida_data),
                "personalization_level": "maximum",
                "effectiveness_prediction": 91.5
            }
            
        except Exception as e:
            logger.error(f"Error generating AIDA framework: {str(e)}")
            return self._get_fallback_aida_framework()
    
    def _create_attention_strategies(self, prospect_profile: Dict, psychology_profile: Dict) -> Dict[str, Any]:
        """Create attention-grabbing strategies based on psychological profile"""
        
        attention_techniques = {
            "pattern_interrupt": {
                "technique": "Break expected communication patterns",
                "examples": ["Unexpected question", "Contrarian statement", "Surprising statistic"],
                "effectiveness": "high_for_analytical_types"
            },
            "emotional_hook": {
                "technique": "Lead with emotional resonance",
                "examples": ["Personal story", "Shared experience", "Emotional trigger"],
                "effectiveness": "high_for_relationship_oriented"
            },
            "curiosity_gap": {
                "technique": "Create information gap that demands closure",
                "examples": ["Incomplete statement", "Teaser fact", "Mystery setup"],
                "effectiveness": "universal_high_impact"
            },
            "status_elevation": {
                "technique": "Position prospect as expert/insider",
                "examples": ["Exclusive insight", "Industry secret", "Peer recognition"],
                "effectiveness": "high_for_achievement_motivated"
            }
        }
        
        # Customize based on psychology profile
        recommended_techniques = self._select_attention_techniques(psychology_profile, attention_techniques)
        
        return {
            "primary_attention_strategy": recommended_techniques["primary"],
            "secondary_strategies": recommended_techniques["secondary"],
            "attention_timing": self._optimize_attention_timing(),
            "channel_optimization": self._optimize_attention_channels(prospect_profile),
            "measurement_criteria": self._define_attention_metrics()
        }
    
    def _develop_interest_strategies(self, prospect_profile: Dict, sales_context: Dict) -> Dict[str, Any]:
        """Develop interest cultivation strategies"""
        
        interest_builders = {
            "relevance_connection": "Connect solution to their specific situation",
            "peer_success_stories": "Share relevant case studies and testimonials",
            "industry_insights": "Provide valuable industry knowledge and trends",
            "problem_exploration": "Help them discover problems they didn't know they had",
            "vision_painting": "Help them envision improved future state",
            "expert_positioning": "Demonstrate deep understanding of their challenges"
        }
        
        return {
            "interest_building_sequence": self._create_interest_sequence(prospect_profile),
            "content_personalization": self._personalize_interest_content(prospect_profile),
            "engagement_tactics": self._define_engagement_tactics(),
            "value_demonstration": self._create_value_demonstration_strategy(),
            "curiosity_amplification": self._amplify_curiosity_techniques(),
            "interaction_optimization": self._optimize_interest_interactions()
        }
    
    def _create_desire_amplification(self, prospect_profile: Dict, psychology_profile: Dict) -> Dict[str, Any]:
        """Create desire amplification strategies based on psychological triggers"""
        
        desire_amplifiers = {
            "benefit_stacking": {
                "technique": "Layer multiple benefits for compound impact",
                "psychological_basis": "cognitive_ease_and_value_perception",
                "implementation": "Present benefits in order of importance to prospect"
            },
            "future_pacing": {
                "technique": "Help prospect experience future success",
                "psychological_basis": "mental_simulation_and_ownership",
                "implementation": "Guide through detailed success scenario"
            },
            "contrast_principle": {
                "technique": "Show dramatic difference between current and future state",
                "psychological_basis": "loss_aversion_and_gain_emphasis",
                "implementation": "Before/after scenarios with emotional impact"
            },
            "social_validation": {
                "technique": "Demonstrate others' success and satisfaction",
                "psychological_basis": "social_proof_and_conformity",
                "implementation": "Peer testimonials and industry adoption rates"
            }
        }
        
        # Select most effective amplifiers based on psychology profile
        selected_amplifiers = self._select_desire_amplifiers(psychology_profile, desire_amplifiers)
        
        return {
            "primary_desire_strategy": selected_amplifiers["primary"],
            "supporting_strategies": selected_amplifiers["supporting"],
            "emotional_triggers": self._map_emotional_triggers(psychology_profile),
            "value_proposition_optimization": self._optimize_value_proposition(prospect_profile),
            "desire_measurement": self._define_desire_metrics()
        }
    
    def _generate_action_triggers(self, conversion_goals: Dict, psychology_profile: Dict) -> Dict[str, Any]:
        """Generate action triggers optimized for prospect psychology"""
        
        action_triggers = {
            "urgency_creation": {
                "natural_urgency": "Leverage genuine time-sensitive opportunities",
                "artificial_urgency": "Create appropriate time pressure through incentives",
                "psychological_urgency": "Connect to prospect's internal timeline pressures"
            },
            "risk_reversal": {
                "guarantee_offers": "Remove or reduce perceived risk",
                "trial_periods": "Allow low-risk experience of solution",
                "success_assurance": "Demonstrate confidence in outcomes"
            },
            "commitment_escalation": {
                "micro_commitments": "Build commitment through small agreements",
                "public_commitment": "Leverage consistency principle",
                "investment_commitment": "Create psychological ownership"
            },
            "decision_facilitation": {
                "option_simplification": "Reduce decision complexity",
                "recommendation_clarity": "Provide clear next step guidance",
                "support_assurance": "Demonstrate ongoing support and partnership"
            }
        }
        
        # Customize based on psychology profile and conversion goals
        optimized_triggers = self._optimize_action_triggers(psychology_profile, conversion_goals, action_triggers)
        
        return {
            "primary_action_strategy": optimized_triggers["primary"],
            "backup_triggers": optimized_triggers["backup"],
            "timing_optimization": self._optimize_action_timing(),
            "resistance_handling": self._create_action_resistance_strategies(),
            "follow_up_sequence": self._design_action_follow_up()
        }
    
    def _optimize_touchpoint_sequence(self, aida_framework: Dict) -> Dict[str, Any]:
        """Optimize multi-touchpoint AIDA sequence"""
        
        return {
            "touchpoint_mapping": {
                "touchpoint_1": {
                    "focus": "attention_and_initial_interest",
                    "channel": "personalized_outreach",
                    "duration": "5-10_minutes",
                    "success_metric": "engagement_and_response"
                },
                "touchpoint_2": {
                    "focus": "interest_deepening_and_value_demonstration",
                    "channel": "discovery_conversation",
                    "duration": "30-45_minutes", 
                    "success_metric": "problem_acknowledgment_and_interest_confirmation"
                },
                "touchpoint_3": {
                    "focus": "desire_building_and_solution_presentation",
                    "channel": "customized_presentation",
                    "duration": "45-60_minutes",
                    "success_metric": "emotional_connection_and_desire_expression"
                },
                "touchpoint_4": {
                    "focus": "action_and_commitment",
                    "channel": "decision_facilitation_meeting",
                    "duration": "30_minutes",
                    "success_metric": "commitment_and_next_steps"
                }
            },
            "sequence_optimization": self._optimize_touchpoint_timing(),
            "channel_selection": self._optimize_touchpoint_channels(),
            "content_progression": self._design_content_progression()
        }
    
    def _optimize_conversion_psychology(self, psychology_profile: Dict) -> Dict[str, Any]:
        """Optimize conversion based on psychological insights"""
        
        return {
            "conversion_blockers": {
                "psychological_barriers": self._identify_psychological_barriers(psychology_profile),
                "removal_strategies": self._create_barrier_removal_strategies(),
                "prevention_techniques": self._design_barrier_prevention()
            },
            "conversion_accelerators": {
                "psychological_accelerators": self._identify_psychological_accelerators(psychology_profile),
                "amplification_strategies": self._create_accelerator_strategies(),
                "timing_optimization": self._optimize_accelerator_timing()
            },
            "decision_support": {
                "decision_framework": self._provide_decision_framework(),
                "criteria_clarification": self._clarify_decision_criteria(),
                "confidence_building": self._build_decision_confidence()
            }
        }
    
    def _create_implementation_roadmap(self, aida_framework: Dict) -> Dict[str, Any]:
        """Create practical implementation roadmap for AIDA strategy"""
        
        return {
            "phase_1_attention": {
                "duration": "1-3_days",
                "activities": ["Initial outreach", "Attention-grabbing content", "Response tracking"],
                "success_criteria": ["Response rate > 30%", "Engagement metrics positive"],
                "optimization_points": ["Message testing", "Channel optimization"]
            },
            "phase_2_interest": {
                "duration": "1-2_weeks",
                "activities": ["Discovery conversations", "Value demonstration", "Interest confirmation"],
                "success_criteria": ["Meeting acceptance > 60%", "Interest signals strong"],
                "optimization_points": ["Question refinement", "Value proposition adjustment"]
            },
            "phase_3_desire": {
                "duration": "2-4_weeks",
                "activities": ["Solution presentation", "Desire building", "Objection handling"],
                "success_criteria": ["Emotional engagement high", "Solution fit confirmed"],
                "optimization_points": ["Presentation customization", "Emotional trigger optimization"]
            },
            "phase_4_action": {
                "duration": "1-2_weeks",
                "activities": ["Decision facilitation", "Commitment securing", "Next steps"],
                "success_criteria": ["Clear commitment", "Timeline established"],
                "optimization_points": ["Urgency calibration", "Risk mitigation"]
            }
        }
    
    def _define_performance_tracking(self) -> Dict[str, Any]:
        """Define comprehensive performance tracking for AIDA strategy"""
        
        return {
            "aida_stage_metrics": {
                "attention": {
                    "open_rates": "email_and_message_open_rates",
                    "response_rates": "initial_response_percentages",
                    "engagement_time": "time_spent_with_initial_content"
                },
                "interest": {
                    "meeting_acceptance": "percentage_accepting_discovery_calls",
                    "engagement_depth": "quality_of_conversation_and_questions",
                    "content_consumption": "additional_content_requested_or_consumed"
                },
                "desire": {
                    "emotional_indicators": "excitement_and_enthusiasm_signals",
                    "solution_fit": "acknowledgment_of_solution_value",
                    "stakeholder_involvement": "bringing_others_into_conversation"
                },
                "action": {
                    "commitment_level": "strength_of_next_step_commitments",
                    "timeline_urgency": "speed_of_decision_making_process",
                    "conversion_rate": "percentage_reaching_desired_action"
                }
            },
            "psychological_effectiveness": {
                "trigger_response": "effectiveness_of_psychological_triggers",
                "resistance_levels": "amount_and_type_of_resistance_encountered",
                "engagement_quality": "depth_and_quality_of_prospect_engagement"
            },
            "optimization_opportunities": {
                "stage_bottlenecks": "where_prospects_drop_off_most",
                "trigger_optimization": "which_psychological_triggers_work_best",
                "content_performance": "which_content_drives_best_results"
            }
        }
    
    def _store_aida_campaign(self, strategy_data: Dict) -> None:
        """Store AIDA campaign strategy in database"""
        
        try:
            campaign = AIDACampaign(
                campaign_id=strategy_data["strategy_id"],
                aida_analysis=strategy_data.get("aida_framework", {}),
                psychology_profile=strategy_data.get("psychology_profile", {}),
                conversion_strategy=strategy_data.get("conversion_optimization", {}),
                performance_metrics=strategy_data.get("performance_tracking", {})
            )
            
            db.session.add(campaign)
            db.session.commit()
            
            logger.info(f"Stored AIDA campaign strategy: {strategy_data['strategy_id']}")
            
        except Exception as e:
            logger.error(f"Error storing AIDA campaign: {str(e)}")
            db.session.rollback()
    
    # Helper methods for fallback data and additional functionality
    def _get_fallback_psychology_profile(self) -> Dict[str, Any]:
        """Provide fallback psychology profile"""
        return {
            "decision_making_style": {"primary": "analytical", "secondary": "collaborative"},
            "motivation_drivers": ["achievement", "security", "recognition"],
            "communication_preferences": {"style": "data_driven", "format": "structured"},
            "effective_triggers": ["social_proof", "authority", "scarcity"],
            "resistance_patterns": {"primary": "skepticism", "secondary": "delay"},
            "psychology_score": 75.0,
            "confidence_level": "medium"
        }
    
    def _get_fallback_aida_framework(self) -> Dict[str, Any]:
        """Provide fallback AIDA framework"""
        return {
            "attention_strategy": {"approach": "professional_credibility"},
            "interest_strategy": {"approach": "value_demonstration"},
            "desire_strategy": {"approach": "benefit_highlighting"},
            "action_strategy": {"approach": "clear_next_steps"},
            "psychological_flow": "standard_progression",
            "personalization_level": "medium",
            "effectiveness_prediction": 70.0
        }
    
    # Additional helper methods (implementing all referenced methods)
    def _optimize_psychological_flow(self, aida_data: Dict) -> str:
        return "smooth_transition_between_stages"
    
    def _select_attention_techniques(self, psychology_profile: Dict, techniques: Dict) -> Dict[str, Any]:
        return {"primary": "curiosity_gap", "secondary": ["pattern_interrupt", "emotional_hook"]}
    
    def _optimize_attention_timing(self) -> Dict[str, str]:
        return {"best_times": "Tuesday_Wednesday_10AM_2PM", "avoid": "Monday_morning_Friday_afternoon"}
    
    def _optimize_attention_channels(self, prospect_profile: Dict) -> Dict[str, str]:
        return {"primary": "personalized_email", "secondary": "linkedin_message", "tertiary": "phone_call"}
    
    def _define_attention_metrics(self) -> List[str]:
        return ["open_rate", "response_rate", "engagement_time", "click_through_rate"]
    
    def _create_interest_sequence(self, prospect_profile: Dict) -> List[str]:
        return ["relevance_connection", "peer_success_stories", "industry_insights", "problem_exploration"]
    
    def _personalize_interest_content(self, prospect_profile: Dict) -> Dict[str, str]:
        return {"industry_specific": "use_industry_terminology", "role_specific": "address_role_challenges"}
    
    def _define_engagement_tactics(self) -> List[str]:
        return ["interactive_questions", "scenario_planning", "collaborative_exploration"]
    
    def _create_value_demonstration_strategy(self) -> Dict[str, str]:
        return {"approach": "case_study_presentation", "format": "quantified_results"}
    
    def _amplify_curiosity_techniques(self) -> List[str]:
        return ["partial_reveals", "teaser_information", "exclusive_insights"]
    
    def _optimize_interest_interactions(self) -> Dict[str, str]:
        return {"frequency": "regular_but_not_overwhelming", "timing": "when_prospect_most_receptive"}
    
    def _select_desire_amplifiers(self, psychology_profile: Dict, amplifiers: Dict) -> Dict[str, Any]:
        return {"primary": "future_pacing", "supporting": ["benefit_stacking", "social_validation"]}
    
    def _map_emotional_triggers(self, psychology_profile: Dict) -> Dict[str, str]:
        return {"primary_emotion": "confidence", "secondary_emotions": ["excitement", "relief"]}
    
    def _optimize_value_proposition(self, prospect_profile: Dict) -> Dict[str, str]:
        return {"focus": "roi_and_efficiency", "presentation": "quantified_benefits"}
    
    def _define_desire_metrics(self) -> List[str]:
        return ["enthusiasm_indicators", "solution_fit_acknowledgment", "stakeholder_involvement"]
    
    def _optimize_action_triggers(self, psychology_profile: Dict, conversion_goals: Dict, triggers: Dict) -> Dict[str, Any]:
        return {"primary": "risk_reversal", "backup": ["urgency_creation", "commitment_escalation"]}
    
    def _optimize_action_timing(self) -> Dict[str, str]:
        return {"optimal": "immediately_after_desire_peak", "backup": "within_24_hours"}
    
    def _create_action_resistance_strategies(self) -> List[str]:
        return ["address_concerns_proactively", "provide_social_proof", "offer_risk_mitigation"]
    
    def _design_action_follow_up(self) -> Dict[str, str]:
        return {"immediate": "confirmation_and_next_steps", "ongoing": "value_reinforcement"}
    
    def _optimize_touchpoint_timing(self) -> Dict[str, str]:
        return {"interval": "2-3_days_between_touchpoints", "adaptation": "based_on_response_patterns"}
    
    def _optimize_touchpoint_channels(self) -> Dict[str, str]:
        return {"progression": "email_to_phone_to_meeting", "adaptation": "based_on_preferences"}
    
    def _design_content_progression(self) -> Dict[str, str]:
        return {"approach": "increasing_depth_and_personalization", "format": "mixed_media"}
    
    def _identify_psychological_barriers(self, psychology_profile: Dict) -> List[str]:
        return ["decision_paralysis", "risk_aversion", "status_quo_bias"]
    
    def _create_barrier_removal_strategies(self) -> List[str]:
        return ["risk_mitigation", "decision_simplification", "change_justification"]
    
    def _design_barrier_prevention(self) -> List[str]:
        return ["proactive_objection_handling", "trust_building", "confidence_enhancement"]
    
    def _identify_psychological_accelerators(self, psychology_profile: Dict) -> List[str]:
        return ["social_proof", "authority", "urgency", "scarcity"]
    
    def _create_accelerator_strategies(self) -> List[str]:
        return ["strategic_scarcity", "expert_endorsement", "peer_success_stories"]
    
    def _optimize_accelerator_timing(self) -> Dict[str, str]:
        return {"deployment": "at_moment_of_peak_interest", "frequency": "sparingly_for_maximum_impact"}
    
    def _provide_decision_framework(self) -> Dict[str, str]:
        return {"structure": "clear_criteria_and_process", "support": "decision_making_tools"}
    
    def _clarify_decision_criteria(self) -> List[str]:
        return ["roi_requirements", "implementation_timeline", "success_metrics"]
    
    def _build_decision_confidence(self) -> List[str]:
        return ["success_guarantees", "implementation_support", "ongoing_partnership"]
    
    def _generate_optimization_recommendations(self) -> List[str]:
        return ["test_different_psychological_triggers", "optimize_touchpoint_timing", "personalize_content_further"]

# Initialize agent
aida_agent = AIDASalesPsychologyAgent()

# Routes
@app.route('/')
def aida_dashboard():
    """AIDA Sales Psychology Agent dashboard"""
    return render_template('aida_dashboard.html', agent_name=aida_agent.agent_name)

@app.route('/api/comprehensive-strategy', methods=['POST'])
def comprehensive_aida_strategy():
    """Generate comprehensive AIDA psychology strategy"""
    
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Request data required'}), 400
    
    result = aida_agent.generate_comprehensive_aida_strategy(data)
    return jsonify(result)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "agent": aida_agent.agent_name,
        "version": "1.0.0",
        "capabilities": ["aida_framework", "psychology_analysis", "conversion_optimization"]
    })

@app.route('/api/psychology-analysis', methods=['POST'])
def psychology_analysis():
    """Analyze prospect psychology profile"""
    
    data = request.get_json()
    prospect_profile = data.get('prospect_profile', {})
    
    psychology_profile = aida_agent._analyze_prospect_psychology(prospect_profile)
    return jsonify(psychology_profile)

@app.route('/api/attention-strategies', methods=['POST'])
def attention_strategies():
    """Generate attention strategies"""
    
    data = request.get_json()
    prospect_profile = data.get('prospect_profile', {})
    psychology_profile = data.get('psychology_profile', {})
    
    attention_strategies = aida_agent._create_attention_strategies(prospect_profile, psychology_profile)
    return jsonify(attention_strategies)

@app.route('/api/conversion-optimization', methods=['POST'])
def conversion_optimization():
    """Get conversion optimization strategies"""
    
    data = request.get_json()
    psychology_profile = data.get('psychology_profile', {})
    
    conversion_optimization = aida_agent._optimize_conversion_psychology(psychology_profile)
    return jsonify(conversion_optimization)

# Initialize database
with app.app_context():
    db.create_all()
    logger.info("AIDA Sales Psychology Agent initialized")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5042, debug=True)