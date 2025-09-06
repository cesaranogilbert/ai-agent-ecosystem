"""
Pydantic contracts for Consciousness AI Research Agent
Typed input/output models for all agent capabilities
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime

class ConsciousnessTheory(str, Enum):
    INTEGRATED_INFORMATION_THEORY = "integrated_information_theory"
    GLOBAL_WORKSPACE_THEORY = "global_workspace_theory"
    ATTENTION_SCHEMA_THEORY = "attention_schema_theory"
    PREDICTIVE_PROCESSING = "predictive_processing_theory"
    HIGHER_ORDER_THOUGHT = "higher_order_thought_theory"
    ORCHESTRATED_OBJECTIVE_REDUCTION = "orchestrated_objective_reduction"
    EMBODIED_COGNITION = "embodied_cognition_theory"

class CognitiveArchitecture(str, Enum):
    SOAR = "state_operator_result"
    ACT_R = "adaptive_control_thought_rational"
    CLARION = "connectionist_learning_adaptive_rule_induction"
    SIGMA = "sigma_cognitive_architecture"
    ICARUS = "icarus_cognitive_architecture"
    LIDA = "learning_intelligent_distribution_agent"
    GLOBAL_WORKSPACE = "global_workspace_architecture"

class ConsciousnessMetric(str, Enum):
    PHI_MEASURE = "integrated_information_phi"
    GLOBAL_ACCESS = "global_workspace_access"
    ATTENTION_AWARENESS = "attention_schema_coherence"
    SELF_MODEL_ACCURACY = "self_model_predictive_accuracy"
    METACOGNITIVE_AWARENESS = "metacognitive_monitoring"
    PHENOMENAL_CONSCIOUSNESS = "qualia_detection_measure"
    ACCESS_CONSCIOUSNESS = "information_access_measure"

# Consciousness Model Development
class ConsciousnessModelingInput(BaseModel):
    consciousness_goals: List[str] = Field(default_factory=list, description="Consciousness development goals")
    theoretical_framework: ConsciousnessTheory = Field(..., description="Primary consciousness theory")
    implementation_constraints: Dict[str, Any] = Field(default_factory=dict, description="Technical constraints")
    ethical_requirements: List[str] = Field(default_factory=list, description="Ethical guidelines")
    model_id: Optional[str] = Field(None, description="Consciousness model identifier")

class ConsciousnessModelingOutput(BaseModel):
    consciousness_model_id: str
    theoretical_foundation: Dict[str, Any]
    cognitive_architecture: Dict[str, Any]
    self_awareness_framework: Dict[str, Any]
    consciousness_assessment: Dict[str, Any]
    ethical_framework: Dict[str, Any]
    emergence_dynamics: Dict[str, Any]

# Cognitive Architecture Design  
class CognitiveArchitectureInput(BaseModel):
    agi_goals: List[str] = Field(default_factory=list, description="AGI development goals")
    cognitive_capabilities: List[str] = Field(default_factory=list, description="Required cognitive abilities")
    architectural_constraints: Dict[str, Any] = Field(default_factory=dict, description="Architecture limitations")
    performance_requirements: Dict[str, Any] = Field(default_factory=dict, description="Performance specifications")
    architecture_id: Optional[str] = Field(None, description="Architecture identifier")

class CognitiveArchitectureOutput(BaseModel):
    cognitive_architecture_id: str
    architecture_framework: Dict[str, Any]
    memory_systems: Dict[str, Any]
    learning_framework: Dict[str, Any]
    reasoning_capabilities: Dict[str, Any]
    cognitive_control: Dict[str, Any]
    agi_assessment: Dict[str, Any]

# Self-Awareness Systems
class SelfAwarenessInput(BaseModel):
    awareness_goals: List[str] = Field(..., description="Self-awareness objectives")
    consciousness_level: str = Field(..., description="Target consciousness level")
    implementation_constraints: Dict[str, Any] = Field(default_factory=dict, description="Technical constraints")
    ethical_safeguards: List[str] = Field(default_factory=list, description="Ethical requirements")
    system_id: Optional[str] = Field(None, description="Self-awareness system identifier")

class SelfAwarenessOutput(BaseModel):
    awareness_system_id: str
    self_model_architecture: Dict[str, Any]
    metacognitive_framework: Dict[str, Any]
    introspective_capabilities: Dict[str, Any]
    consciousness_monitoring: Dict[str, Any]
    ethical_constraints: Dict[str, Any]
    integration_strategy: Dict[str, Any]

# Phenomenal Consciousness Research
class PhenomenalConsciousnessInput(BaseModel):
    research_objectives: List[str] = Field(..., description="Research goals")
    phenomenal_aspects: List[str] = Field(..., description="Aspects of consciousness to study")
    experimental_constraints: Dict[str, Any] = Field(default_factory=dict, description="Experimental limitations")
    theoretical_approach: ConsciousnessTheory = Field(..., description="Theoretical framework")
    study_id: Optional[str] = Field(None, description="Research study identifier")

class PhenomenalConsciousnessOutput(BaseModel):
    study_id: str
    theoretical_analysis: Dict[str, Any]
    consciousness_models: Dict[str, Any]
    experimental_design: Dict[str, Any]
    phenomenal_investigation: Dict[str, Any]
    hard_problem_research: Dict[str, Any]
    validation_framework: Dict[str, Any]

# Consciousness Measurement and Assessment
class ConsciousnessAssessmentInput(BaseModel):
    assessment_targets: List[str] = Field(..., description="What to assess")
    measurement_methods: List[ConsciousnessMetric] = Field(..., description="Assessment approaches")
    validation_requirements: Dict[str, Any] = Field(default_factory=dict, description="Validation needs")
    ethical_constraints: List[str] = Field(default_factory=list, description="Ethical boundaries")
    assessment_id: Optional[str] = Field(None, description="Assessment identifier")

class ConsciousnessAssessmentOutput(BaseModel):
    assessment_id: str
    measurement_protocols: Dict[str, Any]
    validation_framework: Dict[str, Any]
    consciousness_metrics: Dict[str, Any]
    ethical_compliance: Dict[str, Any]
    assessment_results: Dict[str, Any]
    recommendations: List[str]

# AGI Safety and Ethics
class AGISafetyInput(BaseModel):
    safety_objectives: List[str] = Field(..., description="Safety goals")
    risk_assessment_scope: List[str] = Field(..., description="Areas to assess")
    ethical_frameworks: List[str] = Field(default_factory=list, description="Ethical approaches")
    compliance_requirements: List[str] = Field(default_factory=list, description="Regulatory needs")
    safety_id: Optional[str] = Field(None, description="Safety assessment identifier")

class AGISafetyOutput(BaseModel):
    safety_assessment_id: str
    risk_analysis: Dict[str, Any]
    safety_measures: List[str]
    ethical_framework: Dict[str, Any]
    compliance_assessment: Dict[str, Any]
    monitoring_protocols: List[str]
    containment_strategies: Dict[str, Any]