"""
Pydantic contracts for Synthetic Biology Engineering Agent
Typed input/output models for all agent capabilities
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime

class OrganismType(str, Enum):
    BACTERIA = "bacteria"
    YEAST = "yeast"
    MAMMALIAN_CELLS = "mammalian_cells"
    PLANT_CELLS = "plant_cells"
    ALGAE = "algae"
    FUNGI = "fungi"

class BiologyApplicationType(str, Enum):
    THERAPEUTIC_ENGINEERING = "therapeutic_engineering"
    BIOMANUFACTURING = "biomanufacturing"
    AGRICULTURAL_BIOTECHNOLOGY = "agricultural_biotechnology"
    ENVIRONMENTAL_REMEDIATION = "environmental_remediation"
    BIOSENSORS = "biosensors"
    SYNTHETIC_MATERIALS = "synthetic_materials"
    METABOLIC_ENGINEERING = "metabolic_engineering"

# Genetic Circuit Design
class GeneticCircuitDesignInput(BaseModel):
    target_function: str = Field(..., description="Specific biological function to engineer")
    organism: OrganismType = Field(..., description="Target organism type")
    performance_targets: Dict[str, float] = Field(..., description="Performance metrics and targets")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Design constraints")
    design_id: Optional[str] = Field(None, description="Unique design identifier")

class GeneticPartSpec(BaseModel):
    part_name: str
    part_type: str
    sequence: Optional[str] = None
    function: str
    compatibility_score: float

class GeneticCircuitOutput(BaseModel):
    circuit_design_id: str
    architecture: Dict[str, Any]
    genetic_parts: List[GeneticPartSpec]
    performance_predictions: Dict[str, float]
    validation_protocol: Dict[str, Any]
    manufacturing_analysis: Dict[str, Any]
    safety_compliance: Dict[str, Any]

# Biomanufacturing Process
class BiomanufacturingProcessInput(BaseModel):
    target_product: str = Field(..., description="Product to manufacture")
    production_scale: str = Field(..., description="Production scale requirement")
    organism: Optional[OrganismType] = Field(None, description="Preferred organism")
    economic_targets: Dict[str, float] = Field(default_factory=dict, description="Economic objectives")
    process_id: Optional[str] = Field(None, description="Process identifier")

class BiomanufacturingProcessOutput(BaseModel):
    process_id: str
    production_system: Dict[str, Any]
    bioprocess_conditions: Dict[str, Any]
    downstream_processing: Dict[str, Any]
    economic_projections: Dict[str, Any]
    implementation_roadmap: Dict[str, Any]

# Therapeutic Protein Engineering
class TherapeuticProteinInput(BaseModel):
    target_disease: str = Field(..., description="Disease or condition to target")
    protein_target: str = Field(..., description="Target protein or pathway")
    modality: str = Field(..., description="Therapeutic modality (antibody, enzyme, etc.)")
    goals: List[str] = Field(default_factory=list, description="Development goals")
    therapeutic_id: Optional[str] = Field(None, description="Therapeutic identifier")

class TherapeuticProteinOutput(BaseModel):
    therapeutic_id: str
    protein_design: Dict[str, Any]
    therapeutic_properties: Dict[str, Any]
    production_system: Dict[str, Any]
    drug_delivery: Dict[str, Any]
    safety_profile: Dict[str, Any]
    development_pathway: Dict[str, Any]

# Agricultural Biotechnology
class AgriculturalBiotechInput(BaseModel):
    crop_type: str = Field(..., description="Target crop species")
    desired_traits: List[str] = Field(..., description="Traits to engineer")
    market_requirements: Dict[str, Any] = Field(default_factory=dict, description="Market specifications")
    regulatory_constraints: List[str] = Field(default_factory=list, description="Regulatory limitations")
    project_id: Optional[str] = Field(None, description="Project identifier")

class AgriculturalBiotechOutput(BaseModel):
    project_id: str
    trait_development: Dict[str, Any]
    genetic_engineering: Dict[str, Any]
    field_testing: Dict[str, Any]
    environmental_assessment: Dict[str, Any]
    regulatory_strategy: Dict[str, Any]
    commercialization_plan: Dict[str, Any]

# Environmental Remediation
class EnvironmentalRemediationInput(BaseModel):
    contamination_type: str = Field(..., description="Type of environmental contamination")
    contamination_level: Dict[str, float] = Field(..., description="Contamination measurements")
    site_conditions: Dict[str, Any] = Field(..., description="Environmental site conditions")
    regulatory_requirements: List[str] = Field(default_factory=list, description="Regulatory compliance needs")
    project_id: Optional[str] = Field(None, description="Remediation project identifier")

class EnvironmentalRemediationOutput(BaseModel):
    project_id: str
    bioremediation_strategy: Dict[str, Any]
    organism_selection: Dict[str, Any]
    implementation_plan: Dict[str, Any]
    monitoring_protocol: Dict[str, Any]
    regulatory_compliance: Dict[str, Any]
    success_metrics: Dict[str, Any]

# Biosensor Development
class BiosensorDevelopmentInput(BaseModel):
    target_analyte: str = Field(..., description="Target molecule or condition to detect")
    detection_requirements: Dict[str, Any] = Field(..., description="Sensitivity and specificity requirements")
    application_environment: str = Field(..., description="Intended use environment")
    performance_specifications: Dict[str, float] = Field(..., description="Performance criteria")
    sensor_id: Optional[str] = Field(None, description="Biosensor identifier")

class BiosensorDevelopmentOutput(BaseModel):
    sensor_id: str
    sensor_design: Dict[str, Any]
    biological_components: Dict[str, Any]
    detection_mechanism: Dict[str, Any]
    performance_validation: Dict[str, Any]
    manufacturing_process: Dict[str, Any]
    deployment_strategy: Dict[str, Any]

# Safety and Ethics Assessment
class SafetyAssessmentInput(BaseModel):
    project_type: BiologyApplicationType = Field(..., description="Type of synthetic biology project")
    biological_components: List[str] = Field(..., description="Biological parts and organisms involved")
    intended_use: str = Field(..., description="Intended application and deployment")
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    containment_level: Optional[str] = Field(None, description="Required containment level")

class SafetyAssessmentOutput(BaseModel):
    risk_assessment: Dict[str, Any]
    safety_recommendations: List[str]
    containment_requirements: Dict[str, Any]
    regulatory_pathway: Dict[str, Any]
    monitoring_protocols: List[str]
    emergency_procedures: List[str]
    ethical_considerations: Dict[str, Any]