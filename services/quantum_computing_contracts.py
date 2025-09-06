"""
Pydantic contracts for Quantum Computing Optimization Agent
Typed input/output models for all agent capabilities
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime

class QuantumPlatform(str, Enum):
    IBM_QUANTUM = "ibm_quantum_network"
    GOOGLE_QUANTUM_AI = "google_sycamore"
    RIGETTI_FOREST = "rigetti_quantum_cloud"
    AMAZON_BRAKET = "aws_quantum_computing"
    MICROSOFT_AZURE_QUANTUM = "azure_quantum"
    IONQ_QUANTUM = "ionq_trapped_ion"
    XANADU_PENNYLANE = "xanadu_photonic"

class QuantumAlgorithmType(str, Enum):
    OPTIMIZATION = "quantum_optimization"
    MACHINE_LEARNING = "quantum_machine_learning"
    SIMULATION = "quantum_simulation"
    CRYPTOGRAPHY = "quantum_cryptography"
    SEARCH = "quantum_search"
    FACTORING = "quantum_factoring"
    CHEMISTRY = "quantum_chemistry"
    FINANCE = "quantum_finance"

class QuantumHardwareType(str, Enum):
    SUPERCONDUCTING = "superconducting_qubits"
    TRAPPED_ION = "trapped_ion_qubits"
    PHOTONIC = "photonic_qubits"
    NEUTRAL_ATOM = "neutral_atom_qubits"
    TOPOLOGICAL = "topological_qubits"
    DIAMOND_NV = "diamond_nitrogen_vacancy"

# Quantum Algorithm Optimization
class QuantumAlgorithmOptimizationInput(BaseModel):
    problem_type: str = Field(..., description="Type of problem to solve")
    hardware_constraints: Dict[str, Any] = Field(default_factory=dict, description="Hardware limitations")
    performance_targets: Dict[str, Any] = Field(default_factory=dict, description="Performance objectives")
    quantum_advantage_goals: List[str] = Field(default_factory=list, description="Quantum advantage targets")
    optimization_id: Optional[str] = Field(None, description="Optimization identifier")

class QuantumCircuitSpec(BaseModel):
    qubit_count: int
    gate_count: int
    circuit_depth: int
    fidelity: float
    error_rate: float

class QuantumAlgorithmOptimizationOutput(BaseModel):
    optimization_id: str
    algorithm_design: Dict[str, Any]
    circuit_implementation: Dict[str, Any]
    error_mitigation: Dict[str, Any]
    hybrid_framework: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    resource_requirements: Dict[str, Any]

# Quantum Machine Learning
class QuantumMLInput(BaseModel):
    ml_problem_type: str = Field(..., description="Type of machine learning problem")
    data_characteristics: Dict[str, Any] = Field(default_factory=dict, description="Dataset properties")
    quantum_advantage_targets: List[str] = Field(default_factory=list, description="QML advantage goals")
    classical_baseline: Dict[str, Any] = Field(default_factory=dict, description="Classical performance baseline")
    system_id: Optional[str] = Field(None, description="QML system identifier")

class QuantumMLOutput(BaseModel):
    qml_system_id: str
    algorithm_selection: Dict[str, Any]
    quantum_feature_encoding: Dict[str, Any]
    circuit_architecture: Dict[str, Any]
    training_framework: Dict[str, Any]
    quantum_advantage: Dict[str, Any]
    deployment_strategy: Dict[str, Any]

# Quantum Simulation
class QuantumSimulationInput(BaseModel):
    target_system: str = Field(..., description="Physical system to simulate")
    simulation_objectives: List[str] = Field(..., description="Simulation goals")
    computational_resources: Dict[str, Any] = Field(default_factory=dict, description="Available resources")
    accuracy_requirements: Dict[str, float] = Field(default_factory=dict, description="Required precision")
    simulation_id: Optional[str] = Field(None, description="Simulation identifier")

class QuantumSimulationOutput(BaseModel):
    simulation_id: str
    system_modeling: Dict[str, Any]
    algorithm_selection: Dict[str, Any]
    hamiltonian_decomposition: Dict[str, Any]
    error_analysis: Dict[str, Any]
    scientific_insights: Dict[str, Any]
    validation_strategy: Dict[str, Any]

# Fault-Tolerant Quantum Computing
class FaultTolerantQCInput(BaseModel):
    target_application: str = Field(..., description="Target fault-tolerant application")
    error_requirements: Dict[str, float] = Field(..., description="Error rate specifications")
    resource_constraints: Dict[str, Any] = Field(default_factory=dict, description="Resource limitations")
    timeline_requirements: str = Field(..., description="Development timeline")
    system_id: Optional[str] = Field(None, description="FT system identifier")

class FaultTolerantQCOutput(BaseModel):
    ft_system_id: str
    error_correction_strategy: Dict[str, Any]
    logical_qubit_design: Dict[str, Any]
    fault_tolerant_gates: Dict[str, Any]
    resource_optimization: Dict[str, Any]
    error_threshold_analysis: Dict[str, Any]
    implementation_roadmap: Dict[str, Any]

# Quantum Hardware Characterization
class QuantumHardwareCharacterizationInput(BaseModel):
    hardware_type: QuantumHardwareType = Field(..., description="Type of quantum hardware")
    characterization_goals: List[str] = Field(..., description="Characterization objectives")
    measurement_constraints: Dict[str, Any] = Field(default_factory=dict, description="Measurement limitations")
    benchmarking_requirements: Dict[str, Any] = Field(default_factory=dict, description="Benchmark specifications")
    system_id: Optional[str] = Field(None, description="Hardware system identifier")

class QuantumHardwareCharacterizationOutput(BaseModel):
    system_id: str
    hardware_analysis: Dict[str, Any]
    performance_benchmarks: Dict[str, Any]
    noise_characterization: Dict[str, Any]
    optimization_recommendations: Dict[str, Any]
    calibration_protocols: Dict[str, Any]
    performance_projections: Dict[str, Any]

# Quantum Security and Cryptography
class QuantumSecurityInput(BaseModel):
    security_application: str = Field(..., description="Security application type")
    threat_model: Dict[str, Any] = Field(..., description="Security threat model")
    performance_requirements: Dict[str, Any] = Field(default_factory=dict, description="Performance needs")
    compliance_requirements: List[str] = Field(default_factory=list, description="Regulatory compliance")
    security_id: Optional[str] = Field(None, description="Security system identifier")

class QuantumSecurityOutput(BaseModel):
    security_system_id: str
    cryptographic_design: Dict[str, Any]
    security_analysis: Dict[str, Any]
    implementation_strategy: Dict[str, Any]
    performance_evaluation: Dict[str, Any]
    compliance_assessment: Dict[str, Any]
    deployment_considerations: Dict[str, Any]