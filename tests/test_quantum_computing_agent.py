"""
Comprehensive test suite for Quantum Computing Optimization Agent
Tests quantum algorithms, hardware integration, and security protocols
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import Dict, Any

from services.quantum_computing_optimization_agent import QuantumComputingOptimizationAgent
from services.quantum_computing_contracts import (
    QuantumAlgorithmOptimizationInput, QuantumAlgorithmOptimizationOutput,
    QuantumMLInput, QuantumMLOutput,
    QuantumSimulationInput, QuantumSimulationOutput,
    FaultTolerantQCInput, FaultTolerantQCOutput,
    QuantumHardwareCharacterizationInput, QuantumHardwareCharacterizationOutput,
    QuantumSecurityInput, QuantumSecurityOutput,
    QuantumPlatform, OptimizationProblem, QuantumAdvantage, MLType
)


class TestQuantumComputingAgent:
    """Test suite for Quantum Computing Optimization Agent"""
    
    @pytest.fixture
    def agent(self):
        """Create agent instance for testing"""
        return QuantumComputingOptimizationAgent()
    
    @pytest.fixture
    def sample_algorithm_input(self):
        """Sample input for quantum algorithm optimization"""
        return {
            "problem_type": OptimizationProblem.PORTFOLIO_OPTIMIZATION,
            "quantum_advantage_goals": [QuantumAdvantage.EXPONENTIAL_SPEEDUP],
            "hardware_constraints": {"qubit_count": 50, "coherence_time": 100},
            "performance_targets": {"approximation_ratio": 0.9, "runtime": 60},
            "algorithm_id": "test_algo_001"
        }
    
    @pytest.fixture 
    def sample_qml_input(self):
        """Sample input for quantum machine learning"""
        return {
            "ml_type": MLType.CLASSIFICATION,
            "dataset_characteristics": {"samples": 1000, "features": 20, "classes": 2},
            "quantum_advantage_targets": [QuantumAdvantage.FEATURE_SPACE_ENHANCEMENT],
            "classical_baseline": {"algorithm": "SVM", "accuracy": 0.85},
            "qml_id": "test_qml_001"
        }

    # Contract Validation Tests
    @pytest.mark.asyncio
    async def test_algorithm_optimization_contract_validation(self, agent, sample_algorithm_input):
        """Test Pydantic contract validation for algorithm optimization"""
        # Valid input should pass
        input_model = QuantumAlgorithmOptimizationInput(**sample_algorithm_input)
        assert input_model.problem_type == OptimizationProblem.PORTFOLIO_OPTIMIZATION
        assert input_model.quantum_advantage_goals == [QuantumAdvantage.EXPONENTIAL_SPEEDUP]
        
        # Invalid problem type should fail
        invalid_input = sample_algorithm_input.copy()
        invalid_input["problem_type"] = "INVALID_PROBLEM"
        with pytest.raises(ValueError):
            QuantumAlgorithmOptimizationInput(**invalid_input)
    
    @pytest.mark.asyncio
    async def test_quantum_ml_contract_validation(self, agent, sample_qml_input):
        """Test Pydantic contract validation for quantum ML"""
        # Valid input should pass
        input_model = QuantumMLInput(**sample_qml_input)
        assert input_model.ml_type == MLType.CLASSIFICATION
        assert input_model.quantum_advantage_targets == [QuantumAdvantage.FEATURE_SPACE_ENHANCEMENT]
        
        # Missing required field should fail
        invalid_input = sample_qml_input.copy()
        del invalid_input["ml_type"]
        with pytest.raises(ValueError):
            QuantumMLInput(**invalid_input)
    
    @pytest.mark.asyncio
    async def test_quantum_simulation_contracts(self, agent):
        """Test quantum simulation contract validation"""
        valid_input = {
            "target_system": "molecular_hydrogen",
            "simulation_objectives": ["ground_state_energy", "phase_transitions"],
            "accuracy_requirements": {"energy_precision": 0.001, "convergence": 0.01},
            "computational_resources": {"qubit_budget": 100, "time_limit": 3600},
            "simulation_id": "sim_001"
        }
        
        input_model = QuantumSimulationInput(**valid_input)
        assert input_model.target_system == "molecular_hydrogen"
        assert "ground_state_energy" in input_model.simulation_objectives

    # Quantum Security Tests  
    @pytest.mark.asyncio
    async def test_quantum_cryptography_security(self, agent):
        """Test quantum cryptography and security protocols"""
        security_input = {
            "security_objectives": ["quantum_key_distribution", "post_quantum_crypto"],
            "threat_model": ["quantum_adversary", "classical_attacks"],
            "compliance_requirements": ["FIPS_140", "Common_Criteria"],
            "implementation_constraints": {"distance": 100, "key_rate": 1000},
            "security_id": "qcrypto_001"
        }
        
        result = await agent.execute("quantum_security", security_input)
        
        assert "security_system_id" in result
        assert "cryptographic_design" in result
        assert "security_analysis" in result
        assert result["security_analysis"]["threat_resistance"]["quantum_attacks"] == "information_theoretic_security"
    
    @pytest.mark.asyncio
    async def test_export_control_compliance(self, agent):
        """Test export control and dual-use technology compliance"""
        restricted_input = {
            "problem_type": OptimizationProblem.CRYPTOGRAPHY_BREAKING,
            "quantum_advantage_goals": [QuantumAdvantage.EXPONENTIAL_SPEEDUP],
            "hardware_constraints": {"qubit_count": 1000, "gate_fidelity": 0.999},
            "performance_targets": {"key_breaking_time": 1},
            "algorithm_id": "restricted_001"
        }
        
        # Should trigger export control validation
        with patch.object(agent, '_validate_export_controls') as mock_validate:
            mock_validate.return_value = {"approved": False, "reason": "dual_use_concern"}
            
            result = await agent.execute("optimize_quantum_algorithms", restricted_input)
            # Should include compliance warnings
            assert "regulatory_compliance" in result
    
    @pytest.mark.asyncio
    async def test_quantum_hardware_security(self, agent):
        """Test quantum hardware characterization security"""
        hardware_input = {
            "hardware_type": QuantumPlatform.IBM_QUANTUM,
            "characterization_goals": ["performance_benchmarking", "noise_analysis"],
            "security_requirements": ["tamper_detection", "side_channel_protection"],
            "access_level": "restricted",
            "system_id": "secure_hw_001"
        }
        
        result = await agent.execute("quantum_hardware_characterization", hardware_input)
        
        assert "system_id" in result
        assert "optimization_recommendations" in result
        assert "calibration_protocols" in result

    # Quantum Algorithm Tests
    @pytest.mark.asyncio
    async def test_optimize_quantum_algorithms_capability(self, agent, sample_algorithm_input):
        """Test quantum algorithm optimization capability"""
        result = await agent.execute("optimize_quantum_algorithms", sample_algorithm_input)
        
        # Validate output structure
        assert "algorithm_optimization_id" in result
        assert "quantum_algorithm_design" in result
        assert "optimization_strategy" in result
        assert "performance_analysis" in result
        assert "resource_requirements" in result
        
        # Validate algorithm components
        algorithm = result["quantum_algorithm_design"]
        assert "algorithm_choice" in algorithm
        assert "circuit_design" in algorithm
        assert "parameter_optimization" in algorithm
    
    @pytest.mark.asyncio
    async def test_quantum_ml_capability(self, agent, sample_qml_input):
        """Test quantum machine learning capability"""
        result = await agent.execute("develop_quantum_machine_learning", sample_qml_input)
        
        # Validate output structure
        assert "qml_system_id" in result
        assert "algorithm_selection" in result
        assert "quantum_feature_mapping" in result
        assert "training_strategy" in result
        assert "advantage_analysis" in result
        
        # Validate QML approach
        selection = result["algorithm_selection"]
        assert "algorithm_type" in selection
        assert "advantage_source" in selection
    
    @pytest.mark.asyncio
    async def test_quantum_simulation_capability(self, agent):
        """Test quantum simulation capability"""
        simulation_input = {
            "target_system": "chemical_reaction_network",
            "simulation_objectives": ["reaction_pathways", "energy_landscapes"],
            "accuracy_requirements": {"chemical_accuracy": 0.001, "convergence": 0.01},
            "computational_resources": {"qubit_budget": 80, "time_limit": 7200},
            "simulation_id": "chem_sim_001"
        }
        
        result = await agent.execute("simulate_quantum_systems", simulation_input)
        
        assert "simulation_id" in result
        assert "system_modeling" in result
        assert "simulation_strategy" in result
        assert "scientific_insights" in result
        
        # Validate simulation components
        modeling = result["system_modeling"]
        assert "hamiltonian_construction" in modeling
        assert "parameter_mapping" in modeling

    # Hardware Integration Tests
    @pytest.mark.asyncio
    async def test_quantum_platform_integration(self, agent):
        """Test integration with different quantum platforms"""
        platforms = [QuantumPlatform.IBM_QUANTUM, QuantumPlatform.GOOGLE_QUANTUM_AI, QuantumPlatform.RIGETTI_QUANTUM]
        
        for platform in platforms:
            hardware_input = {
                "hardware_type": platform,
                "characterization_goals": ["performance_benchmarking"],
                "security_requirements": [],
                "access_level": "public",
                "system_id": f"test_{platform.value}"
            }
            
            result = await agent.execute("quantum_hardware_characterization", hardware_input)
            assert "system_id" in result
            assert result["hardware_analysis"]["hardware_type"] == platform.value
    
    @pytest.mark.asyncio
    async def test_fault_tolerant_quantum_computing(self, agent):
        """Test fault-tolerant quantum computing capability"""
        ft_input = {
            "target_application": "quantum_simulation",
            "error_correction_requirements": {"logical_error_rate": 1e-6, "code_distance": 7},
            "resource_constraints": {"physical_qubits": 10000, "runtime": 86400},
            "timeline": "5_years",
            "ft_id": "ft_test_001"
        }
        
        result = await agent.execute("design_fault_tolerant_quantum_computing", ft_input)
        
        assert "ft_system_id" in result
        assert "error_correction_design" in result
        assert "resource_optimization" in result
        assert "implementation_roadmap" in result

    # Performance and Scalability Tests
    @pytest.mark.asyncio
    async def test_large_scale_optimization(self, agent):
        """Test large-scale quantum optimization problems"""
        large_input = {
            "problem_type": OptimizationProblem.PORTFOLIO_OPTIMIZATION,
            "quantum_advantage_goals": [QuantumAdvantage.QUADRATIC_SPEEDUP],
            "hardware_constraints": {"qubit_count": 1000, "coherence_time": 1000},
            "performance_targets": {"solution_quality": 0.95, "runtime": 3600},
            "algorithm_id": "large_scale_001"
        }
        
        result = await agent.execute("optimize_quantum_algorithms", large_input)
        assert "algorithm_optimization_id" in result
        assert result["resource_requirements"]["qubit_count"] <= 1000
    
    @pytest.mark.asyncio
    async def test_concurrent_quantum_operations(self, agent):
        """Test concurrent execution of quantum operations"""
        inputs = [
            ("optimize_quantum_algorithms", {
                "problem_type": OptimizationProblem.TRAVELING_SALESMAN,
                "quantum_advantage_goals": [QuantumAdvantage.EXPONENTIAL_SPEEDUP],
                "hardware_constraints": {"qubit_count": 20},
                "performance_targets": {},
                "algorithm_id": "concurrent_1"
            }),
            ("simulate_quantum_systems", {
                "target_system": "spin_glass",
                "simulation_objectives": ["ground_state"],
                "accuracy_requirements": {},
                "computational_resources": {},
                "simulation_id": "concurrent_2"
            })
        ]
        
        tasks = [agent.execute(capability, params) for capability, params in inputs]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 2
        assert all("id" in str(result) for result in results)

    # Integration Tests
    @pytest.mark.asyncio
    async def test_agent_base_inheritance(self, agent):
        """Test proper AgentBase inheritance"""
        capabilities = agent.get_capabilities()
        assert len(capabilities) == 6
        
        capability_names = [cap.name for cap in capabilities]
        expected_capabilities = [
            "optimize_quantum_algorithms",
            "develop_quantum_machine_learning",
            "simulate_quantum_systems",
            "design_fault_tolerant_quantum_computing",
            "quantum_hardware_characterization",
            "quantum_security"
        ]
        for expected in expected_capabilities:
            assert expected in capability_names
    
    @pytest.mark.asyncio
    async def test_quantum_noise_mitigation(self, agent):
        """Test quantum noise mitigation strategies"""
        # This tests the noise mitigation aspects of quantum algorithms
        noisy_input = {
            "problem_type": OptimizationProblem.MAX_CUT,
            "quantum_advantage_goals": [QuantumAdvantage.NOISE_RESILIENCE],
            "hardware_constraints": {"gate_error_rate": 0.01, "coherence_time": 50},
            "performance_targets": {"fidelity": 0.8},
            "algorithm_id": "noisy_test_001"
        }
        
        result = await agent.execute("optimize_quantum_algorithms", noisy_input)
        assert "noise_mitigation" in result["optimization_strategy"]
        assert "error_correction" in result["optimization_strategy"]["noise_mitigation"]

    # Error Handling Tests
    @pytest.mark.asyncio
    async def test_quantum_error_handling(self, agent):
        """Test quantum-specific error handling"""
        # Test invalid qubit count
        invalid_input = {
            "problem_type": OptimizationProblem.PORTFOLIO_OPTIMIZATION,
            "quantum_advantage_goals": [QuantumAdvantage.EXPONENTIAL_SPEEDUP],
            "hardware_constraints": {"qubit_count": -5},  # Invalid
            "performance_targets": {},
            "algorithm_id": "invalid_test"
        }
        
        with pytest.raises(ValueError):
            await agent.execute("optimize_quantum_algorithms", invalid_input)
    
    @pytest.mark.asyncio
    async def test_quantum_resource_validation(self, agent):
        """Test quantum resource constraint validation"""
        # Test unrealistic resource requirements
        unrealistic_input = {
            "target_system": "large_molecule",
            "simulation_objectives": ["full_simulation"],
            "accuracy_requirements": {"precision": 1e-15},  # Unrealistic precision
            "computational_resources": {"qubit_budget": 1000000},  # Unrealistic qubit count
            "simulation_id": "unrealistic_test"
        }
        
        result = await agent.execute("simulate_quantum_systems", unrealistic_input)
        # Should handle gracefully with warnings or adjustments
        assert "simulation_id" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])