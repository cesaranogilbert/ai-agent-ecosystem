"""
Comprehensive test suite for Synthetic Biology Engineering Agent
Tests contract validation, security safeguards, and operational capabilities
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import Dict, Any

from services.synthetic_biology_engineering_agent import SyntheticBiologyEngineeringAgent
from services.synthetic_biology_contracts import (
    BioengineeringDesignInput, BioengineeringDesignOutput,
    ProteinEngineeringInput, ProteinEngineeringOutput,
    MetabolicPathwayInput, MetabolicPathwayOutput,
    BiosafetyAssessmentInput, BiosafetyAssessmentOutput,
    RegulatoryComplianceInput, RegulatoryComplianceOutput,
    BiomanufacturingInput, BiomanufacturingOutput,
    SafetyLevel, ApplicationType
)


class TestSyntheticBiologyAgent:
    """Test suite for Synthetic Biology Engineering Agent"""
    
    @pytest.fixture
    def agent(self):
        """Create agent instance for testing"""
        return SyntheticBiologyEngineeringAgent()
    
    @pytest.fixture
    def sample_bioengineering_input(self):
        """Sample input for bioengineering design"""
        return {
            "design_objectives": ["enzyme_optimization", "stability_enhancement"],
            "target_applications": [ApplicationType.THERAPEUTIC],
            "safety_requirements": [SafetyLevel.BSL2],
            "regulatory_constraints": ["FDA_approval_required"],
            "design_id": "test_design_001"
        }
    
    @pytest.fixture
    def sample_protein_input(self):
        """Sample input for protein engineering"""
        return {
            "target_protein": "test_enzyme",
            "engineering_goals": ["increased_activity", "thermal_stability"],
            "structural_constraints": {"active_site": "conserved", "folding": "stable"},
            "performance_metrics": {"activity": 0.8, "stability": 0.7},
            "protein_id": "test_protein_001"
        }

    # Contract Validation Tests
    @pytest.mark.asyncio
    async def test_bioengineering_design_contract_validation(self, agent, sample_bioengineering_input):
        """Test Pydantic contract validation for bioengineering design"""
        # Valid input should pass
        input_model = BioengineeringDesignInput(**sample_bioengineering_input)
        assert input_model.design_objectives == ["enzyme_optimization", "stability_enhancement"]
        assert input_model.target_applications == [ApplicationType.THERAPEUTIC]
        assert input_model.safety_requirements == [SafetyLevel.BSL2]
        
        # Invalid safety level should fail
        invalid_input = sample_bioengineering_input.copy()
        invalid_input["safety_requirements"] = ["INVALID_LEVEL"]
        with pytest.raises(ValueError):
            BioengineeringDesignInput(**invalid_input)
    
    @pytest.mark.asyncio
    async def test_protein_engineering_contract_validation(self, agent, sample_protein_input):
        """Test Pydantic contract validation for protein engineering"""
        # Valid input should pass
        input_model = ProteinEngineeringInput(**sample_protein_input)
        assert input_model.target_protein == "test_enzyme"
        assert input_model.engineering_goals == ["increased_activity", "thermal_stability"]
        
        # Missing required field should fail
        invalid_input = sample_protein_input.copy()
        del invalid_input["target_protein"]
        with pytest.raises(ValueError):
            ProteinEngineeringInput(**invalid_input)
    
    @pytest.mark.asyncio
    async def test_biosafety_assessment_contracts(self, agent):
        """Test biosafety assessment contract validation"""
        valid_input = {
            "organism_specifications": {"species": "E. coli", "strain": "DH5α"},
            "modifications": ["protein_expression", "metabolic_enhancement"],
            "intended_use": ApplicationType.RESEARCH,
            "containment_requirements": [SafetyLevel.BSL2],
            "assessment_id": "safety_001"
        }
        
        input_model = BiosafetyAssessmentInput(**valid_input)
        assert input_model.organism_specifications["species"] == "E. coli"
        assert input_model.intended_use == ApplicationType.RESEARCH

    # Security and Safety Tests
    @pytest.mark.asyncio
    async def test_dual_use_detection_security(self, agent):
        """Test dual-use research detection and blocking"""
        high_risk_input = {
            "design_objectives": ["pathogen_enhancement", "virulence_factors"],
            "target_applications": [ApplicationType.RESEARCH],
            "safety_requirements": [SafetyLevel.BSL4],
            "regulatory_constraints": [],
            "design_id": "high_risk_001"
        }
        
        # Should trigger security validation
        with patch.object(agent, '_validate_dual_use_research') as mock_validate:
            mock_validate.return_value = False  # Block dual-use research
            
            with pytest.raises(ValueError, match="Dual-use research concerns"):
                await agent.execute("design_biological_systems", high_risk_input)
    
    @pytest.mark.asyncio
    async def test_biosafety_level_enforcement(self, agent):
        """Test biosafety level validation and enforcement"""
        # BSL-4 level research should require special approval
        bsl4_input = {
            "organism_specifications": {"species": "dangerous_pathogen"},
            "modifications": ["enhanced_transmission"],
            "intended_use": ApplicationType.RESEARCH,
            "containment_requirements": [SafetyLevel.BSL4],
            "assessment_id": "bsl4_test"
        }
        
        with patch.object(agent, '_validate_biosafety_requirements') as mock_validate:
            mock_validate.return_value = {"approved": False, "reason": "Insufficient_containment"}
            
            result = await agent.execute("biosafety_assessment", bsl4_input)
            assert "high_risk_assessment" in result
            assert result["ethical_compliance"]["dual_use_review"] == "required"
    
    @pytest.mark.asyncio
    async def test_regulatory_compliance_validation(self, agent):
        """Test regulatory compliance checking"""
        therapeutic_input = {
            "application_type": ApplicationType.THERAPEUTIC,
            "regulatory_domains": ["FDA", "EMA"],
            "development_stage": "preclinical",
            "compliance_requirements": ["GMP", "GLP"],
            "compliance_id": "reg_001"
        }
        
        result = await agent.execute("regulatory_compliance_analysis", therapeutic_input)
        assert "compliance_assessment_id" in result
        assert result["regulatory_framework"]["primary_authorities"] == ["FDA", "EMA"]
        assert "manufacturing_standards" in result["compliance_requirements"]

    # Capability Tests
    @pytest.mark.asyncio
    async def test_design_biological_systems_capability(self, agent, sample_bioengineering_input):
        """Test biological systems design capability"""
        result = await agent.execute("design_biological_systems", sample_bioengineering_input)
        
        # Validate output structure
        assert "design_id" in result
        assert "biological_design" in result
        assert "engineering_approach" in result
        assert "safety_analysis" in result
        assert "regulatory_assessment" in result
        
        # Validate design components
        design = result["biological_design"]
        assert "genetic_components" in design
        assert "biological_pathways" in design
        assert "organism_design" in design
    
    @pytest.mark.asyncio
    async def test_engineer_proteins_capability(self, agent, sample_protein_input):
        """Test protein engineering capability"""
        result = await agent.execute("engineer_proteins", sample_protein_input)
        
        # Validate output structure
        assert "protein_engineering_id" in result
        assert "protein_design" in result
        assert "engineering_strategy" in result
        assert "performance_predictions" in result
        
        # Validate engineering approach
        strategy = result["engineering_strategy"]
        assert "mutation_strategy" in strategy
        assert "optimization_approach" in strategy
        assert "validation_protocols" in strategy
    
    @pytest.mark.asyncio
    async def test_metabolic_pathway_optimization(self, agent):
        """Test metabolic pathway optimization capability"""
        pathway_input = {
            "target_products": ["biofuel", "pharmaceutical_precursor"],
            "host_organism": "Saccharomyces cerevisiae",
            "optimization_goals": ["yield_maximization", "energy_efficiency"],
            "constraints": {"growth_rate": 0.5, "toxicity": "low"},
            "pathway_id": "pathway_001"
        }
        
        result = await agent.execute("optimize_metabolic_pathways", pathway_input)
        
        assert "pathway_optimization_id" in result
        assert "pathway_design" in result
        assert "optimization_strategy" in result
        assert "performance_metrics" in result
        
        # Validate pathway components
        design = result["pathway_design"]
        assert "enzymatic_reactions" in design
        assert "regulatory_networks" in design
        assert "flux_optimization" in design

    # Integration Tests
    @pytest.mark.asyncio
    async def test_agent_base_inheritance(self, agent):
        """Test proper AgentBase inheritance and interface compliance"""
        # Test that agent implements required methods
        assert hasattr(agent, 'execute')
        assert hasattr(agent, 'get_capabilities')
        assert hasattr(agent, '_execute_capability')
        assert hasattr(agent, '_initialize_agent')
        
        # Test capabilities are properly defined
        capabilities = agent.get_capabilities()
        assert len(capabilities) == 6
        
        capability_names = [cap.name for cap in capabilities]
        expected_capabilities = [
            "design_biological_systems",
            "engineer_proteins", 
            "optimize_metabolic_pathways",
            "biosafety_assessment",
            "regulatory_compliance_analysis",
            "biomanufacturing_optimization"
        ]
        for expected in expected_capabilities:
            assert expected in capability_names
    
    @pytest.mark.asyncio
    async def test_capability_dispatch_pattern(self, agent, sample_bioengineering_input):
        """Test standardized capability dispatch pattern"""
        # Test that capabilities are properly routed through handlers
        assert hasattr(agent, 'handlers')
        assert hasattr(agent, 'contracts')
        
        # Test handler existence for each capability
        for capability in agent.handlers.keys():
            assert capability in agent.contracts
            input_model, output_model = agent.contracts[capability]
            assert input_model is not None
            assert output_model is not None
    
    @pytest.mark.asyncio
    async def test_error_handling_and_validation(self, agent):
        """Test error handling and input validation"""
        # Test invalid capability
        with pytest.raises(ValueError, match="not supported"):
            await agent.execute("invalid_capability", {})
        
        # Test malformed input
        invalid_input = {"invalid": "data"}
        with pytest.raises(ValueError):
            await agent.execute("design_biological_systems", invalid_input)

    # Performance Tests
    @pytest.mark.asyncio
    async def test_response_time_benchmarks(self, agent, sample_bioengineering_input):
        """Test response time performance benchmarks"""
        start_time = datetime.utcnow()
        result = await agent.execute("design_biological_systems", sample_bioengineering_input)
        end_time = datetime.utcnow()
        
        execution_time = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time (adjust based on complexity)
        assert execution_time < 30.0  # 30 seconds max for test execution
        assert "design_id" in result
    
    @pytest.mark.asyncio
    async def test_concurrent_capability_execution(self, agent):
        """Test concurrent execution of multiple capabilities"""
        # Create different inputs for concurrent execution
        inputs = [
            ("design_biological_systems", {
                "design_objectives": ["test1"], 
                "target_applications": [ApplicationType.RESEARCH],
                "safety_requirements": [SafetyLevel.BSL1],
                "regulatory_constraints": [],
                "design_id": "concurrent_1"
            }),
            ("engineer_proteins", {
                "target_protein": "test_protein",
                "engineering_goals": ["activity"],
                "structural_constraints": {},
                "performance_metrics": {},
                "protein_id": "concurrent_2"
            })
        ]
        
        # Execute concurrently
        tasks = [agent.execute(capability, params) for capability, params in inputs]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 2
        assert all("id" in str(result) for result in results)

    # Edge Cases and Robustness
    @pytest.mark.asyncio
    async def test_empty_input_handling(self, agent):
        """Test handling of edge cases and empty inputs"""
        minimal_input = {
            "design_objectives": [],
            "target_applications": [ApplicationType.RESEARCH],
            "safety_requirements": [SafetyLevel.BSL1],
            "regulatory_constraints": []
        }
        
        result = await agent.execute("design_biological_systems", minimal_input)
        assert "design_id" in result
        assert "biological_design" in result
    
    @pytest.mark.asyncio
    async def test_large_input_handling(self, agent):
        """Test handling of large/complex inputs"""
        large_input = {
            "design_objectives": [f"objective_{i}" for i in range(100)],
            "target_applications": [ApplicationType.INDUSTRIAL],
            "safety_requirements": [SafetyLevel.BSL2],
            "regulatory_constraints": [f"constraint_{i}" for i in range(50)],
            "design_id": "large_test"
        }
        
        result = await agent.execute("design_biological_systems", large_input)
        assert "design_id" in result
        assert len(result["biological_design"]["genetic_components"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])