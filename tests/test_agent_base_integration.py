"""
Integration tests for AgentBase interface and cross-agent functionality
Tests architectural consistency, security framework, and performance
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import Dict, Any, List

from services.agent_base import AgentBase, AgentCapability, AgentStatus, SecurityLevel
from services.synthetic_biology_engineering_agent import SyntheticBiologyEngineeringAgent
from services.quantum_computing_optimization_agent import QuantumComputingOptimizationAgent
from services.consciousness_ai_research_agent import ConsciousnessAIResearchAgent


class TestAgentBaseIntegration:
    """Integration tests for AgentBase interface and all agents"""
    
    @pytest.fixture
    def all_agents(self):
        """Create instances of all Tier 2/3 agents"""
        return [
            SyntheticBiologyEngineeringAgent(),
            QuantumComputingOptimizationAgent(),
            ConsciousnessAIResearchAgent()
        ]
    
    @pytest.fixture
    def agent_test_cases(self):
        """Test cases for each agent"""
        return {
            "synthetic_biology": {
                "capability": "design_biological_systems",
                "input": {
                    "design_objectives": ["test_objective"],
                    "target_applications": ["research"],
                    "safety_requirements": ["BSL1"],
                    "regulatory_constraints": []
                }
            },
            "quantum_computing": {
                "capability": "optimize_quantum_algorithms",
                "input": {
                    "problem_type": "portfolio_optimization",
                    "quantum_advantage_goals": ["exponential_speedup"],
                    "hardware_constraints": {"qubit_count": 20},
                    "performance_targets": {}
                }
            },
            "consciousness_ai": {
                "capability": "consciousness_assessment",
                "input": {
                    "assessment_targets": ["consciousness_level"],
                    "measurement_methods": ["phi_measure"],
                    "validation_requirements": {},
                    "ethical_constraints": []
                }
            }
        }

    # Architecture Consistency Tests
    @pytest.mark.asyncio
    async def test_agent_base_interface_compliance(self, all_agents):
        """Test that all agents properly implement AgentBase interface"""
        for agent in all_agents:
            # Test inheritance
            assert isinstance(agent, AgentBase)
            
            # Test required methods exist
            assert hasattr(agent, 'execute')
            assert hasattr(agent, 'get_capabilities')
            assert hasattr(agent, '_execute_capability')
            assert hasattr(agent, '_initialize_agent')
            
            # Test required attributes
            assert hasattr(agent, 'agent_id')
            assert hasattr(agent, 'version')
            assert hasattr(agent, 'status')
            assert hasattr(agent, 'capabilities')
            
            # Test initialization
            assert agent.status == AgentStatus.READY
            assert len(agent.capabilities) > 0
    
    @pytest.mark.asyncio
    async def test_capability_consistency(self, all_agents):
        """Test capability definition consistency across agents"""
        for agent in all_agents:
            capabilities = agent.get_capabilities()
            
            # Test that all capabilities have required fields
            for capability in capabilities:
                assert hasattr(capability, 'name')
                assert hasattr(capability, 'description')
                assert hasattr(capability, 'input_types')
                assert hasattr(capability, 'output_types')
                assert hasattr(capability, 'processing_time')
                assert hasattr(capability, 'resource_requirements')
                
                # Test field types
                assert isinstance(capability.name, str)
                assert isinstance(capability.description, str)
                assert isinstance(capability.input_types, list)
                assert isinstance(capability.output_types, list)
                assert isinstance(capability.processing_time, str)
                assert isinstance(capability.resource_requirements, dict)
    
    @pytest.mark.asyncio
    async def test_handler_contract_mapping(self, all_agents):
        """Test that handlers and contracts are properly mapped"""
        for agent in all_agents:
            if hasattr(agent, 'handlers') and hasattr(agent, 'contracts'):
                # Test that all handlers have corresponding contracts
                for capability_name in agent.handlers.keys():
                    assert capability_name in agent.contracts
                    
                    input_model, output_model = agent.contracts[capability_name]
                    assert input_model is not None
                    assert output_model is not None
                
                # Test that all contracts have corresponding handlers
                for capability_name in agent.contracts.keys():
                    assert capability_name in agent.handlers

    # Security Framework Tests
    @pytest.mark.asyncio
    async def test_security_validation_framework(self, all_agents):
        """Test security validation framework across all agents"""
        for agent in all_agents:
            # Test security initialization
            assert hasattr(agent, 'security_config')
            assert agent.security_config is not None
            
            # Test domain-specific security policies
            if agent.agent_id == "synthetic_biology_engineering":
                assert "biosafety_levels" in agent.security_config
                assert "dual_use_research" in agent.security_config
            elif agent.agent_id == "quantum_computing_optimization":
                assert "export_controls" in agent.security_config
                assert "cryptographic_restrictions" in agent.security_config
            elif agent.agent_id == "consciousness_ai_research":
                assert "consciousness_ethics" in agent.security_config
                assert "agi_safety_protocols" in agent.security_config
    
    @pytest.mark.asyncio
    async def test_cross_agent_security_consistency(self, all_agents):
        """Test security framework consistency across agents"""
        security_levels = set()
        
        for agent in all_agents:
            if hasattr(agent, 'security_config'):
                config = agent.security_config
                if "security_levels" in config:
                    security_levels.update(config["security_levels"])
        
        # All agents should recognize common security levels
        common_levels = {"low", "medium", "high", "critical"}
        assert common_levels.issubset(security_levels)
    
    @pytest.mark.asyncio
    async def test_audit_logging_consistency(self, all_agents):
        """Test that all agents implement consistent audit logging"""
        for agent in all_agents:
            # Test that agents have audit logging capabilities
            assert hasattr(agent, '_log_operation')
            
            # Test logging structure
            with patch('logging.Logger.info') as mock_log:
                await agent._log_operation("test_operation", {"test": "data"})
                mock_log.assert_called()

    # Execution Pattern Tests
    @pytest.mark.asyncio
    async def test_standardized_execution_pattern(self, all_agents, agent_test_cases):
        """Test that all agents follow standardized execution pattern"""
        agent_mapping = {
            "synthetic_biology_engineering": agent_test_cases["synthetic_biology"],
            "quantum_computing_optimization": agent_test_cases["quantum_computing"],
            "consciousness_ai_research": agent_test_cases["consciousness_ai"]
        }
        
        for agent in all_agents:
            if agent.agent_id in agent_mapping:
                test_case = agent_mapping[agent.agent_id]
                
                # Test execution
                result = await agent.execute(test_case["capability"], test_case["input"])
                
                # Test standard result structure
                assert isinstance(result, dict)
                assert len(result) > 0
                
                # Test that result contains an ID field
                id_fields = [key for key in result.keys() if key.endswith("_id")]
                assert len(id_fields) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_consistency(self, all_agents):
        """Test consistent error handling across agents"""
        for agent in all_agents:
            # Test invalid capability
            with pytest.raises(ValueError, match="not supported"):
                await agent.execute("invalid_capability", {})
            
            # Test capability validation
            capabilities = agent.get_capabilities()
            if capabilities:
                # Test with empty input to trigger validation
                with pytest.raises((ValueError, TypeError)):
                    await agent.execute(capabilities[0].name, {})

    # Performance and Scalability Tests
    @pytest.mark.asyncio
    async def test_concurrent_multi_agent_execution(self, all_agents, agent_test_cases):
        """Test concurrent execution across multiple agents"""
        agent_mapping = {
            "synthetic_biology_engineering": agent_test_cases["synthetic_biology"],
            "quantum_computing_optimization": agent_test_cases["quantum_computing"],
            "consciousness_ai_research": agent_test_cases["consciousness_ai"]
        }
        
        tasks = []
        for agent in all_agents:
            if agent.agent_id in agent_mapping:
                test_case = agent_mapping[agent.agent_id]
                task = agent.execute(test_case["capability"], test_case["input"])
                tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check that all executions completed
            assert len(results) == len(tasks)
            
            # Check for successful results (not exceptions)
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) > 0
    
    @pytest.mark.asyncio
    async def test_agent_resource_management(self, all_agents):
        """Test resource management and status tracking"""
        for agent in all_agents:
            # Test initial status
            assert agent.status == AgentStatus.READY
            
            # Test status during mock execution
            with patch.object(agent, '_execute_capability') as mock_execute:
                mock_execute.return_value = {"test": "result"}
                
                # Start execution
                if agent.get_capabilities():
                    capability = agent.get_capabilities()[0]
                    task = agent.execute(capability.name, {})
                    
                    # Should complete successfully
                    result = await task
                    assert isinstance(result, dict)
                    assert agent.status == AgentStatus.READY
    
    @pytest.mark.asyncio
    async def test_memory_and_state_isolation(self, all_agents):
        """Test that agents maintain proper state isolation"""
        for agent in all_agents:
            # Test that agents don't share state
            original_id = agent.agent_id
            
            # Modify one agent's state
            agent._test_field = "test_value"
            
            # Check other agents don't have this field
            for other_agent in all_agents:
                if other_agent != agent:
                    assert not hasattr(other_agent, '_test_field')
            
            # Cleanup
            delattr(agent, '_test_field')
            assert agent.agent_id == original_id

    # Integration Workflow Tests
    @pytest.mark.asyncio
    async def test_multi_agent_workflow_simulation(self, all_agents):
        """Test simulated multi-agent workflow collaboration"""
        # Simulate a complex workflow involving multiple agents
        workflow_steps = []
        
        # Step 1: Consciousness AI assesses requirements
        consciousness_agent = next(a for a in all_agents if a.agent_id == "consciousness_ai_research")
        assessment_result = await consciousness_agent.execute("consciousness_assessment", {
            "assessment_targets": ["system_requirements"],
            "measurement_methods": ["phi_measure"],
            "validation_requirements": {},
            "ethical_constraints": ["safety_first"]
        })
        workflow_steps.append(("consciousness_assessment", assessment_result))
        
        # Step 2: Quantum agent optimizes computational approach
        quantum_agent = next(a for a in all_agents if a.agent_id == "quantum_computing_optimization")
        quantum_result = await quantum_agent.execute("optimize_quantum_algorithms", {
            "problem_type": "optimization_generic",
            "quantum_advantage_goals": ["computational_efficiency"],
            "hardware_constraints": {"qubit_count": 30},
            "performance_targets": {"runtime": 300}
        })
        workflow_steps.append(("quantum_optimization", quantum_result))
        
        # Step 3: Synthetic biology agent designs implementation
        bio_agent = next(a for a in all_agents if a.agent_id == "synthetic_biology_engineering")
        bio_result = await bio_agent.execute("design_biological_systems", {
            "design_objectives": ["computational_implementation"],
            "target_applications": ["research"],
            "safety_requirements": ["BSL1"],
            "regulatory_constraints": []
        })
        workflow_steps.append(("biological_design", bio_result))
        
        # Validate workflow completion
        assert len(workflow_steps) == 3
        for step_name, result in workflow_steps:
            assert isinstance(result, dict)
            assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_agent_ecosystem_health_check(self, all_agents):
        """Test overall ecosystem health and readiness"""
        health_metrics = {}
        
        for agent in all_agents:
            agent_health = {
                "status": agent.status.value,
                "capabilities_count": len(agent.get_capabilities()),
                "has_handlers": hasattr(agent, 'handlers'),
                "has_contracts": hasattr(agent, 'contracts'),
                "has_security_config": hasattr(agent, 'security_config')
            }
            health_metrics[agent.agent_id] = agent_health
        
        # Validate ecosystem health
        for agent_id, health in health_metrics.items():
            assert health["status"] == "ready"
            assert health["capabilities_count"] > 0
            assert health["has_handlers"] is True
            assert health["has_contracts"] is True
            assert health["has_security_config"] is True
        
        # Test ecosystem readiness score
        total_capabilities = sum(h["capabilities_count"] for h in health_metrics.values())
        assert total_capabilities >= 15  # Minimum expected capabilities across all agents


if __name__ == "__main__":
    pytest.main([__file__, "-v"])