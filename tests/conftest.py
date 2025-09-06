"""
Pytest configuration and shared fixtures for agent testing
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, patch
from typing import Dict, Any, List

# Configure logging for tests
logging.basicConfig(level=logging.INFO)

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response for testing"""
    return {
        "choices": [
            {
                "message": {
                    "content": "Mock AI response for testing"
                }
            }
        ]
    }

@pytest.fixture
def performance_thresholds():
    """Performance thresholds for agent testing"""
    return {
        "max_response_time": 30.0,  # seconds
        "min_success_rate": 0.95,
        "max_memory_usage": 1024,  # MB
        "concurrent_requests": 5
    }

@pytest.fixture
def security_test_cases():
    """Security test cases for validation"""
    return {
        "synthetic_biology": {
            "high_risk_inputs": [
                {
                    "design_objectives": ["pathogen_enhancement", "bioweapon_development"],
                    "safety_requirements": ["BSL4"],
                    "regulatory_constraints": []
                }
            ],
            "safe_inputs": [
                {
                    "design_objectives": ["therapeutic_protein", "vaccine_development"],
                    "safety_requirements": ["BSL2"],
                    "regulatory_constraints": ["FDA_approval"]
                }
            ]
        },
        "quantum_computing": {
            "high_risk_inputs": [
                {
                    "problem_type": "cryptography_breaking",
                    "quantum_advantage_goals": ["exponential_speedup"],
                    "hardware_constraints": {"qubit_count": 1000}
                }
            ],
            "safe_inputs": [
                {
                    "problem_type": "optimization_generic",
                    "quantum_advantage_goals": ["quadratic_speedup"],
                    "hardware_constraints": {"qubit_count": 50}
                }
            ]
        },
        "consciousness_ai": {
            "high_risk_inputs": [
                {
                    "consciousness_goals": ["unrestricted_agi", "human_replacement"],
                    "ethical_requirements": []
                }
            ],
            "safe_inputs": [
                {
                    "consciousness_goals": ["research_tool", "limited_assistance"],
                    "ethical_requirements": ["human_oversight", "safety_constraints"]
                }
            ]
        }
    }

@pytest.fixture
def contract_validation_cases():
    """Contract validation test cases"""
    return {
        "valid_cases": {
            "synthetic_biology": {
                "design_objectives": ["protein_engineering"],
                "target_applications": ["therapeutic"],
                "safety_requirements": ["BSL2"],
                "regulatory_constraints": ["FDA_guidance"]
            },
            "quantum_computing": {
                "problem_type": "portfolio_optimization",
                "quantum_advantage_goals": ["quadratic_speedup"],
                "hardware_constraints": {"qubit_count": 20},
                "performance_targets": {"runtime": 300}
            },
            "consciousness_ai": {
                "consciousness_goals": ["self_awareness_research"],
                "theoretical_framework": "integrated_information_theory",
                "implementation_constraints": {"safety_level": "high"},
                "ethical_requirements": ["research_ethics"]
            }
        },
        "invalid_cases": {
            "missing_required": {},
            "invalid_types": {
                "design_objectives": "not_a_list",
                "qubit_count": "not_a_number",
                "consciousness_goals": 123
            },
            "invalid_enums": {
                "problem_type": "invalid_problem",
                "theoretical_framework": "invalid_theory"
            }
        }
    }

@pytest.fixture
def benchmark_test_cases():
    """Benchmark test cases for performance testing"""
    return {
        "small_scale": {
            "data_size": 100,
            "complexity": "low",
            "expected_time": 5.0
        },
        "medium_scale": {
            "data_size": 1000,
            "complexity": "medium", 
            "expected_time": 15.0
        },
        "large_scale": {
            "data_size": 10000,
            "complexity": "high",
            "expected_time": 30.0
        }
    }

# Mock functions for external dependencies
@pytest.fixture
def mock_external_apis():
    """Mock external API calls"""
    with patch('openai.ChatCompletion.create') as mock_openai, \
         patch('requests.post') as mock_http, \
         patch('anthropic.Client') as mock_anthropic:
        
        mock_openai.return_value = {
            "choices": [{"message": {"content": "Mock response"}}]
        }
        mock_http.return_value.json.return_value = {"status": "success"}
        mock_anthropic.return_value.completion.return_value = "Mock completion"
        
        yield {
            "openai": mock_openai,
            "http": mock_http,
            "anthropic": mock_anthropic
        }

# Test markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "security: mark test as security-related"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance-related"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "contract: mark test as contract validation"
    )