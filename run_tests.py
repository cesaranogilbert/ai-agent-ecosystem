#!/usr/bin/env python3
"""
Comprehensive test runner for AI Agent Ecosystem
Executes all test suites with proper configuration and reporting
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and return success status"""
    print(f"\\n{'='*60}")
    print(f"Running: {description or cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"Errors/Warnings:\\n{result.stderr}")
    
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Run AI Agent Ecosystem Tests")
    parser.add_argument("--suite", choices=["unit", "integration", "performance", "security", "all"], 
                       default="all", help="Test suite to run")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Install test dependencies
    print("Installing test dependencies...")
    if not run_command("pip install -r requirements-test.txt", "Installing test dependencies"):
        print("❌ Failed to install test dependencies")
        return 1
    
    # Base pytest command
    pytest_cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        pytest_cmd.append("-v")
    
    if args.coverage:
        pytest_cmd.extend(["--cov=services", "--cov-report=html", "--cov-report=term"])
    
    if args.parallel:
        pytest_cmd.extend(["-n", "auto"])
    
    if args.fast:
        pytest_cmd.extend(["-m", "not slow"])
    
    # Test suite selection
    test_commands = []
    
    if args.suite == "unit" or args.suite == "all":
        cmd = " ".join(pytest_cmd + ["tests/test_*_agent.py", "-m", "not integration and not performance"])
        test_commands.append((cmd, "Unit Tests"))
    
    if args.suite == "integration" or args.suite == "all":
        cmd = " ".join(pytest_cmd + ["tests/test_agent_base_integration.py", "-m", "integration"])
        test_commands.append((cmd, "Integration Tests"))
    
    if args.suite == "performance" or args.suite == "all":
        cmd = " ".join(pytest_cmd + ["tests/test_performance_benchmarks.py", "-m", "performance"])
        test_commands.append((cmd, "Performance Tests"))
    
    if args.suite == "security" or args.suite == "all":
        cmd = " ".join(pytest_cmd + ["-m", "security"])
        test_commands.append((cmd, "Security Tests"))
    
    # Run all selected test suites
    all_passed = True
    results = {}
    
    for cmd, description in test_commands:
        success = run_command(cmd, description)
        results[description] = "✅ PASSED" if success else "❌ FAILED"
        if not success:
            all_passed = False
    
    # Run additional quality checks
    print("\\n" + "="*60)
    print("Code Quality Checks")
    print("="*60)
    
    quality_checks = [
        ("python -m black --check services/ tests/", "Black Code Formatting"),
        ("python -m isort --check-only services/ tests/", "Import Sorting"),
        ("python -m flake8 services/ tests/", "Flake8 Linting"),
        ("python -m mypy services/", "Type Checking"),
        ("python -m bandit -r services/", "Security Analysis")
    ]
    
    for cmd, description in quality_checks:
        success = run_command(cmd, description)
        results[description] = "✅ PASSED" if success else "❌ FAILED"
        if not success and args.suite == "all":
            all_passed = False
    
    # Final report
    print("\\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    for test_name, status in results.items():
        print(f"{status} {test_name}")
    
    if all_passed:
        print("\\n🎉 All tests passed successfully!")
        return 0
    else:
        print("\\n⚠️  Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())