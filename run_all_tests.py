#!/usr/bin/env python3
"""
Comprehensive Test Runner for Replit Manager
Runs all test suites with detailed reporting
"""

import os
import sys
import unittest
import time
from io import StringIO


def run_test_suite(test_module_name, description):
    """Run a specific test suite and return results"""
    print(f"\n{'='*80}")
    print(f"RUNNING {description.upper()}")
    print(f"{'='*80}")
    
    # Capture test output
    test_output = StringIO()
    
    # Load and run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_module_name)
    runner = unittest.TextTestRunner(
        stream=test_output,
        verbosity=2,
        buffer=True
    )
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print results
    output = test_output.getvalue()
    print(output)
    
    # Summary
    duration = end_time - start_time
    print(f"\n{description} Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Status: {'PASSED' if result.wasSuccessful() else 'FAILED'}")
    
    if result.failures:
        print(f"\n  FAILURES:")
        for test, traceback in result.failures:
            print(f"    - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n  ERRORS:")
        for test, traceback in result.errors:
            print(f"    - {test}: {traceback.split(':', 1)[-1].strip()}")
    
    return result


def main():
    """Run all test suites"""
    print("REPLIT MANAGER - COMPREHENSIVE TEST SUITE")
    print("Running all tests for deployment readiness...")
    
    # Set test environment
    os.environ['TESTING'] = 'true'
    
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    # Test suites to run
    test_suites = [
        ('tests.test_unit', 'Unit Tests - Individual Component Testing'),
        ('tests.test_integration', 'Integration Tests - Component Interaction Testing'),
        ('tests.test_system', 'System Tests - End-to-End System Testing'),
        ('tests.test_ux', 'UX Tests - User Experience and Interface Testing')
    ]
    
    results = []
    total_start_time = time.time()
    
    # Run each test suite
    for module_name, description in test_suites:
        try:
            result = run_test_suite(module_name, description)
            results.append((description, result))
        except Exception as e:
            print(f"\nERROR running {description}: {e}")
            # Create a mock result for error cases
            mock_result = unittest.TestResult()
            mock_result.errors.append((module_name, str(e)))
            results.append((description, mock_result))
    
    total_end_time = time.time()
    
    # Overall summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*80}")
    
    overall_success = True
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for description, result in results:
        success = result.wasSuccessful() if hasattr(result, 'wasSuccessful') else False
        tests_run = getattr(result, 'testsRun', 0)
        failures = len(getattr(result, 'failures', []))
        errors = len(getattr(result, 'errors', []))
        
        total_tests += tests_run
        total_failures += failures
        total_errors += errors
        
        if not success:
            overall_success = False
        
        status = "PASSED" if success else "FAILED"
        print(f"{description:<50} {status}")
        if tests_run > 0:
            print(f"  └─ {tests_run} tests, {failures} failures, {errors} errors")
    
    print(f"\nOverall Results:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Total Failures: {total_failures}")
    print(f"  Total Errors: {total_errors}")
    print(f"  Duration: {total_end_time - total_start_time:.2f} seconds")
    print(f"  Overall Status: {'PASSED' if overall_success else 'FAILED'}")
    
    # Deployment readiness assessment
    print(f"\n{'='*80}")
    print("DEPLOYMENT READINESS ASSESSMENT")
    print(f"{'='*80}")
    
    if overall_success and total_tests > 0:
        print("✅ DEPLOYMENT READY")
        print("   All tests passed. System is ready for production deployment.")
        print("\n✅ Quality Assurance Checklist:")
        print("   - Unit tests: All components working correctly")
        print("   - Integration tests: Services communicate properly")
        print("   - System tests: End-to-end functionality verified")
        print("   - UX tests: User interface and experience validated")
        
    elif total_tests == 0:
        print("⚠️  NO TESTS RUN")
        print("   Could not execute test suites. Check test environment.")
        
    else:
        print("❌ NOT READY FOR DEPLOYMENT")
        print("   Tests failed. Review and fix issues before deploying.")
        
        if total_failures > 0:
            print(f"   - {total_failures} test failures need to be resolved")
        if total_errors > 0:
            print(f"   - {total_errors} test errors need to be fixed")
    
    # Return appropriate exit code
    sys.exit(0 if overall_success else 1)


if __name__ == '__main__':
    main()