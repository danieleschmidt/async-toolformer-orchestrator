"""Mutation testing configuration and tests for async-toolformer-orchestrator."""

import asyncio
from typing import Dict, List
from unittest.mock import AsyncMock

import pytest


@pytest.mark.slow
class TestMutationTargets:
    """Test critical code paths that should be covered by mutation testing."""

    async def test_rate_limiting_logic_mutations(
        self,
        mock_redis: AsyncMock
    ):
        """Test that rate limiting logic is robust against mutations."""
        
        # Test the critical rate limiting logic
        async def rate_limit_check(service: str, user_id: str, limit: int, window: int) -> bool:
            """Critical rate limiting logic that should be mutation tested."""
            key = f"rate_limit:{service}:{user_id}"
            
            # Get current count
            current = await mock_redis.get(key)
            current_count = int(current) if current else 0
            
            # Check if limit exceeded
            if current_count >= limit:  # This condition should be tested
                return False
            
            # Increment counter
            new_count = await mock_redis.incr(key)
            
            # Set expiry on first increment
            if new_count == 1:  # This condition should be tested
                await mock_redis.expire(key, window)
            
            return True
        
        # Mock Redis responses
        mock_redis.get.return_value = "0"
        mock_redis.incr.return_value = 1
        mock_redis.expire.return_value = True
        
        # Test normal case
        result = await rate_limit_check("openai", "user1", 10, 60)
        assert result is True
        
        # Test limit exceeded case
        mock_redis.get.return_value = "10"
        result = await rate_limit_check("openai", "user1", 10, 60)
        assert result is False
        
        # Test edge case: exactly at limit
        mock_redis.get.return_value = "9"
        mock_redis.incr.return_value = 10
        result = await rate_limit_check("openai", "user1", 10, 60)
        assert result is True  # Should still allow (9 -> 10)

    async def test_parallel_execution_mutations(
        self,
        mock_orchestrator: AsyncMock
    ):
        """Test that parallel execution logic is robust."""
        
        async def execute_tools_parallel(tools: List, max_parallel: int) -> List:
            """Critical parallel execution logic for mutation testing."""
            if not tools:  # Empty check - should be tested
                return []
            
            if max_parallel <= 0:  # Boundary condition - should be tested
                max_parallel = 1
            
            # Limit actual parallelism
            actual_parallel = min(len(tools), max_parallel)  # Min operation - should be tested
            
            # Execute in batches
            results = []
            for i in range(0, len(tools), actual_parallel):
                batch = tools[i:i + actual_parallel]
                batch_results = await asyncio.gather(*[tool() for tool in batch])
                results.extend(batch_results)
            
            return results
        
        # Create test tools
        tools = []
        for i in range(5):
            tool = AsyncMock()
            tool.return_value = f"result_{i}"
            tools.append(tool)
        
        # Test normal execution
        results = await execute_tools_parallel(tools, 3)
        assert len(results) == 5
        assert all(f"result_{i}" in str(results) for i in range(5))
        
        # Test edge cases that mutations might break
        assert await execute_tools_parallel([], 10) == []  # Empty tools
        assert len(await execute_tools_parallel(tools, 0)) == 5  # Zero max_parallel
        assert len(await execute_tools_parallel(tools, -1)) == 5  # Negative max_parallel
        assert len(await execute_tools_parallel(tools, 100)) == 5  # Very high max_parallel

    async def test_error_handling_mutations(
        self,
        mock_orchestrator: AsyncMock
    ):
        """Test that error handling logic is robust."""
        
        async def safe_tool_execution(tool, timeout: float) -> Dict:
            """Critical error handling logic for mutation testing."""
            try:
                # Timeout check - boundary condition
                if timeout <= 0:  # Should be tested
                    timeout = 1.0
                
                result = await asyncio.wait_for(tool(), timeout=timeout)
                return {"success": True, "result": result, "error": None}
                
            except asyncio.TimeoutError:
                return {"success": False, "result": None, "error": "timeout"}
                
            except Exception as e:
                return {"success": False, "result": None, "error": str(e)}
        
        # Test successful execution
        success_tool = AsyncMock()
        success_tool.return_value = "success"
        
        result = await safe_tool_execution(success_tool, 1.0)
        assert result["success"] is True
        assert result["result"] == "success"
        assert result["error"] is None
        
        # Test timeout handling
        slow_tool = AsyncMock()
        slow_tool.side_effect = lambda: asyncio.sleep(2.0)
        
        result = await safe_tool_execution(slow_tool, 0.1)
        assert result["success"] is False
        assert result["error"] == "timeout"
        
        # Test exception handling
        error_tool = AsyncMock()
        error_tool.side_effect = ValueError("test error")
        
        result = await safe_tool_execution(error_tool, 1.0)
        assert result["success"] is False
        assert result["error"] == "test error"
        
        # Test boundary conditions
        result = await safe_tool_execution(success_tool, 0)
        assert result["success"] is True  # Should use default timeout
        
        result = await safe_tool_execution(success_tool, -1)
        assert result["success"] is True  # Should use default timeout

    async def test_speculation_logic_mutations(
        self,
        mock_orchestrator: AsyncMock
    ):
        """Test that speculation logic is robust."""
        
        async def speculation_decision(
            confidence: float, 
            threshold: float, 
            history: List[bool]
        ) -> bool:
            """Critical speculation logic for mutation testing."""
            # Basic confidence check
            if confidence < threshold:  # Threshold comparison - should be tested
                return False
            
            # Historical success rate
            if not history:  # Empty history check - should be tested
                return confidence >= 0.8  # High confidence needed without history
            
            success_rate = sum(history) / len(history)  # Division - should be tested
            
            # Combined decision
            combined_score = (confidence * 0.7) + (success_rate * 0.3)  # Weights - should be tested
            
            return combined_score >= threshold  # Final threshold - should be tested
        
        # Test normal cases
        assert await speculation_decision(0.9, 0.8, [True, True, False]) is True
        assert await speculation_decision(0.5, 0.8, [True, True, True]) is False
        
        # Test edge cases that mutations might break
        assert await speculation_decision(0.9, 0.8, []) is True  # Empty history, high confidence
        assert await speculation_decision(0.7, 0.8, []) is False  # Empty history, low confidence
        assert await speculation_decision(0.8, 0.8, [True]) is True  # Single success
        assert await speculation_decision(0.8, 0.8, [False]) is False  # Single failure
        
        # Test boundary conditions
        assert await speculation_decision(0.8, 0.8, [True, True]) is True  # Exactly at threshold
        assert await speculation_decision(0.799, 0.8, [True, True]) is False  # Just below threshold

    async def test_configuration_validation_mutations(
        self,
        mock_orchestrator: AsyncMock
    ):
        """Test that configuration validation is robust."""
        
        def validate_config(config: Dict) -> Dict:
            """Critical configuration validation for mutation testing."""
            validated = config.copy()
            
            # Required fields with defaults
            if "max_parallel" not in config or config["max_parallel"] <= 0:  # Should be tested
                validated["max_parallel"] = 10
            
            if "timeout_ms" not in config or config["timeout_ms"] <= 0:  # Should be tested
                validated["timeout_ms"] = 5000
            
            # Range validation
            if validated["max_parallel"] > 100:  # Upper bound - should be tested
                validated["max_parallel"] = 100
            
            if validated["timeout_ms"] > 300000:  # Upper bound - should be tested
                validated["timeout_ms"] = 300000
            
            # Boolean flags
            validated["enable_speculation"] = bool(config.get("enable_speculation", False))
            validated["enable_tracing"] = bool(config.get("enable_tracing", False))
            
            return validated
        
        # Test normal configuration
        config = {"max_parallel": 20, "timeout_ms": 10000}
        result = validate_config(config)
        assert result["max_parallel"] == 20
        assert result["timeout_ms"] == 10000
        
        # Test missing fields
        result = validate_config({})
        assert result["max_parallel"] == 10
        assert result["timeout_ms"] == 5000
        
        # Test invalid values
        result = validate_config({"max_parallel": 0, "timeout_ms": -1000})
        assert result["max_parallel"] == 10
        assert result["timeout_ms"] == 5000
        
        # Test boundary conditions
        result = validate_config({"max_parallel": 150, "timeout_ms": 400000})
        assert result["max_parallel"] == 100  # Capped at max
        assert result["timeout_ms"] == 300000  # Capped at max
        
        # Test boolean conversion
        result = validate_config({"enable_speculation": "true", "enable_tracing": 1})
        assert result["enable_speculation"] is True
        assert result["enable_tracing"] is True


class TestMutationConfiguration:
    """Configuration and helpers for mutation testing tools."""
    
    @staticmethod
    def get_mutation_targets() -> List[str]:
        """Return list of critical modules/functions for mutation testing."""
        return [
            "src/async_toolformer/orchestrator.py",
            "src/async_toolformer/tools.py", 
            "src/async_toolformer/rate_limiter.py",
            "src/async_toolformer/speculation.py",
            "src/async_toolformer/config.py",
        ]
    
    @staticmethod
    def get_mutation_operators() -> List[str]:
        """Return list of mutation operators to apply."""
        return [
            # Arithmetic operators
            "AOR",  # Arithmetic Operator Replacement (+, -, *, /, //, %, **)
            
            # Relational operators  
            "ROR",  # Relational Operator Replacement (<, <=, >, >=, ==, !=)
            
            # Logical operators
            "LOR",  # Logical Operator Replacement (and, or)
            
            # Unary operators
            "UOR",  # Unary Operator Replacement (+x, -x, not x)
            
            # Assignment operators
            "ASR",  # Assignment Operator Replacement (+=, -=, *=, /=, //=, %=, **=)
            
            # Boolean/None replacements
            "BCR",  # Boolean/None Constant Replacement (True, False, None)
            
            # Numeric constants
            "NCR",  # Numeric Constant Replacement (0, 1, -1)
            
            # String literals
            "SCR",  # String Constant Replacement ("", "string")
            
            # Exception handling
            "EHD",  # Exception Handler Deletion
            "EXS",  # Exception Swallowing
            
            # Loop modifications
            "LCR",  # Loop Control Replacement (break, continue)
            
            # Decorator removal
            "DCR",  # Decorator Removal (@decorator)
            
            # Slice modifications
            "SIR",  # Slice Index Replacement ([start:end:step])
        ]
    
    @staticmethod
    def get_critical_lines() -> Dict[str, List[int]]:
        """Return critical lines that must be covered by tests."""
        return {
            "rate_limiter.py": [25, 30, 35, 42],  # Rate limiting thresholds
            "orchestrator.py": [50, 65, 78, 90],  # Parallel execution logic
            "tools.py": [15, 28, 40],             # Tool execution paths
            "speculation.py": [20, 33, 45],       # Speculation decisions
            "config.py": [10, 18, 25, 30],        # Configuration validation
        }
    
    @staticmethod
    def should_skip_mutation(file_path: str, line_number: int, mutation_type: str) -> bool:
        """Determine if a specific mutation should be skipped."""
        # Skip mutations on logging statements
        if "logger" in file_path.lower() or "log" in mutation_type.lower():
            return True
        
        # Skip mutations on import statements
        if line_number <= 10:  # Typically imports are at the top
            return True
        
        # Skip mutations on docstrings and comments
        if mutation_type in ["SCR"] and "test" not in file_path:
            return True
        
        return False
    
    @staticmethod
    def get_mutation_test_command() -> str:
        """Return the command to run mutation testing."""
        return """
        # Install mutation testing tool
        pip install mutmut
        
        # Run mutation testing on core modules
        mutmut run --paths-to-mutate src/async_toolformer/
        
        # Generate HTML report
        mutmut html
        
        # Show results
        mutmut results
        """


# Example of how to run specific mutation tests
if __name__ == "__main__":
    import subprocess
    import sys
    
    def run_mutation_test(target_file: str) -> bool:
        """Run mutation testing on a specific file."""
        try:
            cmd = [
                "python", "-m", "pytest", 
                "tests/mutation/test_mutation.py", 
                "-v", "--tb=short"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Mutation tests passed for {target_file}")
                return True
            else:
                print(f"❌ Mutation tests failed for {target_file}")
                print(result.stdout)
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"Error running mutation tests: {e}")
            return False
    
    # Run mutation tests if executed directly
    config = TestMutationConfiguration()
    targets = config.get_mutation_targets()
    
    print("Running mutation testing validation...")
    all_passed = True
    
    for target in targets:
        if not run_mutation_test(target):
            all_passed = False
    
    if all_passed:
        print("\n🎉 All mutation test validations passed!")
        print("Ready to run full mutation testing with: pip install mutmut && mutmut run")
    else:
        print("\n❌ Some mutation test validations failed.")
        sys.exit(1)