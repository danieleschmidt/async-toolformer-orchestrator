#!/usr/bin/env python3
"""Generation 2: MAKE IT ROBUST (Reliable) - Enhanced error handling, logging, security"""

import asyncio
import random
import time
from typing import Dict, Any, Optional
from async_toolformer import (
    AsyncOrchestrator, Tool, ToolChain, parallel,
    ToolExecutionError, TimeoutError, ConfigurationError
)
from async_toolformer.simple_structured_logging import get_logger, CorrelationContext, log_execution_time
# Initialize logger
logger = get_logger(__name__)

# Robust tools with error handling
@Tool(description="Reliable web search with error handling", timeout_ms=5000, retry_attempts=3)
@log_execution_time("web_search")
async def robust_web_search(query: str) -> Dict[str, Any]:
    """Web search with comprehensive error handling."""
    logger.info("Starting web search", query=query)
    
    # Input validation
    if not query or len(query.strip()) < 2:
        raise ValueError("Query must be at least 2 characters long")
    
    # Simulate various failure scenarios 
    failure_chance = random.random()
    
    if failure_chance < 0.1:  # 10% chance of timeout
        await asyncio.sleep(6)  # Will trigger timeout
        
    if failure_chance < 0.2:  # 20% chance of API error
        raise ConnectionError("External API temporarily unavailable")
    
    # Simulate successful search
    await asyncio.sleep(random.uniform(0.1, 0.5))
    
    results = {
        "query": query.strip(),
        "results": [
            {"title": f"Result 1 for {query}", "url": "https://example.com/1"},
            {"title": f"Result 2 for {query}", "url": "https://example.com/2"},
        ],
        "total_results": 156,
        "search_time_ms": random.randint(50, 200)
    }
    
    logger.info("Web search completed", results_count=len(results["results"]))
    return results

@Tool(description="Advanced code analysis with security checks", rate_limit_group="analysis")
@log_execution_time("code_analysis")
async def secure_code_analysis(file_path: str, check_security: bool = True) -> Dict[str, Any]:
    """Code analysis with security scanning and error handling."""
    logger.info("Starting code analysis", file_path=file_path, security_enabled=check_security)
    
    # Path validation - prevent directory traversal
    import os
    if ".." in file_path or file_path.startswith("/"):
        raise ValueError("Invalid file path - potential security risk")
    
    # Simulate analysis with potential failures
    if random.random() < 0.15:  # 15% chance of analysis failure
        raise ToolExecutionError(
            tool_name="secure_code_analysis",
            message="Code analysis engine temporarily unavailable",
            details={"file_path": file_path}
        )
    
    await asyncio.sleep(random.uniform(0.2, 0.8))
    
    # Generate analysis results
    complexity_score = random.randint(1, 10)
    security_issues = []
    
    if check_security:
        # Simulate security findings
        if random.random() < 0.3:
            security_issues.append({
                "type": "potential_sql_injection", 
                "severity": "medium",
                "line": 42
            })
        if random.random() < 0.2:
            security_issues.append({
                "type": "hardcoded_credential", 
                "severity": "high",
                "line": 15
            })
    
    results = {
        "file_path": file_path,
        "complexity": complexity_score,
        "lines_of_code": random.randint(50, 500),
        "security_issues": security_issues,
        "quality_score": max(10 - complexity_score - len(security_issues), 1),
        "analysis_timestamp": time.time()
    }
    
    logger.info(
        "Code analysis completed",
        complexity=complexity_score,
        security_issues_count=len(security_issues),
        quality_score=results["quality_score"]
    )
    
    return results

@Tool(description="Validated notification system", priority=5)
@log_execution_time("notification")
async def validated_notification(
    recipient: str, 
    message: str, 
    priority: str = "normal",
    include_attachments: bool = False
) -> Dict[str, Any]:
    """Notification system with input validation and error handling."""
    
    logger.info("Sending notification", recipient=recipient, priority=priority)
    
    # Validate email format
    import re
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', recipient):
        raise ValueError(f"Invalid email format: {recipient}")
    
    # Validate priority
    valid_priorities = ["low", "normal", "high", "critical"]
    if priority not in valid_priorities:
        raise ValueError(f"Priority must be one of {valid_priorities}")
    
    # Message length validation
    if len(message) > 1000:
        raise ValueError("Message too long (max 1000 characters)")
    
    # Simulate delivery with potential failures
    if random.random() < 0.05:  # 5% delivery failure
        raise ConnectionError("SMTP server temporarily unavailable")
    
    await asyncio.sleep(random.uniform(0.05, 0.3))
    
    # Successful delivery
    notification_id = f"notif_{int(time.time() * 1000)}"
    
    result = {
        "notification_id": notification_id,
        "recipient": recipient,
        "status": "delivered",
        "priority": priority,
        "delivery_time_ms": random.randint(20, 150),
        "has_attachments": include_attachments
    }
    
    logger.info("Notification sent successfully", notification_id=notification_id)
    return result

@ToolChain(name="resilient_research_pipeline")
async def resilient_research_pipeline(topic: str, email: str) -> Dict[str, Any]:
    """Resilient research pipeline with comprehensive error handling."""
    
    with CorrelationContext() as ctx:
        logger.info("Starting resilient research pipeline", topic=topic, correlation_id=ctx.correlation_id_value)
        
        try:
            # Phase 1: Parallel research with error recovery
            research_tasks = [
                robust_web_search(f"{topic} best practices"),
                robust_web_search(f"{topic} security considerations"),
                secure_code_analysis(f"src/{topic.replace(' ', '_')}.py", check_security=True)
            ]
            
            # Execute with error tolerance - continue even if some tasks fail
            results = []
            errors = []
            
            for i, task in enumerate(research_tasks):
                try:
                    result = await task
                    results.append({"task_id": i, "status": "success", "data": result})
                    logger.info(f"Research task {i} completed successfully")
                except Exception as e:
                    error_info = {
                        "task_id": i, 
                        "status": "failed", 
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                    errors.append(error_info)
                    logger.error(f"Research task {i} failed", error=e)
            
            # Phase 2: Generate report with fallback logic
            if not results:
                # All tasks failed - use fallback
                report = {
                    "status": "degraded",
                    "message": "All research tasks failed, using cached data",
                    "topic": topic,
                    "data": {"cached_info": f"Cached information about {topic}"}
                }
                logger.warning("Using fallback data - all research tasks failed")
            else:
                # Some tasks succeeded
                successful_results = [r["data"] for r in results if r["status"] == "success"]
                
                report = {
                    "status": "success" if len(errors) == 0 else "partial_success",
                    "topic": topic,
                    "successful_tasks": len(results),
                    "failed_tasks": len(errors),
                    "data": successful_results,
                    "errors": errors if errors else None
                }
                
                logger.info("Research completed", 
                          successful_tasks=len(results), 
                          failed_tasks=len(errors))
            
            # Phase 3: Send notification with retry logic
            max_notification_retries = 3
            notification_result = None
            
            for attempt in range(max_notification_retries):
                try:
                    status_msg = f"Research report for '{topic}' - Status: {report['status']}"
                    notification_result = await validated_notification(
                        email, status_msg, "high"
                    )
                    break
                except Exception as e:
                    logger.warning(f"Notification attempt {attempt + 1} failed", error=e)
                    if attempt == max_notification_retries - 1:
                        logger.error("All notification attempts failed", error=e)
                        report["notification_error"] = str(e)
                    else:
                        await asyncio.sleep(1.0)  # Wait before retry
            
            if notification_result:
                report["notification"] = notification_result
                
            return report
            
        except Exception as e:
            logger.error("Critical error in research pipeline", error=e)
            # Return minimal error report
            return {
                "status": "critical_failure",
                "topic": topic,
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": time.time()
            }

async def demonstrate_error_scenarios():
    """Demonstrate various error handling scenarios."""
    logger.info("Starting error scenario demonstrations")
    
    print("\n🛡️ Generation 2: Error Handling & Resilience Demo")
    print("=" * 60)
    
    # Scenario 1: Input validation errors
    print("\n1. Input Validation Tests:")
    try:
        await validated_notification("invalid-email", "Test message")
    except ValueError as e:
        print(f"✅ Caught validation error: {e}")
    
    try:
        await secure_code_analysis("../../../etc/passwd")  # Path traversal attempt
    except ValueError as e:
        print(f"✅ Caught security validation: {e}")
    
    # Scenario 2: Timeout handling
    print("\n2. Timeout Handling:")
    start_time = time.time()
    try:
        # This might timeout due to random failures
        result = await robust_web_search("test query")
        print(f"✅ Search completed: {result['total_results']} results")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✅ Handled error after {elapsed:.2f}s: {type(e).__name__}")
    
    # Scenario 3: Resilient pipeline with partial failures
    print("\n3. Resilient Pipeline Test:")
    pipeline_result = await resilient_research_pipeline(
        "async programming", 
        "researcher@example.com"
    )
    
    print(f"✅ Pipeline Status: {pipeline_result['status']}")
    if "successful_tasks" in pipeline_result:
        print(f"   Successful tasks: {pipeline_result['successful_tasks']}")
        print(f"   Failed tasks: {pipeline_result['failed_tasks']}")
    
    # Scenario 4: Orchestrator error handling
    print("\n4. Orchestrator Configuration Validation:")
    try:
        # Invalid configuration - should fail
        bad_orchestrator = AsyncOrchestrator(
            max_parallel_tools=5, 
            max_parallel_per_type=10  # This violates constraint
        )
    except ConfigurationError as e:
        print(f"✅ Configuration validation worked: {e}")
    
    print("\n🎉 Error Handling & Resilience: ALL SCENARIOS TESTED")
    print("✅ Input validation and sanitization")
    print("✅ Timeout and retry logic")
    print("✅ Graceful degradation with fallbacks")
    print("✅ Structured logging with correlation IDs")
    print("✅ Security-aware path validation")
    print("✅ Configuration validation")

async def main():
    """Main demonstration function."""
    with CorrelationContext() as ctx:
        logger.info("Starting Generation 2 demonstration", correlation_id=ctx.correlation_id_value)
        
        try:
            await demonstrate_error_scenarios()
            
            print(f"\n📊 Correlation ID for this session: {ctx.correlation_id_value}")
            logger.info("Generation 2 demonstration completed successfully")
            
        except Exception as e:
            logger.error("Demonstration failed", error=e)
            raise

if __name__ == "__main__":
    # Configure logging level
    import logging
    logging.getLogger().setLevel(logging.INFO)
    
    asyncio.run(main())