#!/usr/bin/env python3
"""Production-ready example with all features enabled."""

import asyncio
import logging
import os
import time
from typing import Dict, Any
from async_toolformer import (
    AsyncOrchestrator, Tool, OrchestratorConfig, 
    RateLimitConfig, MemoryConfig
)


# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('orchestrator.log')
    ]
)

logger = logging.getLogger(__name__)


@Tool(description="Production data processor", priority=1, tags=["production"])
async def process_data(data_id: str, processing_level: str = "standard") -> dict:
    """Process data with production-level reliability."""
    logger.info(f"Processing data: {data_id} with level: {processing_level}")
    
    # Simulate processing time based on level
    processing_times = {"fast": 0.1, "standard": 0.3, "deep": 0.8}
    await asyncio.sleep(processing_times.get(processing_level, 0.3))
    
    return {
        "data_id": data_id,
        "processing_level": processing_level,
        "status": "completed",
        "processed_at": time.time(),
        "checksum": hash(data_id) % 10000
    }


@Tool(description="External API integration", priority=2, tags=["external", "api"])
async def call_external_api(endpoint: str, timeout: float = 5.0) -> dict:
    """Call external API with proper error handling."""
    logger.info(f"Calling external API: {endpoint}")
    
    # Simulate network call
    await asyncio.sleep(min(timeout * 0.1, 0.5))
    
    # Simulate occasional failures for robustness testing
    import random
    if random.random() < 0.05:  # 5% failure rate
        raise Exception(f"External API {endpoint} temporarily unavailable")
    
    return {
        "endpoint": endpoint,
        "response_code": 200,
        "data": f"response_from_{endpoint}",
        "latency_ms": timeout * 100,
        "timestamp": time.time()
    }


@Tool(description="Database operation", priority=1, rate_limit_group="database", tags=["database"])
async def database_operation(operation: str, table: str, record_id: str = None) -> dict:
    """Perform database operations with connection pooling."""
    logger.info(f"Database {operation} on {table}")
    
    # Simulate database operation
    operation_times = {"select": 0.05, "insert": 0.1, "update": 0.08, "delete": 0.06}
    await asyncio.sleep(operation_times.get(operation, 0.1))
    
    return {
        "operation": operation,
        "table": table,
        "record_id": record_id or f"auto_{int(time.time())}",
        "status": "success",
        "rows_affected": 1 if operation != "select" else random.randint(1, 10),
        "execution_time_ms": operation_times.get(operation, 0.1) * 1000
    }


@Tool(description="Report generator", priority=3, tags=["reporting", "no-cache"])
async def generate_report(report_type: str, format: str = "json") -> dict:
    """Generate reports with various formats."""
    logger.info(f"Generating {report_type} report in {format} format")
    
    # Reports should not be cached as they change frequently
    await asyncio.sleep(0.4)  # Report generation takes time
    
    return {
        "report_type": report_type,
        "format": format,
        "generated_at": time.time(),
        "size_bytes": random.randint(1024, 10240),
        "records_processed": random.randint(100, 1000),
        "checksum": f"{report_type}_{format}_{int(time.time())}"
    }


async def create_production_orchestrator() -> AsyncOrchestrator:
    """Create production-ready orchestrator configuration."""
    
    # Production memory configuration
    memory_config = MemoryConfig(
        max_memory_gb=4.0,  # 4GB memory limit
        compress_results=True,
        max_result_size_mb=50  # 50MB max result size
    )
    
    # Production rate limiting
    rate_config = RateLimitConfig(
        global_max=1000,  # 1000 requests per minute globally
        service_limits={
            "database": {"calls": 500, "window": 60},  # 500 DB ops per minute
            "external": {"calls": 200, "window": 60},  # 200 API calls per minute
            "reporting": {"calls": 50, "window": 300}  # 50 reports per 5 minutes
        },
        use_redis=False,  # Set to True if Redis is available
        redis_url=os.getenv("REDIS_URL")
    )
    
    # Main orchestrator configuration
    config = OrchestratorConfig(
        max_parallel_tools=50,  # High parallelism for production
        max_parallel_per_type=20,
        tool_timeout_ms=30000,  # 30 second timeout
        total_timeout_ms=180000,  # 3 minute total timeout
        retry_attempts=3,
        memory_config=memory_config,
        rate_limit_config=rate_config
    )
    
    # Create orchestrator with all production tools
    orchestrator = AsyncOrchestrator(
        tools=[process_data, call_external_api, database_operation, generate_report],
        config=config
    )
    
    logger.info("Production orchestrator initialized")
    return orchestrator


async def run_production_workload(orchestrator: AsyncOrchestrator):
    """Run a realistic production workload."""
    logger.info("Starting production workload simulation")
    
    # Simulate various production scenarios
    scenarios = [
        "Process customer data batch_001 with standard processing",
        "Generate daily sales report in PDF format", 
        "Update customer records in user_profiles table",
        "Call external payment API for transaction validation",
        "Process high-priority data batch_002 with deep processing",
        "Retrieve order history from orders table",
        "Generate weekly analytics report in Excel format",
        "Call external inventory API for stock check",
        "Insert new customer data into customer_data table",
        "Process standard data batch_003 with fast processing"
    ]
    
    # Execute scenarios concurrently
    start_time = time.time()
    
    tasks = []
    for i, scenario in enumerate(scenarios):
        task = orchestrator.execute(f"Scenario {i+1}: {scenario}")
        tasks.append(task)
        
        # Stagger requests slightly to simulate real-world patterns
        await asyncio.sleep(0.1)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful = sum(1 for r in results if not isinstance(r, Exception) and r.get('status') == 'completed')
    errors = sum(1 for r in results if isinstance(r, Exception))
    timeouts = sum(1 for r in results if not isinstance(r, Exception) and 'timeout' in str(r))
    
    logger.info(f"Production workload completed in {total_time:.2f}s")
    logger.info(f"Results: {successful} successful, {errors} errors, {timeouts} timeouts")
    
    return {
        "total_scenarios": len(scenarios),
        "successful": successful,
        "errors": errors,
        "timeouts": timeouts,
        "total_time": total_time,
        "throughput": len(scenarios) / total_time
    }


async def monitor_system_health(orchestrator: AsyncOrchestrator):
    """Monitor system health and performance."""
    logger.info("Collecting system health metrics")
    
    # Get comprehensive metrics
    metrics = await orchestrator.get_metrics()
    
    # Log key performance indicators
    cache_hit_rate = metrics.get('cache', {}).get('hit_rate', 0)
    total_requests = metrics.get('connection_pools', {}).get('total_requests', 0)
    active_tasks = metrics.get('active_tasks', 0)
    
    logger.info(f"Cache hit rate: {cache_hit_rate:.1%}")
    logger.info(f"Total network requests: {total_requests}")
    logger.info(f"Active tasks: {active_tasks}")
    
    # Check for potential issues
    warnings = []
    
    if cache_hit_rate < 0.5:
        warnings.append("Low cache hit rate - consider tuning cache configuration")
    
    if active_tasks > 100:
        warnings.append("High number of active tasks - potential performance bottleneck")
    
    if warnings:
        logger.warning("System health warnings:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
    else:
        logger.info("System health: All metrics within normal ranges")
    
    return metrics


async def main():
    """Main production example."""
    logger.info("🚀 Starting AsyncOrchestrator Production Example")
    logger.info("="*60)
    
    try:
        # Create production orchestrator
        orchestrator = await create_production_orchestrator()
        
        # Run health check
        logger.info("📊 Initial Health Check")
        await monitor_system_health(orchestrator)
        
        # Run production workload
        logger.info("⚡ Running Production Workload")
        workload_results = await run_production_workload(orchestrator)
        
        # Final health check
        logger.info("📊 Final Health Check")
        final_metrics = await monitor_system_health(orchestrator)
        
        # Performance summary
        logger.info("🎯 Performance Summary")
        logger.info(f"  Throughput: {workload_results['throughput']:.1f} operations/second")
        logger.info(f"  Success rate: {workload_results['successful']/workload_results['total_scenarios']:.1%}")
        logger.info(f"  Total scenarios: {workload_results['total_scenarios']}")
        logger.info(f"  Processing time: {workload_results['total_time']:.2f}s")
        
        # Resource utilization
        cache_metrics = final_metrics.get('cache', {})
        logger.info("📈 Resource Utilization")
        logger.info(f"  Cache entries: {cache_metrics.get('size', 0)}")
        logger.info(f"  Cache hit rate: {cache_metrics.get('hit_rate', 0):.1%}")
        logger.info(f"  Memory efficiency: High" if cache_metrics.get('hit_rate', 0) > 0.7 else "  Memory efficiency: Moderate")
        
        logger.info("✅ Production example completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Production example failed: {e}")
        raise
    
    finally:
        # Cleanup
        if 'orchestrator' in locals():
            await orchestrator.cleanup()
            logger.info("🧹 Cleanup completed")


if __name__ == "__main__":
    # Set up production environment
    os.environ.setdefault('ENVIRONMENT', 'production')
    os.environ.setdefault('LOG_LEVEL', 'INFO')
    
    # Run production example
    asyncio.run(main())