#!/usr/bin/env python3
"""
Quantum-Enhanced AsyncOrchestrator Example.

This example demonstrates the quantum-inspired task planning and execution
capabilities of the QuantumAsyncOrchestrator.
"""

import asyncio
import aiohttp
import time
from async_toolformer import (
    QuantumAsyncOrchestrator,
    QuantumToolRegistry,
    Tool,
    create_quantum_orchestrator
)


@Tool(description="Search the web for information about a topic")
async def quantum_web_search(query: str) -> dict:
    """Simulate quantum-enhanced web search."""
    await asyncio.sleep(0.8)  # Simulate API call with some variance
    return {
        "query": query,
        "results": [
            f"Quantum result 1 for {query}",
            f"Quantum result 2 for {query}",
            f"Quantum result 3 for {query}"
        ],
        "quantum_relevance_score": 0.95,
        "search_time_ms": 800
    }


@Tool(description="Analyze data using quantum algorithms")
async def quantum_data_analysis(data_source: str, analysis_type: str = "pattern") -> dict:
    """Simulate quantum data analysis."""
    await asyncio.sleep(1.2)  # More intensive computation
    return {
        "data_source": data_source,
        "analysis_type": analysis_type,
        "quantum_patterns": [
            "Superposition pattern detected",
            "Entanglement correlation: 0.87",
            "Quantum interference: constructive"
        ],
        "confidence": 0.92,
        "quantum_advantage": True,
        "processing_time_ms": 1200
    }


@Tool(description="Fetch data from multiple quantum sensors")
async def quantum_sensor_data(sensor_id: str, measurement_type: str = "amplitude") -> dict:
    """Simulate quantum sensor data collection."""
    await asyncio.sleep(0.3)
    return {
        "sensor_id": sensor_id,
        "measurement_type": measurement_type,
        "quantum_state": {
            "amplitude": 0.87,
            "phase": 2.34,
            "coherence": 0.94,
            "entanglement_degree": 0.76
        },
        "timestamp": time.time(),
        "measurement_uncertainty": 0.001
    }


@Tool(description="Optimize quantum circuits for better performance")
async def quantum_circuit_optimizer(circuit_description: str) -> dict:
    """Simulate quantum circuit optimization."""
    await asyncio.sleep(2.0)  # Complex optimization process
    return {
        "original_circuit": circuit_description,
        "optimized_gates": [
            "H(q0)", "CNOT(q0,q1)", "RZ(π/4,q1)", "CNOT(q0,q1)", "H(q0)"
        ],
        "gate_reduction": "35%",
        "depth_reduction": "28%",
        "fidelity_improvement": 0.97,
        "quantum_volume": 64,
        "optimization_time_ms": 2000
    }


@Tool(description="Simulate quantum machine learning algorithms")
async def quantum_ml_training(dataset: str, algorithm: str = "QSVM") -> dict:
    """Simulate quantum machine learning training."""
    await asyncio.sleep(1.5)
    return {
        "dataset": dataset,
        "algorithm": algorithm,
        "quantum_features": 128,
        "classical_features": 256,
        "training_accuracy": 0.94,
        "quantum_speedup": "4.2x",
        "convergence_iterations": 42,
        "final_loss": 0.032,
        "training_time_ms": 1500
    }


async def demonstrate_quantum_planning():
    """Demonstrate quantum-inspired task planning."""
    print("🌌 QUANTUM TASK PLANNING DEMO")
    print("=" * 50)
    
    # Create quantum orchestrator
    orchestrator = create_quantum_orchestrator(
        tools=[
            quantum_web_search,
            quantum_data_analysis,
            quantum_sensor_data,
            quantum_circuit_optimizer,
            quantum_ml_training
        ],
        quantum_config={
            "max_parallel_tasks": 8,
            "optimization_iterations": 75,
            "quantum_coherence_decay": 0.96,
            "enable_entanglement": True,
            "resource_limits": {
                "cpu": 200.0,
                "memory": 4096.0,
                "network": 1000.0,
                "io": 150.0,
            }
        }
    )
    
    print(f"🚀 Initialized QuantumAsyncOrchestrator")
    print(f"📊 Registered {len(orchestrator.registry._tools)} quantum tools")
    print()
    
    # Show initial quantum state
    print("🌊 Initial Quantum State:")
    print(orchestrator.get_quantum_state_visualization())
    print()
    
    # Example 1: Complex research task with quantum optimization
    print("🔬 Example 1: Quantum Research Pipeline")
    print("-" * 40)
    
    research_prompt = """
    I need to conduct advanced quantum computing research. Please:
    1. Search for the latest quantum computing breakthroughs
    2. Analyze quantum sensor data from multiple sources
    3. Optimize quantum circuits for better performance
    4. Train quantum machine learning models on the collected data
    5. Perform quantum data analysis on all results
    
    Use quantum-inspired optimization to maximize parallel execution and minimize total time.
    """
    
    start_time = time.time()
    result = await orchestrator.quantum_execute(
        prompt=research_prompt,
        optimize_plan=True,
        use_speculation=False,
        progress_callback=lambda phase, total, phase_result: print(
            f"  ⚡ Phase {phase}/{total} completed: "
            f"{len(phase_result['results'])} tasks in "
            f"{phase_result['execution_time_ms']:.1f}ms"
        )
    )
    
    execution_time = time.time() - start_time
    
    print(f"\n📈 Quantum Execution Results:")
    print(f"  Execution ID: {result['execution_id']}")
    print(f"  Status: {result['status']}")
    print(f"  Total Time: {result['total_time_ms']:.1f}ms")
    print(f"  Tools Executed: {result['tools_executed']}")
    print(f"  Successful: {result['successful_tools']}")
    print(f"  Failed: {result['failed_tools']}")
    print(f"  Quantum Coherence: {result['quantum_metrics']['quantum_coherence']:.3f}")
    print(f"  Parallelism Achieved: {result['quantum_metrics']['parallelism_achieved']:.2f}x")
    print(f"  Time Efficiency: {result['quantum_metrics']['time_efficiency']:.2f}")
    print(f"  Optimization Score: {result['quantum_metrics']['optimization_score']:.3f}")
    print()
    
    print(f"📊 Execution Plan:")
    plan = result['execution_plan']
    print(f"  Phases: {plan['phases']}")
    print(f"  Estimated Time: {plan['total_estimated_time_ms']:.1f}ms")
    print(f"  Parallelism Factor: {plan['parallelism_factor']:.2f}x")
    print(f"  Resource Utilization:")
    for resource, usage in plan['resource_utilization'].items():
        print(f"    {resource}: {usage:.1f}")
    print()
    
    # Show individual tool results
    if result.get('results'):
        print("🔧 Individual Tool Results:")
        for i, tool_result in enumerate(result['results']):
            print(f"  {i+1}. {tool_result.tool_name}")
            print(f"     Success: {tool_result.success}")
            print(f"     Time: {tool_result.execution_time_ms:.1f}ms")
            if tool_result.success and tool_result.data:
                # Show key metrics from quantum tools
                data = tool_result.data
                if isinstance(data, dict):
                    for key, value in list(data.items())[:3]:  # Show first 3 items
                        print(f"     {key}: {value}")
            print()
    
    # Example 2: Streaming quantum execution
    print("📡 Example 2: Quantum Streaming Execution")
    print("-" * 40)
    
    streaming_prompt = """
    Perform parallel quantum computations:
    - Collect quantum sensor data from sensors A, B, and C
    - Search for quantum algorithms information
    - Train a quantum ML model on sensor data
    - Optimize quantum circuits based on the results
    """
    
    print("Streaming quantum execution results...")
    phase_count = 0
    total_time = 0
    
    async for phase_result in orchestrator.quantum_stream_execute(
        prompt=streaming_prompt,
        optimize_plan=True
    ):
        phase_count += 1
        total_time = phase_result['cumulative_time_ms']
        
        print(f"  🌊 Phase {phase_result['phase']}/{phase_result['total_phases']}:")
        print(f"     Tasks: {phase_result['tasks_in_phase']}")
        print(f"     Phase Time: {phase_result['phase_execution_time_ms']:.1f}ms")
        print(f"     Cumulative Time: {phase_result['cumulative_time_ms']:.1f}ms")
        print(f"     Quantum Coherence: {phase_result['quantum_coherence']:.3f}")
        
        successful_tasks = sum(1 for r in phase_result['results'] if r.success)
        print(f"     Successful Tasks: {successful_tasks}/{len(phase_result['results'])}")
        
        print()
    
    print(f"📊 Streaming completed: {phase_count} phases in {total_time:.1f}ms")
    print()
    
    # Example 3: Quantum state analysis
    print("🔍 Example 3: Quantum State Analysis")
    print("-" * 40)
    
    # Show final quantum state
    print("Final Quantum State:")
    print(orchestrator.get_quantum_state_visualization())
    print()
    
    # Get enhanced metrics
    metrics = orchestrator.get_enhanced_metrics()
    print("📈 Enhanced Metrics:")
    print(f"  Total Registered Tools: {metrics['registered_tools']}")
    print(f"  Quantum Tasks: {metrics['quantum']['registered_tasks']}")
    print(f"  Entangled Pairs: {metrics['quantum']['entangled_pairs']}")
    print(f"  System Coherence: {metrics['quantum']['quantum_coherence']:.3f}")
    print(f"  Active Execution: {metrics['quantum_execution_active']}")
    
    if metrics['current_plan']:
        plan_info = metrics['current_plan']
        print(f"  Last Plan Phases: {plan_info['phases']}")
        print(f"  Optimization Score: {plan_info['optimization_score']:.3f}")
        print(f"  Parallelism Factor: {plan_info['parallelism_factor']:.2f}x")
    
    print()
    
    # Example 4: Tool optimization analysis
    print("⚙️ Example 4: Tool Optimization Analysis")
    print("-" * 40)
    
    optimization_result = await orchestrator.optimize_registered_tools()
    
    print("Tool Optimization Results:")
    print(f"  Total Tools: {optimization_result['total_tools']}")
    print(f"  Execution Phases: {optimization_result['execution_phases']}")
    print(f"  Estimated Time: {optimization_result['estimated_time_ms']:.1f}ms")
    print(f"  Parallelism Factor: {optimization_result['parallelism_factor']:.2f}x")
    print(f"  Optimization Score: {optimization_result['optimization_score']:.3f}")
    print()
    
    print("Phase Breakdown:")
    for phase_info in optimization_result['phase_breakdown']:
        print(f"  Phase {phase_info['phase']}: {phase_info['tasks']} tasks")
        print(f"    Tasks: {', '.join(phase_info['task_names'])}")
        print(f"    Estimated Time: {phase_info['estimated_time_ms']:.1f}ms")
        print()
    
    # Cleanup
    await orchestrator.cleanup()
    print("🎉 Quantum demonstration completed!")


async def demonstrate_quantum_registry():
    """Demonstrate quantum tool registry features."""
    print("\n🔧 QUANTUM TOOL REGISTRY DEMO")
    print("=" * 50)
    
    # Create quantum registry
    registry = QuantumToolRegistry()
    
    # Register tools with quantum characteristics
    registry.register_quantum_tool(
        tool=quantum_web_search,
        estimated_duration_ms=800,
        resource_requirements={"cpu": 15.0, "memory": 100.0, "network": 200.0},
        success_probability=0.95,
        quantum_priority=1.5,
        search_type="quantum_enhanced",
        optimization_level="high"
    )
    
    registry.register_quantum_tool(
        tool=quantum_circuit_optimizer,
        estimated_duration_ms=2000,
        resource_requirements={"cpu": 50.0, "memory": 300.0, "network": 10.0},
        success_probability=0.88,
        quantum_priority=2.0,
        complexity="high",
        quantum_advantage="significant"
    )
    
    print(f"📚 Registered {len(registry._tools)} tools with quantum characteristics")
    print()
    
    # Show quantum characteristics
    for tool_name in registry._tools.keys():
        characteristics = registry.get_quantum_characteristics(tool_name)
        if characteristics:
            print(f"🔬 {tool_name}:")
            print(f"  Duration: {characteristics['estimated_duration_ms']}ms")
            print(f"  Resources: {characteristics['resource_requirements']}")
            print(f"  Success Probability: {characteristics['success_probability']}")
            print(f"  Quantum Priority: {characteristics['quantum_priority']}")
            print(f"  Additional: {', '.join(k for k in characteristics.keys() if k not in ['estimated_duration_ms', 'resource_requirements', 'success_probability', 'quantum_priority'])}")
            print()


async def main():
    """Run the complete quantum demonstration."""
    print("🌌 QUANTUM-INSPIRED ASYNC ORCHESTRATOR")
    print("=" * 60)
    print("Demonstrating quantum-enhanced task planning and execution")
    print("=" * 60)
    print()
    
    try:
        # Run main quantum planning demo
        await demonstrate_quantum_planning()
        
        # Run quantum registry demo
        await demonstrate_quantum_registry()
        
        print("\n✨ All quantum demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Use uvloop if available for better async performance
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        print("🚀 Using uvloop for enhanced performance")
    except ImportError:
        print("📝 Using standard asyncio event loop")
    
    asyncio.run(main())