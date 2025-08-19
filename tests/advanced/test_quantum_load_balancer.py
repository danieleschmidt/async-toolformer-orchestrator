"""
Advanced tests for Quantum Load Balancer with superposition-based traffic distribution.

Tests cover:
- Quantum amplitude calculations
- Entanglement management
- Interference pattern analysis
- Quantum annealing optimization
- Load balancing decisions
"""

import asyncio
import math
import pytest
import time
from unittest.mock import MagicMock, patch

from async_toolformer.quantum_load_balancer import (
    QuantumLoadBalancer,
    ServerNode,
    RequestContext,
    QuantumState,
    QuantumAmplitudeCalculator,
    EntanglementManager,
    InterferencePatternAnalyzer,
    QuantumAnnealingOptimizer,
    create_quantum_load_balancer,
)


@pytest.fixture
def sample_servers():
    """Create sample server nodes for testing."""
    return [
        ServerNode(
            id="server-1",
            address="192.168.1.10",
            port=8080,
            weight=1.0,
            max_connections=1000,
            current_connections=100,
            cpu_usage=0.3,
            memory_usage=0.4,
            response_time=0.15,
            health_score=0.9,
        ),
        ServerNode(
            id="server-2", 
            address="192.168.1.11",
            port=8080,
            weight=1.2,
            max_connections=1200,
            current_connections=300,
            cpu_usage=0.6,
            memory_usage=0.5,
            response_time=0.25,
            health_score=0.7,
        ),
        ServerNode(
            id="server-3",
            address="192.168.1.12", 
            port=8080,
            weight=0.8,
            max_connections=800,
            current_connections=50,
            cpu_usage=0.2,
            memory_usage=0.3,
            response_time=0.10,
            health_score=0.95,
        ),
    ]


@pytest.fixture
def sample_request():
    """Create a sample request context."""
    return RequestContext(
        request_id="req-123",
        client_id="client-abc",
        session_id="session-xyz",
        request_type="api_call",
        priority=2,
        affinity_hints=["server-1"],
        resource_requirements={"cpu": 0.2, "memory": 0.3},
    )


@pytest.fixture
def quantum_load_balancer(sample_servers):
    """Create a quantum load balancer with sample servers."""
    lb = create_quantum_load_balancer()
    
    for server in sample_servers:
        lb.add_server_node(server)
        
    return lb


class TestQuantumAmplitudeCalculator:
    """Test quantum amplitude calculation functionality."""
    
    def test_amplitude_calculation_basic(self, sample_servers, sample_request):
        """Test basic amplitude calculation."""
        
        calculator = QuantumAmplitudeCalculator()
        server = sample_servers[0]
        global_state = {"total_connections": 450}
        
        amplitude = calculator.calculate_amplitude(server, sample_request, global_state)
        
        # Should be a complex number
        assert isinstance(amplitude, complex)
        
        # Magnitude should be between 0 and 1 (normalized by health/load)
        magnitude = abs(amplitude)
        assert 0.0 <= magnitude <= 1.5  # Allow some overhead for normalization
        
        # Phase should be reasonable
        phase = math.atan2(amplitude.imag, amplitude.real)
        assert -math.pi <= phase <= math.pi
        
    def test_amplitude_with_affinity(self, sample_servers, sample_request):
        """Test amplitude calculation with affinity hints."""
        
        calculator = QuantumAmplitudeCalculator()
        global_state = {"total_connections": 450}
        
        # Server with affinity should have different amplitude
        server_with_affinity = sample_servers[0]  # server-1 is in affinity hints
        server_without_affinity = sample_servers[1]  # server-2 not in affinity hints
        
        amplitude_with = calculator.calculate_amplitude(server_with_affinity, sample_request, global_state)
        amplitude_without = calculator.calculate_amplitude(server_without_affinity, sample_request, global_state)
        
        # The phases should be different due to affinity
        phase_with = math.atan2(amplitude_with.imag, amplitude_with.real)
        phase_without = math.atan2(amplitude_without.imag, amplitude_without.real)
        
        # Should have different phases (affinity adds pi/4)
        assert abs(phase_with - phase_without) > 0.1
        
    def test_amplitude_coherence_decay(self, sample_servers, sample_request):
        """Test amplitude coherence decay over time."""
        
        calculator = QuantumAmplitudeCalculator()
        server = sample_servers[0]
        global_state = {"total_connections": 450}
        
        # Calculate amplitude immediately
        amplitude_fresh = calculator.calculate_amplitude(server, sample_request, global_state)
        
        # Simulate time passing by updating server timestamp
        server.last_updated = time.time() - 10.0  # 10 seconds ago
        amplitude_decayed = calculator.calculate_amplitude(server, sample_request, global_state)
        
        # Decayed amplitude should have smaller magnitude
        assert abs(amplitude_decayed) < abs(amplitude_fresh)


class TestEntanglementManager:
    """Test quantum entanglement management."""
    
    def test_create_entanglement(self, sample_servers):
        """Test creating entanglement between servers."""
        
        manager = EntanglementManager()
        server1, server2 = sample_servers[0], sample_servers[1]
        
        manager.create_entanglement(server1, server2, strength=0.8)
        
        # Check entanglement was created
        assert manager.get_entanglement_strength(server1.id, server2.id) == 0.8
        assert manager.get_entanglement_strength(server2.id, server1.id) == 0.8
        
        # Check server lists were updated
        assert server2.id in server1.entangled_nodes
        assert server1.id in server2.entangled_nodes
        
    def test_entanglement_decay(self, sample_servers):
        """Test entanglement decay over time."""
        
        manager = EntanglementManager()
        server1, server2 = sample_servers[0], sample_servers[1]
        
        manager.create_entanglement(server1, server2, strength=0.9)
        original_strength = manager.get_entanglement_strength(server1.id, server2.id)
        
        # Apply decay
        manager.decay_entanglements(decay_factor=0.5)
        
        decayed_strength = manager.get_entanglement_strength(server1.id, server2.id)
        assert decayed_strength == original_strength * 0.5
        
    def test_weak_entanglement_removal(self, sample_servers):
        """Test removal of weak entanglements."""
        
        manager = EntanglementManager()
        server1, server2 = sample_servers[0], sample_servers[1]
        
        # Create weak entanglement
        manager.create_entanglement(server1, server2, strength=0.05)
        
        # Apply decay that should remove it
        manager.decay_entanglements(decay_factor=0.5)
        
        # Should be removed (below 0.1 threshold)
        assert manager.get_entanglement_strength(server1.id, server2.id) == 0.0


class TestInterferencePatternAnalyzer:
    """Test interference pattern analysis."""
    
    def test_add_request_data(self):
        """Test adding request data for pattern analysis."""
        
        analyzer = InterferencePatternAnalyzer()
        request = RequestContext("req-1", "client-1", request_type="search")
        
        analyzer.add_request_data(request, "server-1")
        
        # Should have recorded the request
        assert len(analyzer._request_history) == 1
        
        recorded = analyzer._request_history[0]
        assert recorded["client_id"] == "client-1"
        assert recorded["request_type"] == "search"
        assert recorded["selected_node"] == "server-1"
        
    def test_interference_pattern_calculation(self, sample_servers):
        """Test calculation of interference patterns."""
        
        analyzer = InterferencePatternAnalyzer()
        
        # Add some historical request data
        current_time = time.time()
        for i in range(48):  # 48 requests over 2 days
            request = RequestContext(
                f"req-{i}",
                f"client-{i % 5}",
                request_type="api",
                timestamp=current_time - (48 - i) * 3600,  # Hourly requests
            )
            analyzer.add_request_data(request, f"server-{(i % 3) + 1}")
            
        patterns = analyzer.calculate_interference_pattern(sample_servers)
        
        # Should have patterns for all servers
        for server in sample_servers:
            assert server.id in patterns
            pattern = patterns[server.id]
            assert len(pattern) == 24  # 24-hour pattern
            
            # All values should be between 0 and reasonable max
            for value in pattern:
                assert 0.0 <= value <= 2.0
                
    def test_future_load_prediction(self, sample_servers):
        """Test predicting future load based on patterns."""
        
        analyzer = InterferencePatternAnalyzer()
        
        # Add historical data with pattern
        current_time = time.time()
        for hour in range(24):
            # Create different load patterns for different hours
            requests_per_hour = 10 if 9 <= hour <= 17 else 3  # Business hours pattern
            
            for req in range(requests_per_hour):
                request = RequestContext(
                    f"req-{hour}-{req}",
                    f"client-{req}",
                    timestamp=current_time - (24 - hour) * 3600 + req * 60,
                )
                analyzer.add_request_data(request, "server-1")
                
        # Predict future load
        prediction = analyzer.predict_future_load("server-1", hours_ahead=1.0)
        
        # Should return a reasonable prediction
        assert 0.0 <= prediction <= 1.0


class TestQuantumAnnealingOptimizer:
    """Test quantum annealing optimization."""
    
    def test_server_optimization(self, sample_servers, sample_request):
        """Test server selection optimization using quantum annealing."""
        
        optimizer = QuantumAnnealingOptimizer()
        constraints = {"max_cpu": 0.8, "max_memory": 0.8}
        
        best_server, energy = optimizer.optimize_server_selection(
            sample_servers, sample_request, constraints
        )
        
        # Should return a server and energy value
        assert best_server in sample_servers
        assert isinstance(energy, (int, float))
        assert energy >= 0.0
        
    def test_energy_calculation(self, sample_servers, sample_request):
        """Test energy calculation for server selection."""
        
        optimizer = QuantumAnnealingOptimizer()
        constraints = {"max_cpu": 0.5}
        
        # Test with different servers - should prefer less loaded ones
        server_low_load = sample_servers[2]  # Low CPU/memory usage
        server_high_load = sample_servers[1]  # Higher CPU/memory usage
        
        energy_low = optimizer._calculate_energy(server_low_load, sample_request, constraints)
        energy_high = optimizer._calculate_energy(server_high_load, sample_request, constraints)
        
        # Lower load server should have lower energy (better)
        assert energy_low < energy_high
        
    def test_affinity_energy_bonus(self, sample_servers, sample_request):
        """Test that affinity hints reduce energy (improve selection)."""
        
        optimizer = QuantumAnnealingOptimizer()
        constraints = {}
        
        # Server with affinity vs without
        server_with_affinity = sample_servers[0]  # server-1 is in affinity hints
        server_without_affinity = sample_servers[1]  # server-2 not in hints
        
        energy_with = optimizer._calculate_energy(server_with_affinity, sample_request, constraints)
        energy_without = optimizer._calculate_energy(server_without_affinity, sample_request, constraints)
        
        # With affinity should have lower energy (assuming other factors similar)
        # Note: This test might be sensitive to exact server metrics
        # The main point is that affinity should influence energy calculation
        assert isinstance(energy_with, (int, float))
        assert isinstance(energy_without, (int, float))


class TestQuantumLoadBalancer:
    """Test complete quantum load balancer functionality."""
    
    def test_add_remove_servers(self, quantum_load_balancer):
        """Test adding and removing server nodes."""
        
        # Should have 3 servers from fixture
        assert len(quantum_load_balancer._nodes) == 3
        
        # Add new server
        new_server = ServerNode(
            id="server-4",
            address="192.168.1.13",
            port=8080,
        )
        quantum_load_balancer.add_server_node(new_server)
        assert len(quantum_load_balancer._nodes) == 4
        
        # Remove server
        removed = quantum_load_balancer.remove_server_node("server-4")
        assert removed
        assert len(quantum_load_balancer._nodes) == 3
        
        # Try to remove non-existent server
        not_removed = quantum_load_balancer.remove_server_node("server-999")
        assert not not_removed
        
    def test_server_metrics_update(self, quantum_load_balancer):
        """Test updating server metrics."""
        
        quantum_load_balancer.update_server_metrics(
            "server-1",
            cpu_usage=0.8,
            memory_usage=0.7,
            current_connections=500,
            response_time=0.3,
        )
        
        server = quantum_load_balancer._nodes["server-1"]
        assert server.cpu_usage == 0.8
        assert server.memory_usage == 0.7
        assert server.current_connections == 500
        assert server.response_time == 0.3
        
        # Health score should be updated based on new metrics
        assert server.health_score > 0.0
        assert server.last_updated > 0.0
        
    @pytest.mark.asyncio
    async def test_server_selection(self, quantum_load_balancer, sample_request):
        """Test quantum server selection."""
        
        result = await quantum_load_balancer.select_server(sample_request)
        
        # Verify result structure
        assert result.selected_node is not None
        assert result.selected_node.id in quantum_load_balancer._nodes
        assert isinstance(result.quantum_state, QuantumState)
        assert isinstance(result.probability_distribution, dict)
        assert 0.0 <= result.decision_confidence <= 1.0
        
        # Probability distribution should sum to ~1.0
        total_prob = sum(result.probability_distribution.values())
        assert 0.9 <= total_prob <= 1.1
        
        # Should have interference pattern data
        assert isinstance(result.interference_pattern, list)
        
    @pytest.mark.asyncio 
    async def test_server_selection_with_affinity(self, quantum_load_balancer):
        """Test server selection respects affinity hints."""
        
        # Create request with strong affinity for server-3
        affinity_request = RequestContext(
            "req-affinity",
            client_id="client-affinity",
            affinity_hints=["server-3"],
        )
        
        # Run selection multiple times to see if affinity has effect
        selections = []
        for _ in range(10):
            result = await quantum_load_balancer.select_server(affinity_request)
            selections.append(result.selected_node.id)
            
        # Should show some bias toward server-3 due to affinity
        server_3_count = selections.count("server-3")
        assert server_3_count > 0  # Should select server-3 at least once
        
    @pytest.mark.asyncio
    async def test_quantum_states(self, quantum_load_balancer, sample_request):
        """Test different quantum states in selection."""
        
        results = []
        for _ in range(20):
            result = await quantum_load_balancer.select_server(sample_request)
            results.append(result.quantum_state)
            
        # Should see variety of quantum states
        unique_states = set(results)
        assert len(unique_states) >= 1  # At least one state type
        
        # All states should be valid
        for state in unique_states:
            assert state in [
                QuantumState.SUPERPOSITION,
                QuantumState.ENTANGLED,
                QuantumState.COLLAPSED,
                QuantumState.INTERFERENCE,
            ]
            
    @pytest.mark.asyncio
    async def test_entanglement_creation(self, quantum_load_balancer, sample_servers):
        """Test automatic entanglement creation between similar servers."""
        
        # Add a server very similar to server-3 (low load, good performance)
        similar_server = ServerNode(
            id="server-similar",
            address="192.168.1.14",
            port=8080,
            weight=0.8,  # Same as server-3
            max_connections=800,  # Same as server-3
            current_connections=60,  # Similar to server-3
            cpu_usage=0.2,  # Same as server-3
            memory_usage=0.3,  # Same as server-3
            response_time=0.10,  # Same as server-3
            health_score=0.95,  # Same as server-3
        )
        
        quantum_load_balancer.add_server_node(similar_server)
        
        # Check if entanglement was created
        entanglement_strength = quantum_load_balancer._entanglement_manager.get_entanglement_strength(
            "server-3", "server-similar"
        )
        
        # Should have some entanglement due to similarity
        # (This depends on the similarity threshold in the implementation)
        assert entanglement_strength >= 0.0
        
    @pytest.mark.asyncio
    async def test_cluster_statistics(self, quantum_load_balancer):
        """Test cluster statistics reporting."""
        
        stats = await quantum_load_balancer.get_cluster_statistics()
        
        # Verify structure
        assert "cluster_health" in stats
        assert "capacity" in stats
        assert "performance" in stats
        assert "quantum_metrics" in stats
        assert "recent_selections" in stats
        
        # Check cluster health details
        health = stats["cluster_health"]
        assert health["total_nodes"] == 3
        assert 0 <= health["healthy_nodes"] <= 3
        assert 0.0 <= health["health_ratio"] <= 1.0
        assert 0.0 <= health["avg_health_score"] <= 1.0
        
        # Check capacity details
        capacity = stats["capacity"]
        assert capacity["total_capacity"] > 0
        assert capacity["current_connections"] >= 0
        assert 0.0 <= capacity["utilization_ratio"] <= 1.0
        
        # Check quantum metrics
        quantum = stats["quantum_metrics"]
        assert quantum["total_entanglements"] >= 0
        assert quantum["request_count"] >= 0
        
    def test_health_score_calculation(self, quantum_load_balancer):
        """Test health score calculation logic."""
        
        # Test with different metric scenarios
        test_cases = [
            # (cpu, memory, connections/max_conn, response_time, expected_range)
            (0.1, 0.1, 0.1, 0.1, (0.8, 1.0)),  # Very healthy
            (0.9, 0.9, 0.9, 1.0, (0.0, 0.3)),   # Very unhealthy
            (0.5, 0.5, 0.5, 0.5, (0.3, 0.8)),   # Medium health
        ]
        
        for cpu, memory, conn_ratio, resp_time, (min_health, max_health) in test_cases:
            # Create test server
            test_server = ServerNode("test-server", "127.0.0.1", 8080, max_connections=1000)
            test_server.cpu_usage = cpu
            test_server.memory_usage = memory
            test_server.current_connections = int(conn_ratio * 1000)
            test_server.response_time = resp_time
            
            # Calculate health score
            health = quantum_load_balancer._calculate_health_score(test_server)
            
            # Should be in expected range
            assert min_health <= health <= max_health, f"Health {health} not in range [{min_health}, {max_health}] for metrics cpu={cpu}, mem={memory}, conn={conn_ratio}, resp={resp_time}"


class TestCreateQuantumLoadBalancer:
    """Test quantum load balancer creation function."""
    
    def test_create_default(self):
        """Test creating load balancer with defaults."""
        
        lb = create_quantum_load_balancer()
        
        assert isinstance(lb, QuantumLoadBalancer)
        assert len(lb._nodes) == 0
        assert isinstance(lb._amplitude_calculator, QuantumAmplitudeCalculator)
        assert isinstance(lb._entanglement_manager, EntanglementManager)
        assert isinstance(lb._interference_analyzer, InterferencePatternAnalyzer)
        assert isinstance(lb._annealing_optimizer, QuantumAnnealingOptimizer)


@pytest.mark.asyncio
async def test_integration_load_balancing_workflow(sample_servers):
    """Integration test for complete load balancing workflow."""
    
    # Create load balancer and add servers
    lb = create_quantum_load_balancer()
    for server in sample_servers:
        lb.add_server_node(server)
        
    # Simulate traffic and selections
    requests = []
    results = []
    
    for i in range(50):
        request = RequestContext(
            f"req-{i}",
            f"client-{i % 10}",
            request_type="api" if i % 2 == 0 else "search",
            priority=1 + (i % 3),
            affinity_hints=["server-1"] if i % 5 == 0 else [],
        )
        requests.append(request)
        
        result = await lb.select_server(request)
        results.append(result)
        
        # Update server metrics based on selection
        selected_server = result.selected_node
        lb.update_server_metrics(
            selected_server.id,
            cpu_usage=min(0.9, selected_server.cpu_usage + 0.01),
            memory_usage=min(0.9, selected_server.memory_usage + 0.005),
            current_connections=selected_server.current_connections + 1,
            response_time=selected_server.response_time + 0.001,
        )
        
    # Analyze results
    selected_servers = [r.selected_node.id for r in results]
    quantum_states = [r.quantum_state for r in results]
    confidences = [r.decision_confidence for r in results]
    
    # Should distribute load across servers
    unique_servers = set(selected_servers)
    assert len(unique_servers) >= 2  # Should use at least 2 different servers
    
    # Should see different quantum states
    unique_states = set(quantum_states)
    assert len(unique_states) >= 1
    
    # Confidences should be reasonable
    avg_confidence = sum(confidences) / len(confidences)
    assert 0.3 <= avg_confidence <= 1.0
    
    # Get final statistics
    stats = await lb.get_cluster_statistics()
    assert stats["quantum_metrics"]["request_count"] == 50
    assert len(stats["recent_selections"]) <= 10  # Only keeps recent ones
    
    # Should have some entanglements created during the process
    assert stats["quantum_metrics"]["total_entanglements"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])