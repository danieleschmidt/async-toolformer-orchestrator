"""
Quantum-Inspired Load Balancer - Generation 3 Implementation.

Implements quantum-inspired load balancing algorithms with:
- Superposition-based traffic distribution
- Entanglement for correlated resource allocation
- Quantum annealing for optimal server selection
- Interference patterns for predictive scaling
"""

import asyncio
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import structlog

logger = structlog.get_logger(__name__)


class QuantumState(Enum):
    """Quantum states for load balancing."""
    SUPERPOSITION = "superposition"
    ENTANGLED = "entangled"
    COLLAPSED = "collapsed"
    INTERFERENCE = "interference"


@dataclass
class ServerNode:
    """Represents a server node in the load balancer."""
    id: str
    address: str
    port: int
    weight: float = 1.0
    max_connections: int = 1000
    current_connections: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    response_time: float = 0.0
    health_score: float = 1.0
    quantum_amplitude: complex = complex(1.0, 0.0)
    entangled_nodes: List[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


@dataclass
class RequestContext:
    """Context for a load balancing request."""
    request_id: str
    client_id: Optional[str] = None
    session_id: Optional[str] = None
    request_type: str = "default"
    priority: int = 1
    affinity_hints: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class QuantumLoadBalancingResult:
    """Result of quantum load balancing."""
    selected_node: ServerNode
    quantum_state: QuantumState
    probability_distribution: Dict[str, float]
    entanglement_strength: float
    interference_pattern: List[float]
    decision_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumAmplitudeCalculator:
    """Calculates quantum amplitudes for server nodes."""
    
    def __init__(self):
        self._coherence_decay = 0.95
        self._entanglement_strength = 0.8
        
    def calculate_amplitude(
        self, 
        node: ServerNode, 
        request: RequestContext,
        global_state: Dict[str, Any],
    ) -> complex:
        """Calculate quantum amplitude for a server node."""
        
        # Base amplitude from health and performance metrics
        health_component = node.health_score
        load_component = 1.0 - (node.current_connections / max(node.max_connections, 1))
        performance_component = 1.0 - min(node.cpu_usage + node.memory_usage, 1.0)
        
        # Combine components
        magnitude = (health_component * 0.4 + load_component * 0.3 + performance_component * 0.3)
        
        # Add quantum phase based on request context
        phase = self._calculate_quantum_phase(node, request)
        
        # Create complex amplitude
        amplitude = complex(magnitude * math.cos(phase), magnitude * math.sin(phase))
        
        # Apply coherence decay
        time_factor = time.time() - node.last_updated
        decay_factor = self._coherence_decay ** time_factor
        
        return amplitude * decay_factor
        
    def _calculate_quantum_phase(self, node: ServerNode, request: RequestContext) -> float:
        """Calculate quantum phase for amplitude."""
        
        # Base phase from node characteristics
        base_phase = hash(node.id) % (2 * math.pi)
        
        # Modify phase based on request affinity
        affinity_phase = 0.0
        if request.affinity_hints and node.id in request.affinity_hints:
            affinity_phase = math.pi / 4  # Positive interference
            
        # Add request priority influence
        priority_phase = (request.priority - 1) * math.pi / 8
        
        return base_phase + affinity_phase + priority_phase


class EntanglementManager:
    """Manages quantum entanglement between server nodes."""
    
    def __init__(self):
        self._entanglement_matrix: Dict[Tuple[str, str], float] = {}
        self._entanglement_history: deque = deque(maxlen=1000)
        
    def create_entanglement(
        self, 
        node1: ServerNode, 
        node2: ServerNode, 
        strength: float = 0.8
    ) -> None:
        """Create entanglement between two nodes."""
        
        key1 = (node1.id, node2.id)
        key2 = (node2.id, node1.id)
        
        self._entanglement_matrix[key1] = strength
        self._entanglement_matrix[key2] = strength
        
        # Update node entanglement lists
        if node2.id not in node1.entangled_nodes:
            node1.entangled_nodes.append(node2.id)
        if node1.id not in node2.entangled_nodes:
            node2.entangled_nodes.append(node1.id)
            
        self._entanglement_history.append({
            "timestamp": time.time(),
            "node1": node1.id,
            "node2": node2.id,
            "strength": strength,
            "action": "create",
        })
        
        logger.info(
            "Quantum entanglement created",
            node1=node1.id,
            node2=node2.id,
            strength=strength,
        )
        
    def get_entanglement_strength(self, node1_id: str, node2_id: str) -> float:
        """Get entanglement strength between two nodes."""
        return self._entanglement_matrix.get((node1_id, node2_id), 0.0)
        
    def update_entanglement_from_traffic(
        self, 
        nodes: List[ServerNode], 
        traffic_correlation: Dict[Tuple[str, str], float]
    ) -> None:
        """Update entanglements based on traffic correlation patterns."""
        
        for (node1_id, node2_id), correlation in traffic_correlation.items():
            if correlation > 0.7:  # High correlation threshold
                node1 = next((n for n in nodes if n.id == node1_id), None)
                node2 = next((n for n in nodes if n.id == node2_id), None)
                
                if node1 and node2:
                    current_strength = self.get_entanglement_strength(node1_id, node2_id)
                    new_strength = min(correlation, 0.95)  # Cap at 0.95
                    
                    if new_strength > current_strength:
                        self.create_entanglement(node1, node2, new_strength)
                        
    def decay_entanglements(self, decay_factor: float = 0.99) -> None:
        """Apply decay to all entanglements."""
        
        keys_to_remove = []
        
        for key, strength in self._entanglement_matrix.items():
            new_strength = strength * decay_factor
            
            if new_strength < 0.1:  # Remove weak entanglements
                keys_to_remove.append(key)
            else:
                self._entanglement_matrix[key] = new_strength
                
        for key in keys_to_remove:
            del self._entanglement_matrix[key]


class InterferencePatternAnalyzer:
    """Analyzes interference patterns for predictive load balancing."""
    
    def __init__(self):
        self._request_history: deque = deque(maxlen=5000)
        self._pattern_cache: Dict[str, List[float]] = {}
        self._cache_ttl = 300  # 5 minutes
        
    def add_request_data(self, request: RequestContext, selected_node_id: str) -> None:
        """Add request data for pattern analysis."""
        
        self._request_history.append({
            "timestamp": request.timestamp,
            "client_id": request.client_id,
            "request_type": request.request_type,
            "selected_node": selected_node_id,
            "priority": request.priority,
        })
        
    def calculate_interference_pattern(
        self, 
        nodes: List[ServerNode], 
        time_window: float = 300.0
    ) -> Dict[str, List[float]]:
        """Calculate interference patterns for each node."""
        
        current_time = time.time()
        cache_key = f"interference_{int(current_time // self._cache_ttl)}"
        
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key]
            
        # Filter recent requests
        recent_requests = [
            req for req in self._request_history
            if current_time - req["timestamp"] <= time_window
        ]
        
        if not recent_requests:
            return {node.id: [0.0] * 24 for node in nodes}  # 24-hour pattern
            
        patterns = {}
        
        for node in nodes:
            node_requests = [req for req in recent_requests if req["selected_node"] == node.id]
            pattern = self._generate_hourly_pattern(node_requests, current_time)
            patterns[node.id] = pattern
            
        self._pattern_cache[cache_key] = patterns
        return patterns
        
    def _generate_hourly_pattern(
        self, 
        requests: List[Dict[str, Any]], 
        current_time: float
    ) -> List[float]:
        """Generate 24-hour interference pattern."""
        
        hourly_counts = [0] * 24
        
        for req in requests:
            # Convert timestamp to hour of day
            req_time = req["timestamp"]
            hour = int((req_time % 86400) // 3600)  # 86400 seconds per day
            hourly_counts[hour] += 1
            
        # Normalize to create interference pattern
        max_count = max(hourly_counts) if hourly_counts else 1
        normalized_pattern = [count / max_count for count in hourly_counts]
        
        # Apply quantum interference formula
        interference_pattern = []
        for i, amplitude in enumerate(normalized_pattern):
            # Add wave-like interference
            wave_component = 0.1 * math.sin(2 * math.pi * i / 24)
            interference_value = amplitude + wave_component
            interference_pattern.append(max(0.0, interference_value))
            
        return interference_pattern
        
    def predict_future_load(
        self, 
        node_id: str, 
        hours_ahead: float = 1.0
    ) -> float:
        """Predict future load based on interference patterns."""
        
        patterns = self.calculate_interference_pattern([])
        if node_id not in patterns:
            return 0.5  # Default prediction
            
        pattern = patterns[node_id]
        current_hour = int(time.time() % 86400 // 3600)
        future_hour = int((current_hour + hours_ahead) % 24)
        
        return pattern[future_hour]


class QuantumAnnealingOptimizer:
    """Quantum annealing optimizer for load balancing decisions."""
    
    def __init__(self):
        self._temperature = 1.0
        self._cooling_rate = 0.95
        self._min_temperature = 0.01
        
    def optimize_server_selection(
        self,
        nodes: List[ServerNode],
        request: RequestContext,
        constraints: Dict[str, Any],
    ) -> Tuple[ServerNode, float]:
        """Optimize server selection using quantum annealing."""
        
        if not nodes:
            raise ValueError("No nodes available for optimization")
            
        current_solution = random.choice(nodes)
        current_energy = self._calculate_energy(current_solution, request, constraints)
        
        best_solution = current_solution
        best_energy = current_energy
        
        temperature = self._temperature
        
        # Annealing iterations
        for iteration in range(100):
            # Generate neighbor solution
            neighbor = self._generate_neighbor(nodes, current_solution)
            neighbor_energy = self._calculate_energy(neighbor, request, constraints)
            
            # Accept or reject based on temperature
            energy_diff = neighbor_energy - current_energy
            
            if energy_diff < 0 or random.random() < math.exp(-energy_diff / temperature):
                current_solution = neighbor
                current_energy = neighbor_energy
                
                # Update best solution
                if neighbor_energy < best_energy:
                    best_solution = neighbor
                    best_energy = neighbor_energy
                    
            # Cool down
            temperature *= self._cooling_rate
            temperature = max(temperature, self._min_temperature)
            
        return best_solution, best_energy
        
    def _calculate_energy(
        self, 
        node: ServerNode, 
        request: RequestContext, 
        constraints: Dict[str, Any]
    ) -> float:
        """Calculate energy (cost) for a node selection."""
        
        # Load balancing energy
        load_ratio = node.current_connections / max(node.max_connections, 1)
        load_energy = load_ratio ** 2
        
        # Performance energy
        perf_energy = (node.cpu_usage + node.memory_usage) / 2
        
        # Health energy (lower health = higher energy)
        health_energy = 1.0 - node.health_score
        
        # Affinity energy (lower if node matches affinity hints)
        affinity_energy = 0.0
        if request.affinity_hints and node.id not in request.affinity_hints:
            affinity_energy = 0.3
            
        # Resource constraint energy
        constraint_energy = 0.0
        for resource, requirement in request.resource_requirements.items():
            if resource == "cpu" and node.cpu_usage > (1.0 - requirement):
                constraint_energy += 0.5
            elif resource == "memory" and node.memory_usage > (1.0 - requirement):
                constraint_energy += 0.5
                
        # Total energy
        total_energy = (
            load_energy * 0.3 +
            perf_energy * 0.2 +
            health_energy * 0.2 +
            affinity_energy * 0.2 +
            constraint_energy * 0.1
        )
        
        return total_energy
        
    def _generate_neighbor(self, nodes: List[ServerNode], current: ServerNode) -> ServerNode:
        """Generate a neighbor solution for annealing."""
        
        # Simple neighbor: random selection from available nodes
        available_nodes = [n for n in nodes if n.id != current.id]
        
        if not available_nodes:
            return current
            
        return random.choice(available_nodes)


class QuantumLoadBalancer:
    """Quantum-inspired load balancer with advanced algorithms."""
    
    def __init__(self):
        self._nodes: Dict[str, ServerNode] = {}
        self._amplitude_calculator = QuantumAmplitudeCalculator()
        self._entanglement_manager = EntanglementManager()
        self._interference_analyzer = InterferencePatternAnalyzer()
        self._annealing_optimizer = QuantumAnnealingOptimizer()
        
        # Statistics
        self._request_count = 0
        self._selection_history: deque = deque(maxlen=1000)
        self._performance_metrics: Dict[str, float] = defaultdict(float)
        
    def add_server_node(self, node: ServerNode) -> None:
        """Add a server node to the load balancer."""
        
        self._nodes[node.id] = node
        logger.info("Server node added", node_id=node.id, address=f"{node.address}:{node.port}")
        
        # Create entanglements with existing nodes if beneficial
        self._maybe_create_entanglements(node)
        
    def remove_server_node(self, node_id: str) -> bool:
        """Remove a server node from the load balancer."""
        
        if node_id not in self._nodes:
            return False
            
        node = self._nodes[node_id]
        
        # Remove entanglements
        for entangled_id in node.entangled_nodes:
            entangled_node = self._nodes.get(entangled_id)
            if entangled_node and node_id in entangled_node.entangled_nodes:
                entangled_node.entangled_nodes.remove(node_id)
                
        del self._nodes[node_id]
        logger.info("Server node removed", node_id=node_id)
        return True
        
    def update_server_metrics(
        self, 
        node_id: str, 
        cpu_usage: float,
        memory_usage: float,
        current_connections: int,
        response_time: float,
    ) -> None:
        """Update server metrics for load balancing decisions."""
        
        if node_id not in self._nodes:
            return
            
        node = self._nodes[node_id]
        node.cpu_usage = max(0.0, min(1.0, cpu_usage))
        node.memory_usage = max(0.0, min(1.0, memory_usage))
        node.current_connections = max(0, current_connections)
        node.response_time = max(0.0, response_time)
        node.last_updated = time.time()
        
        # Update health score based on metrics
        node.health_score = self._calculate_health_score(node)
        
        logger.debug(
            "Server metrics updated",
            node_id=node_id,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            connections=current_connections,
            health_score=node.health_score,
        )
        
    def _calculate_health_score(self, node: ServerNode) -> float:
        """Calculate health score for a server node."""
        
        # Base score from resource utilization
        resource_score = 1.0 - (node.cpu_usage + node.memory_usage) / 2
        
        # Connection load score
        load_ratio = node.current_connections / max(node.max_connections, 1)
        load_score = 1.0 - min(load_ratio, 1.0)
        
        # Response time score (assume baseline of 100ms)
        response_score = max(0.0, 1.0 - node.response_time / 1.0)
        
        # Weighted combination
        health_score = (resource_score * 0.4 + load_score * 0.4 + response_score * 0.2)
        
        return max(0.0, min(1.0, health_score))
        
    async def select_server(self, request: RequestContext) -> QuantumLoadBalancingResult:
        """Select optimal server using quantum-inspired algorithms."""
        
        if not self._nodes:
            raise ValueError("No server nodes available")
            
        self._request_count += 1
        available_nodes = [node for node in self._nodes.values() if node.health_score > 0.1]
        
        if not available_nodes:
            # Emergency fallback to any node
            available_nodes = list(self._nodes.values())
            
        # Step 1: Calculate quantum amplitudes
        amplitudes = {}
        global_state = {"total_connections": sum(n.current_connections for n in available_nodes)}
        
        for node in available_nodes:
            amplitude = self._amplitude_calculator.calculate_amplitude(node, request, global_state)
            amplitudes[node.id] = amplitude
            
        # Step 2: Apply entanglement effects
        entangled_amplitudes = self._apply_entanglement_effects(amplitudes, available_nodes)
        
        # Step 3: Calculate probability distribution
        probability_dist = self._calculate_probability_distribution(entangled_amplitudes)
        
        # Step 4: Use quantum annealing for final optimization
        constraints = {"max_cpu": 0.9, "max_memory": 0.9}
        optimized_node, energy = self._annealing_optimizer.optimize_server_selection(
            available_nodes, request, constraints
        )
        
        # Step 5: Calculate interference patterns
        interference_patterns = self._interference_analyzer.calculate_interference_pattern(available_nodes)
        current_hour = int(time.time() % 86400 // 3600)
        interference_values = [
            patterns.get(node.id, [0.0] * 24)[current_hour]
            for node, patterns in zip(available_nodes, [interference_patterns] * len(available_nodes))
        ]
        
        # Step 6: Final selection with quantum state determination
        selected_node, quantum_state = self._make_final_selection(
            optimized_node, probability_dist, entangled_amplitudes
        )
        
        # Step 7: Calculate decision confidence
        confidence = self._calculate_decision_confidence(selected_node, available_nodes, probability_dist)
        
        # Step 8: Record selection and update patterns
        self._record_selection(request, selected_node)
        self._interference_analyzer.add_request_data(request, selected_node.id)
        
        # Create result
        result = QuantumLoadBalancingResult(
            selected_node=selected_node,
            quantum_state=quantum_state,
            probability_distribution=probability_dist,
            entanglement_strength=self._get_max_entanglement_strength(selected_node),
            interference_pattern=interference_values,
            decision_confidence=confidence,
            metadata={
                "optimization_energy": energy,
                "total_nodes": len(available_nodes),
                "request_count": self._request_count,
            },
        )
        
        logger.info(
            "Server selected",
            selected_node=selected_node.id,
            quantum_state=quantum_state.value,
            confidence=confidence,
            request_id=request.request_id,
        )
        
        return result
        
    def _apply_entanglement_effects(
        self, 
        amplitudes: Dict[str, complex], 
        nodes: List[ServerNode]
    ) -> Dict[str, complex]:
        """Apply quantum entanglement effects to amplitudes."""
        
        entangled_amplitudes = amplitudes.copy()
        
        for node in nodes:
            if not node.entangled_nodes:
                continue
                
            node_amplitude = amplitudes.get(node.id, complex(0, 0))
            
            # Apply entanglement effects
            for entangled_id in node.entangled_nodes:
                if entangled_id not in amplitudes:
                    continue
                    
                entanglement_strength = self._entanglement_manager.get_entanglement_strength(
                    node.id, entangled_id
                )
                entangled_amplitude = amplitudes[entangled_id]
                
                # Quantum entanglement: correlated amplitudes
                correlation_factor = entanglement_strength * 0.5
                entangled_amplitudes[node.id] += correlation_factor * entangled_amplitude
                
        return entangled_amplitudes
        
    def _calculate_probability_distribution(
        self, 
        amplitudes: Dict[str, complex]
    ) -> Dict[str, float]:
        """Calculate probability distribution from quantum amplitudes."""
        
        # Calculate squared magnitudes (Born rule)
        probabilities = {}
        total_magnitude_squared = 0.0
        
        for node_id, amplitude in amplitudes.items():
            magnitude_squared = abs(amplitude) ** 2
            probabilities[node_id] = magnitude_squared
            total_magnitude_squared += magnitude_squared
            
        # Normalize probabilities
        if total_magnitude_squared > 0:
            for node_id in probabilities:
                probabilities[node_id] /= total_magnitude_squared
        else:
            # Equal probabilities as fallback
            equal_prob = 1.0 / len(amplitudes) if amplitudes else 0.0
            probabilities = {node_id: equal_prob for node_id in amplitudes}
            
        return probabilities
        
    def _make_final_selection(
        self,
        optimized_node: ServerNode,
        probability_dist: Dict[str, float],
        amplitudes: Dict[str, complex],
    ) -> Tuple[ServerNode, QuantumState]:
        """Make final server selection and determine quantum state."""
        
        # Determine quantum state based on selection method
        max_prob_node_id = max(probability_dist, key=probability_dist.get)
        max_prob = probability_dist[max_prob_node_id]
        
        if max_prob > 0.7:
            # High probability = collapsed state
            quantum_state = QuantumState.COLLAPSED
            selected_node = self._nodes[max_prob_node_id]
        elif optimized_node.entangled_nodes:
            # Has entanglements = entangled state
            quantum_state = QuantumState.ENTANGLED
            selected_node = optimized_node
        elif max_prob < 0.3:
            # Low max probability = superposition state
            quantum_state = QuantumState.SUPERPOSITION
            # Use weighted random selection
            selected_node = self._weighted_random_selection(probability_dist)
        else:
            # Interference patterns detected
            quantum_state = QuantumState.INTERFERENCE
            selected_node = optimized_node
            
        return selected_node, quantum_state
        
    def _weighted_random_selection(self, probability_dist: Dict[str, float]) -> ServerNode:
        """Perform weighted random selection based on probability distribution."""
        
        rand_val = random.random()
        cumulative_prob = 0.0
        
        for node_id, prob in probability_dist.items():
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return self._nodes[node_id]
                
        # Fallback to first node
        return self._nodes[next(iter(probability_dist.keys()))]
        
    def _calculate_decision_confidence(
        self,
        selected_node: ServerNode,
        available_nodes: List[ServerNode],
        probability_dist: Dict[str, float],
    ) -> float:
        """Calculate confidence in the selection decision."""
        
        selected_prob = probability_dist.get(selected_node.id, 0.0)
        
        # Confidence based on probability dominance
        prob_confidence = selected_prob
        
        # Health score confidence
        health_confidence = selected_node.health_score
        
        # Load balancing confidence (lower load = higher confidence)
        load_ratio = selected_node.current_connections / max(selected_node.max_connections, 1)
        load_confidence = 1.0 - min(load_ratio, 1.0)
        
        # Combined confidence
        total_confidence = (
            prob_confidence * 0.4 +
            health_confidence * 0.3 +
            load_confidence * 0.3
        )
        
        return max(0.0, min(1.0, total_confidence))
        
    def _get_max_entanglement_strength(self, node: ServerNode) -> float:
        """Get maximum entanglement strength for a node."""
        
        max_strength = 0.0
        
        for entangled_id in node.entangled_nodes:
            strength = self._entanglement_manager.get_entanglement_strength(node.id, entangled_id)
            max_strength = max(max_strength, strength)
            
        return max_strength
        
    def _record_selection(self, request: RequestContext, selected_node: ServerNode) -> None:
        """Record selection for analytics and pattern learning."""
        
        selection_record = {
            "timestamp": time.time(),
            "request_id": request.request_id,
            "client_id": request.client_id,
            "selected_node": selected_node.id,
            "node_health": selected_node.health_score,
            "node_load": selected_node.current_connections / max(selected_node.max_connections, 1),
        }
        
        self._selection_history.append(selection_record)
        
    def _maybe_create_entanglements(self, new_node: ServerNode) -> None:
        """Create entanglements for new node if beneficial."""
        
        if len(self._nodes) < 2:
            return  # Need at least 2 nodes for entanglement
            
        # Find nodes with similar characteristics for entanglement
        for existing_node in self._nodes.values():
            if existing_node.id == new_node.id:
                continue
                
            # Calculate similarity score
            similarity = self._calculate_node_similarity(new_node, existing_node)
            
            if similarity > 0.7:  # High similarity threshold
                self._entanglement_manager.create_entanglement(
                    new_node, existing_node, similarity * 0.8
                )
                
    def _calculate_node_similarity(self, node1: ServerNode, node2: ServerNode) -> float:
        """Calculate similarity between two nodes."""
        
        # Capacity similarity
        capacity_diff = abs(node1.max_connections - node2.max_connections)
        capacity_similarity = 1.0 - min(capacity_diff / max(node1.max_connections, node2.max_connections, 1), 1.0)
        
        # Performance similarity
        perf_diff = abs(node1.response_time - node2.response_time)
        perf_similarity = 1.0 - min(perf_diff / max(node1.response_time, node2.response_time, 0.1), 1.0)
        
        # Weight similarity
        weight_diff = abs(node1.weight - node2.weight)
        weight_similarity = 1.0 - min(weight_diff / max(node1.weight, node2.weight, 0.1), 1.0)
        
        # Combined similarity
        overall_similarity = (
            capacity_similarity * 0.4 +
            perf_similarity * 0.3 +
            weight_similarity * 0.3
        )
        
        return overall_similarity
        
    async def get_cluster_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cluster statistics."""
        
        total_nodes = len(self._nodes)
        healthy_nodes = sum(1 for node in self._nodes.values() if node.health_score > 0.5)
        total_connections = sum(node.current_connections for node in self._nodes.values())
        total_capacity = sum(node.max_connections for node in self._nodes.values())
        
        # Calculate average metrics
        if self._nodes:
            avg_cpu = sum(node.cpu_usage for node in self._nodes.values()) / total_nodes
            avg_memory = sum(node.memory_usage for node in self._nodes.values()) / total_nodes
            avg_response_time = sum(node.response_time for node in self._nodes.values()) / total_nodes
            avg_health = sum(node.health_score for node in self._nodes.values()) / total_nodes
        else:
            avg_cpu = avg_memory = avg_response_time = avg_health = 0.0
            
        # Entanglement statistics
        total_entanglements = len(self._entanglement_manager._entanglement_matrix) // 2
        
        return {
            "cluster_health": {
                "total_nodes": total_nodes,
                "healthy_nodes": healthy_nodes,
                "health_ratio": healthy_nodes / total_nodes if total_nodes > 0 else 0.0,
                "avg_health_score": avg_health,
            },
            "capacity": {
                "total_capacity": total_capacity,
                "current_connections": total_connections,
                "utilization_ratio": total_connections / total_capacity if total_capacity > 0 else 0.0,
            },
            "performance": {
                "avg_cpu_usage": avg_cpu,
                "avg_memory_usage": avg_memory,
                "avg_response_time": avg_response_time,
            },
            "quantum_metrics": {
                "total_entanglements": total_entanglements,
                "entanglement_density": total_entanglements / max(total_nodes, 1),
                "request_count": self._request_count,
            },
            "recent_selections": [
                {
                    "node": record["selected_node"],
                    "health": record["node_health"],
                    "load": record["node_load"],
                    "timestamp": record["timestamp"],
                }
                for record in list(self._selection_history)[-10:]  # Last 10 selections
            ],
        }


def create_quantum_load_balancer() -> QuantumLoadBalancer:
    """Create a quantum load balancer with default configuration."""
    
    load_balancer = QuantumLoadBalancer()
    
    logger.info("Quantum load balancer created")
    
    return load_balancer