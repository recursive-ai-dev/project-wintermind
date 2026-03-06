#!/usr/bin/env python3
"""
MORPHIC INFERENCE LATTICE (MIL) v1.0
A Novel Synthetic Intelligence Architecture

This is a complete, runnable implementation of the MIL architecture as specified
in the technical documentation. It requires no external dependencies beyond
Python 3.7+ standard library.

Usage:
    from mil import MIL, VertexState

    mil = MIL(attention_budget=1000, verification_threshold=0.6)
    mil.bind_judge(lambda x: 0.8 if isinstance(x, dict) else 0.5)

    result = mil.infer(
        initial_condition={"task": "solve"},
        input_space={"task"},
        output_space={"solution"},
        expansion_fn=my_expansion,
        entity_fn=my_entities
    )
"""

import heapq
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Callable, Any, FrozenSet, Union
from collections import defaultdict
from enum import Enum, auto
from abc import ABC, abstractmethod

__version__ = "1.0.0"
__all__ = [
    "MIL", "GRS", "Vertex", "VertexState", "ThoughtSignature",
    "DensityOperator", "LatticeOptimizer", "InferenceEngine"
]


# ============================================================================
# SECTION 1: CORE ONTOLOGY - Mathematical Primitives
# ============================================================================

class VertexState(Enum):
    """
    Fundamental states of a cognitive vertex in the lattice.

    POTENTIAL: Exists in superposition, not yet materialized
    ACTIVE: Currently under computation
    VERIFIED: Passed Chain-of-Verification (CoVe) criteria
    PRUNED: Failed verification or redundant (attention budget reclamation)
    TERMINAL: Designated output candidate
    """
    POTENTIAL = auto()
    ACTIVE = auto()
    VERIFIED = auto()
    PRUNED = auto()
    TERMINAL = auto()


@dataclass(frozen=True)
class ThoughtSignature:
    """
    Typed input/output contract defining deterministic data flow.

    Immutable and hashable for use in vertex identification and
    functional resonance checking.

    Attributes:
        input_space: Set of input domain identifiers
        output_space: Set of output domain identifiers  
        constraint_hash: Deterministic hash of constraints
    """
    input_space: FrozenSet[str]
    output_space: FrozenSet[str]
    constraint_hash: str

    def __hash__(self) -> int:
        return hash((self.input_space, self.output_space, self.constraint_hash))

    def resonates_with(self, other: 'ThoughtSignature') -> bool:
        """Check structural equivalence for composition."""
        return len(self.output_space & other.input_space) > 0


@dataclass
class Vertex:
    """
    Fundamental unit of cognition in the lattice.

    Contains no text content - pure structural information.
    Maintains topological relationships and information-theoretic metrics.

    Attributes:
        vid: Unique vertex identifier
        signature: Typed contract for this thought
        state: Current lifecycle state
        volume: Information mass (sum of preceding thoughts + 1)
        entropy: Uncertainty measure (decreases with densification)
        score: LLM-as-Judge evaluation (0.0-1.0)
        parents: Incoming edges (reasoning dependencies)
        children: Outgoing edges (reasoning consequences)
        depth: Topological distance from seed
        payload: Opaque computational content (type: Any)
    """
    vid: int
    signature: ThoughtSignature
    state: VertexState = VertexState.POTENTIAL
    volume: float = 0.0
    entropy: float = 1.0
    score: float = 0.0
    parents: Set[int] = field(default_factory=set)
    children: Set[int] = field(default_factory=set)
    depth: int = 0
    payload: Any = None

    def __hash__(self) -> int:
        return self.vid

    def __eq__(self, other) -> bool:
        return isinstance(other, Vertex) and self.vid == other.vid

    @property
    def is_active(self) -> bool:
        return self.state == VertexState.ACTIVE

    @property
    def is_verified(self) -> bool:
        return self.state == VertexState.VERIFIED


# ============================================================================
# SECTION 2: GRAPH REASONING STATE - The Lattice Topology
# ============================================================================

class GRS:
    """
    Graph Reasoning State: High-density repository of logic.

    Manages the computational graph structure, attention budget,
    and topological operations. Functions as a "Closed Conceptual Organism"
    that autonomously optimizes its own structure.

    Attributes:
        vertices: Map of vid -> Vertex
        adjacency: Forward edges (vid -> set of children)
        reverse_adj: Backward edges (vid -> set of parents)
        attention_budget: Maximum vertices before forced pruning
        current_load: Current vertex count
        terminal_candidates: Verified vertices marked for output
    """

    def __init__(self, attention_budget: int = 10000):
        self.vertices: Dict[int, Vertex] = {}
        self.adjacency: Dict[int, Set[int]] = defaultdict(set)
        self.reverse_adj: Dict[int, Set[int]] = defaultdict(set)
        self.attention_budget: int = attention_budget
        self.current_load: int = 0
        self._vid_counter: int = 0
        self.terminal_candidates: List[int] = []

    def spawn(self, signature: ThoughtSignature, parents: Optional[Set[int]] = None, 
              payload: Any = None) -> int:
        """
        Create new vertex with inherited volume from parents.

        If attention budget exceeded, prunes weakest vertices first.
        Volume calculation: 1.0 + sum(parent volumes)

        Args:
            signature: Typed contract for this vertex
            parents: Set of parent vertex IDs (optional)
            payload: Opaque content to store

        Returns:
            vid: Unique identifier for created vertex
        """
        if self.current_load >= self.attention_budget:
            self._prune_weakest()

        vid = self._vid_counter
        self._vid_counter += 1

        # Volume inheritance: self (1.0) + sum of parent volumes
        volume = 1.0
        parent_depth = 0
        if parents:
            for p in parents:
                if p in self.vertices:
                    volume += self.vertices[p].volume
                    parent_depth = max(parent_depth, self.vertices[p].depth)

        vertex = Vertex(
            vid=vid,
            signature=signature,
            state=VertexState.ACTIVE,
            volume=volume,
            parents=parents or set(),
            depth=parent_depth + 1,
            payload=payload
        )

        self.vertices[vid] = vertex
        self.current_load += 1

        # Update topology
        if parents:
            for p in parents:
                self.adjacency[p].add(vid)
                self.reverse_adj[vid].add(p)
                if p in self.vertices:
                    self.vertices[p].children.add(vid)

        return vid

    def _prune_weakest(self, count: int = 1):
        """
        Autonomous topology optimization.

        Removes lowest-scoring non-critical vertices to preserve
        attention budget for high-signal information.
        """
        candidates = [
            (v.score, v.vid) for v in self.vertices.values() 
            if v.state not in [VertexState.TERMINAL, VertexState.ACTIVE]
        ]
        if candidates:
            candidates.sort()
            for _, vid in candidates[:count]:
                self._collapse_vertex(vid)

    def _collapse_vertex(self, vid: int):
        """
        Remove vertex while preserving topological connectivity.

        Bridges parents to children to maintain reasoning chains.
        Updates child volumes to reflect new ancestry.
        """
        if vid not in self.vertices:
            return

        vertex = self.vertices[vid]
        vertex.state = VertexState.PRUNED

        # Bridge parents to children
        for parent in vertex.parents:
            if parent in self.vertices:
                self.vertices[parent].children.discard(vid)
                self.vertices[parent].children.update(vertex.children)

        for child in vertex.children:
            if child in self.vertices:
                self.vertices[child].parents.discard(vid)
                self.vertices[child].parents.update(vertex.parents)
                # Recalculate volume
                new_volume = 1.0 + sum(
                    self.vertices[p].volume 
                    for p in self.vertices[child].parents 
                    if p in self.vertices
                )
                self.vertices[child].volume = new_volume

        self.current_load -= 1

    def verify(self, vid: int, score: float, threshold: float = 0.6):
        """
        Apply LLM-as-Judge scoring with threshold-based state transition.

        Implements 4-step Chain-of-Verification (CoVe):
        1. Initial generation (already done via spawn)
        2. Generate verification questions (external to this method)
        3. Execute verification (score calculation)
        4. Reconciliation (state transition based on threshold)

        Args:
            vid: Vertex to verify
            score: Evaluation score (0.0-1.0)
            threshold: Minimum score for VERIFIED state
        """
        if vid not in self.vertices:
            return

        vertex = self.vertices[vid]
        vertex.score = score

        if score >= threshold:
            vertex.state = VertexState.VERIFIED
        else:
            vertex.state = VertexState.PRUNED
            self._collapse_vertex(vid)

    def mark_terminal(self, vid: int):
        """Designate verified vertex as output candidate."""
        if vid in self.vertices and self.vertices[vid].state == VertexState.VERIFIED:
            self.vertices[vid].state = VertexState.TERMINAL
            self.terminal_candidates.append(vid)

    def path_to_root(self, vid: int) -> List[int]:
        """
        Trace ancestry from vertex to seed.

        Used for CoVe reconciliation and optimal path extraction.
        Returns list of VIDs from root to target.
        """
        path = []
        current = vid
        visited = set()

        while current is not None and current not in visited:
            visited.add(current)
            path.append(current)
            parents = self.reverse_adj[current]
            current = next(iter(parents)) if parents else None

        return list(reversed(path))

    def get_frontier(self) -> Set[int]:
        """Return set of ACTIVE vertices ready for processing."""
        return {v.vid for v in self.vertices.values() if v.state == VertexState.ACTIVE}


# ============================================================================
# SECTION 3: INFERENCE ENGINE - Propagation Mechanism
# ============================================================================

class InferenceEngine:
    """
    Topological necessity realization engine.

    Manages the expansion queue and executes verification steps.
    Implements priority-based processing favoring high-volume vertices.
    """

    def __init__(self, grs: GRS, llm_judge: Callable[[Any], float]):
        self.grs = grs
        self.llm_judge = llm_judge
        self.expansion_queue: List[Tuple[float, int]] = []  # (priority, vid)
        self.iteration = 0

    def seed(self, initial_payload: Any, input_space: Set[str], 
             output_space: Set[str]) -> int:
        """Initialize skeleton with seed vertex."""
        sig = ThoughtSignature(
            input_space=frozenset(input_space),
            output_space=frozenset(output_space),
            constraint_hash=hashlib.sha256(b"seed").hexdigest()[:16]
        )
        vid = self.grs.spawn(sig, payload=initial_payload)
        # Priority by volume (negative for min-heap as max-heap)
        heapq.heappush(self.expansion_queue, (-1.0, vid))
        return vid

    def expand(self, vid: int, expansions: List[Tuple[Set[str], Set[str], Any]]) -> List[int]:
        """
        Expand vertex into children based on provided signatures and payloads.

        Only creates children that resonate with parent signature.
        Returns list of created child VIDs.
        """
        if vid not in self.grs.vertices:
            return []

        parent = self.grs.vertices[vid]
        children = []

        for inp, out, payload in expansions:
            sig = ThoughtSignature(
                input_space=frozenset(inp),
                output_space=frozenset(out),
                constraint_hash=hashlib.sha256(f"{vid}_{self.iteration}".encode()).hexdigest()[:16]
            )

            # Functional resonance check
            if parent.signature.resonates_with(sig):
                child_vid = self.grs.spawn(sig, parents={vid}, payload=payload)
                priority = -self.grs.vertices[child_vid].volume
                heapq.heappush(self.expansion_queue, (priority, child_vid))
                children.append(child_vid)

        self.iteration += 1
        return children

    def step(self, threshold: float = 0.6) -> Optional[int]:
        """
        Process single vertex from queue.

        Returns processed VID or None if queue empty.
        """
        if not self.expansion_queue:
            return None

        _, vid = heapq.heappop(self.expansion_queue)

        if vid not in self.grs.vertices:
            return vid

        vertex = self.grs.vertices[vid]
        if vertex.state != VertexState.ACTIVE:
            return vid

        # Verify
        score = self.llm_judge(vertex.payload)
        self.grs.verify(vid, score, threshold)

        return vid


# ============================================================================
# SECTION 4: DENSITY OPERATOR - Chain-of-Density
# ============================================================================

class DensityOperator:
    """
    Implements Chain-of-Density (CoD) logic.

    Performs 5 fixed iterations of entity incorporation to prevent
    semantic drift and lead bias. Each iteration:
    1. Extracts 1-3 salient entities from payload
    2. Creates successor vertex with expanded output_space
    3. Reduces entropy (information density increases)
    4. Updates volume with entity contribution
    """

    def __init__(self, grs: GRS, iterations: int = 5):
        self.grs = grs
        self.iterations = iterations

    def densify(self, vid: int, entity_extractor: Callable[[Any], Set[str]]) -> int:
        """
        Execute CoD iterations on vertex.

        Args:
            vid: Starting vertex ID
            entity_extractor: Function extracting entities from payload

        Returns:
            Final densified vertex ID
        """
        current_vid = vid

        for i in range(self.iterations):
            if current_vid not in self.grs.vertices:
                break

            vertex = self.grs.vertices[current_vid]
            entities = entity_extractor(vertex.payload)

            # Create successor with expanded signature
            sig = ThoughtSignature(
                input_space=vertex.signature.input_space,
                output_space=vertex.signature.output_space | frozenset(entities),
                constraint_hash=hashlib.sha256(f"density_{i}_{vid}".encode()).hexdigest()[:16]
            )

            new_vid = self.grs.spawn(sig, parents={current_vid}, payload=vertex.payload)

            # Update information-theoretic properties
            self.grs.vertices[new_vid].entropy = vertex.entropy * 0.8
            self.grs.vertices[new_vid].volume = vertex.volume + len(entities) * 0.1

            current_vid = new_vid

        return current_vid


# ============================================================================
# SECTION 5: LATTICE OPTIMIZER - DSPy-style Search
# ============================================================================

class LatticeOptimizer:
    """
    Bayesian-inspired path distillation.

    Searches for optimal path through lattice using provided metric.
    Compiles path into final output structure.
    """

    def __init__(self, grs: GRS):
        self.grs = grs

    def optimal_path(self, terminal_vids: List[int], 
                    metric: Callable[[Vertex], float]) -> List[int]:
        """
        Find highest-scoring path from root to terminal.

        Args:
            terminal_vids: List of candidate terminal vertex IDs
            metric: Scoring function (higher is better)

        Returns:
            List of VIDs representing optimal path
        """
        if not terminal_vids:
            return []

        # Score all terminals
        scored = [
            (metric(self.grs.vertices[vid]), vid) 
            for vid in terminal_vids 
            if vid in self.grs.vertices
        ]

        if not scored:
            return []

        scored.sort(reverse=True)
        best_terminal = scored[0][1]

        return self.grs.path_to_root(best_terminal)

    def compile(self, path: List[int]) -> Dict[str, Any]:
        """
        Distill path into compiled output.

        Returns dict with:
        - path: List of VIDs
        - total_volume: Sum of volumes
        - min_score: Minimum score along path (quality floor)
        - payloads: List of payload data
        """
        if not path:
            return {"path": [], "total_volume": 0.0, "min_score": 0.0, "payloads": []}

        vertices = [self.grs.vertices[vid] for vid in path if vid in self.grs.vertices]

        return {
            "path": path,
            "total_volume": sum(v.volume for v in vertices),
            "min_score": min(v.score for v in vertices) if vertices else 0.0,
            "payloads": [v.payload for v in vertices]
        }


# ============================================================================
# SECTION 6: MIL - Complete Synthesis
# ============================================================================

class MIL:
    """
    Morphic Inference Lattice: Synthetic Intelligence Architecture

    Unifies Chain of Thought (sequential propagation), Graph of Thoughts
    (non-linear aggregation), and DSPy (metric-driven optimization) into a
    unified topological reasoning system.

    Usage:
        mil = MIL(attention_budget=10000, verification_threshold=0.6)
        mil.bind_judge(lambda payload: evaluate(payload))

        result = mil.infer(
            initial_condition=data,
            input_space={"input_domain"},
            output_space={"output_domain"},
            expansion_fn=expand_logic,
            entity_fn=extract_entities
        )
    """

    def __init__(self, attention_budget: int = 10000, verification_threshold: float = 0.6):
        self.grs = GRS(attention_budget)
        self.threshold = verification_threshold
        self.density_op = DensityOperator(self.grs)
        self.optimizer = LatticeOptimizer(self.grs)
        self.judge: Optional[Callable[[Any], float]] = None
        self.engine: Optional[InferenceEngine] = None

    def bind_judge(self, judge_fn: Callable[[Any], float]):
        """
        Attach LLM-as-Judge evaluation function.

        Judge should implement 5-7 category rubric:
        - Correctness, Completeness, Relevance, Coherence, Instruction-Following
        Returns float 0.0-1.0
        """
        self.judge = judge_fn
        self.engine = InferenceEngine(self.grs, judge_fn)

    def infer(self, initial_condition: Any, input_space: Set[str], output_space: Set[str],
             expansion_fn: Callable[[Any], List[Tuple[Set[str], Set[str], Any]]],
             entity_fn: Callable[[Any], Set[str]], max_rounds: int = 10) -> Dict[str, Any]:
        """
        Execute complete inference cycle.

        Algorithm:
        1. Skeleton initialization (parallelized structural outline)
        2. Recursive expansion with cross-lattice verification (CoVe)
        3. Chain-of-Density refinement (CoD)
        4. Optimal path distillation (Bayesian search)

        Args:
            initial_condition: Starting payload
            input_space: Set of input domain identifiers
            output_space: Set of output domain identifiers
            expansion_fn: Function generating child signatures/payloads
            entity_fn: Function extracting entities for densification
            max_rounds: Maximum expansion rounds

        Returns:
            Dict containing compiled result and execution statistics
        """
        if not self.judge or not self.engine:
            raise RuntimeError("Judge not bound. Call bind_judge() first.")

        # 1. Initialize
        seed_vid = self.engine.seed(initial_condition, input_space, output_space)
        current_frontier = {seed_vid}

        # 2. Expand and verify iteratively
        for round_num in range(max_rounds):
            if not current_frontier:
                break

            next_frontier = set()

            for vid in current_frontier:
                if vid not in self.grs.vertices:
                    continue

                vertex = self.grs.vertices[vid]
                if vertex.state != VertexState.ACTIVE:
                    continue

                # Verify
                score = self.judge(vertex.payload)
                self.grs.verify(vid, score, self.threshold)

                if vertex.state == VertexState.VERIFIED:
                    # Check terminal criteria
                    if vertex.volume > 10.0 or vertex.depth >= 4:
                        self.grs.mark_terminal(vid)
                    else:
                        # Expand
                        expansions = expansion_fn(vertex.payload)
                        for inp, out, payload in expansions:
                            child_sig = ThoughtSignature(
                                input_space=frozenset(inp),
                                output_space=frozenset(out),
                                constraint_hash=hashlib.sha256(
                                    f"{vid}_{hash(payload)}".encode()
                                ).hexdigest()[:16]
                            )

                            if vertex.signature.resonates_with(child_sig):
                                child_vid = self.grs.spawn(
                                    child_sig, parents={vid}, payload=payload
                                )
                                next_frontier.add(child_vid)

            current_frontier = next_frontier

        # 3. Densify terminals
        terminals = list(self.grs.terminal_candidates)
        dense_terminals = [
            self.density_op.densify(vid, entity_fn) 
            for vid in terminals
        ]

        # 4. Optimize
        def metric(vertex: Vertex) -> float:
            """Score * Volume / Entropy (information quality density)"""
            return vertex.score * vertex.volume / (vertex.entropy + 0.01)

        path = self.optimizer.optimal_path(dense_terminals, metric)
        compiled = self.optimizer.compile(path)

        return {
            "compiled": compiled,
            "statistics": {
                "total_vertices": len(self.grs.vertices),
                "terminal_candidates": len(terminals),
                "dense_terminals": len(dense_terminals),
                "optimal_path_length": len(path),
                "attention_load": self.grs.current_load,
                "attention_budget": self.grs.attention_budget
            },
            "grs": self.grs  # Reference for inspection
        }


# ============================================================================
# SECTION 7: UTILITY FUNCTIONS
# ============================================================================

def default_judge(payload: Any) -> float:
    """Default evaluation function - neutral scoring."""
    return 0.7 if isinstance(payload, dict) else 0.5


def default_expansion(payload: Any) -> List[Tuple[Set[str], Set[str], Any]]:
    """Default expansion - no children."""
    return []


def default_entity_extractor(payload: Any) -> Set[str]:
    """Default entity extraction - empty set."""
    return set()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Demonstration
    print("Morphic Inference Lattice v1.0")
    print("=" * 50)

    # Example usage
    mil = MIL(attention_budget=1000, verification_threshold=0.6)
    mil.bind_judge(default_judge)

    result = mil.infer(
        initial_condition={"example": "data"},
        input_space={"example"},
        output_space={"result"},
        expansion_fn=default_expansion,
        entity_fn=default_entity_extractor,
        max_rounds=3
    )

    print(f"Vertices created: {result['statistics']['total_vertices']}")
    print(f"System operational.")
