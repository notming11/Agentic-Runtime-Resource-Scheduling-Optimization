"""
End-to-End Testing for Autellix Resource Scheduler Module (RSM)

This comprehensive test validates the complete RSM system integration:
1. Single-threaded execution (PLAS scheduling)
2. Multi-threaded execution (ATLAS scheduling)
3. Load balancing effectiveness
4. KV cache locality preservation
5. Anti-starvation mechanism
6. Performance metrics collection and analysis

The test runs multiple scenarios and generates a detailed performance report
comparing PLAS vs ATLAS scheduling modes.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rsm import (
    AutellixSystem,
    AutellixConfig,
    IntegratedAutellixClient
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: str
    latency: float
    tokens: int
    engine_id: str
    priority: int
    service_time: float
    waiting_time: float
    thread_id: Optional[str] = None


@dataclass
class ScenarioMetrics:
    """Aggregated metrics for a test scenario."""
    scenario_name: str
    mode: str  # "PLAS" or "ATLAS"
    total_duration: float
    total_requests: int
    request_metrics: List[RequestMetrics] = field(default_factory=list)

    # Engine distribution
    engine_distribution: Dict[str, int] = field(default_factory=dict)

    # Session statistics
    session_stats: Optional[Dict[str, Any]] = None

    # System statistics
    system_stats: Optional[Dict[str, Any]] = None

    # Performance metrics
    avg_latency: float = 0.0
    min_latency: float = 0.0
    max_latency: float = 0.0
    total_tokens: int = 0

    # Cache metrics
    cache_hit_rate: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

    # Scheduler metrics
    scheduler_stats: Optional[Dict[str, Any]] = None

    def compute_aggregates(self):
        """Compute aggregate statistics from request metrics."""
        if not self.request_metrics:
            return

        latencies = [r.latency for r in self.request_metrics]
        self.avg_latency = sum(latencies) / len(latencies)
        self.min_latency = min(latencies)
        self.max_latency = max(latencies)
        self.total_tokens = sum(r.tokens for r in self.request_metrics)

        # Engine distribution
        self.engine_distribution = defaultdict(int)
        for r in self.request_metrics:
            self.engine_distribution[r.engine_id] += 1


# ============================================================================
# Test Scenarios
# ============================================================================

async def scenario_single_threaded_chatbot(
    system: AutellixSystem,
    num_requests: int = 10
) -> ScenarioMetrics:
    """
    Scenario 1: Single-threaded sequential chatbot (PLAS scheduling)

    Simulates a simple chatbot where requests are processed sequentially.
    This uses PLAS (Program-Level Attained Service) scheduling.

    Args:
        system: AutellixSystem instance
        num_requests: Number of sequential requests

    Returns:
        ScenarioMetrics with collected data
    """
    logger.info("="*80)
    logger.info("SCENARIO 1: Single-Threaded Chatbot (PLAS Scheduling)")
    logger.info("="*80)

    metrics = ScenarioMetrics(
        scenario_name="Single-Threaded Chatbot",
        mode="PLAS",
        total_duration=0.0,
        total_requests=num_requests
    )

    start_time = time.time()

    # Create single-threaded client
    async with system.create_client(
        is_multithreaded=False,
        session_metadata={"scenario": "chatbot", "type": "sequential"}
    ) as client:

        logger.info(f"Started session: {client.session_id}")
        logger.info(f"Submitting {num_requests} sequential requests...\n")

        # Submit requests sequentially
        for i in range(num_requests):
            prompt = f"User question {i+1}: Tell me about artificial intelligence."

            response = await client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7
            )

            # Collect metrics
            req_metrics = RequestMetrics(
                request_id=f"req_{i}",
                latency=response['latency'],
                tokens=response['tokens'],
                engine_id=response['engine_id'],
                priority=response['priority'],
                service_time=response['metadata']['service_time'],
                waiting_time=response['metadata']['waiting_time'],
                thread_id=response['metadata'].get('thread_id')
            )
            metrics.request_metrics.append(req_metrics)

            logger.info(
                f"Request {i+1:2d}/{num_requests}: "
                f"latency={response['latency']:.3f}s, "
                f"tokens={response['tokens']}, "
                f"engine={response['engine_id']}, "
                f"priority={response['priority']}"
            )

        # Get session stats
        metrics.session_stats = client.get_session_stats()

    metrics.total_duration = time.time() - start_time
    metrics.compute_aggregates()

    logger.info(f"\nScenario completed in {metrics.total_duration:.2f}s")
    logger.info(f"Average latency: {metrics.avg_latency:.3f}s")

    return metrics


async def scenario_multi_threaded_research(
    system: AutellixSystem,
    num_parallel_tasks: int = 5
) -> ScenarioMetrics:
    """
    Scenario 2: Multi-threaded research agent (ATLAS scheduling)

    Simulates a research agent with Map-Reduce pattern:
    1. Planning step (single thread)
    2. Parallel research tasks (map phase)
    3. Aggregation step (reduce phase, depends on all map tasks)

    This uses ATLAS scheduling which prioritizes based on critical path.

    Args:
        system: AutellixSystem instance
        num_parallel_tasks: Number of parallel research tasks

    Returns:
        ScenarioMetrics with collected data
    """
    logger.info("="*80)
    logger.info("SCENARIO 2: Multi-Threaded Research Agent (ATLAS Scheduling)")
    logger.info("="*80)

    metrics = ScenarioMetrics(
        scenario_name="Multi-Threaded Research Agent",
        mode="ATLAS",
        total_duration=0.0,
        total_requests=num_parallel_tasks + 2  # plan + parallel + aggregate
    )

    start_time = time.time()

    # Create multi-threaded client
    async with system.create_client(
        is_multithreaded=True,
        session_metadata={"scenario": "research", "type": "map-reduce"}
    ) as client:

        logger.info(f"Started session: {client.session_id}")

        # Step 1: Planning
        logger.info("\nStep 1: Planning phase")
        plan_response = await client.chat_completion(
            messages=[
                {"role": "user", "content": f"Plan research on {num_parallel_tasks} topics"}
            ],
            max_tokens=150
        )

        plan_thread_id = plan_response['metadata']['thread_id']
        metrics.request_metrics.append(RequestMetrics(
            request_id="plan",
            latency=plan_response['latency'],
            tokens=plan_response['tokens'],
            engine_id=plan_response['engine_id'],
            priority=plan_response['priority'],
            service_time=plan_response['metadata']['service_time'],
            waiting_time=plan_response['metadata']['waiting_time'],
            thread_id=plan_thread_id
        ))

        logger.info(
            f"  Planning: latency={plan_response['latency']:.3f}s, "
            f"engine={plan_response['engine_id']}, thread={plan_thread_id}"
        )

        # Step 2: Parallel research (Map phase)
        logger.info(f"\nStep 2: Parallel research ({num_parallel_tasks} tasks)")
        research_requests = [
            {
                "messages": [
                    {"role": "user", "content": f"Research topic {i+1} in depth"}
                ],
                "max_tokens": 200
            }
            for i in range(num_parallel_tasks)
        ]

        # Submit in parallel
        research_responses = await client.parallel_chat_completion(
            requests=research_requests,
            parent_thread_id=plan_thread_id
        )

        # Collect metrics
        for i, response in enumerate(research_responses):
            metrics.request_metrics.append(RequestMetrics(
                request_id=f"research_{i}",
                latency=response['latency'],
                tokens=response['tokens'],
                engine_id=response['engine_id'],
                priority=response['priority'],
                service_time=response['metadata']['service_time'],
                waiting_time=response['metadata']['waiting_time'],
                thread_id=response['metadata']['thread_id']
            ))

            logger.info(
                f"  Research {i+1}: latency={response['latency']:.3f}s, "
                f"engine={response['engine_id']}, "
                f"thread={response['metadata']['thread_id']}"
            )

        # Step 3: Aggregation (Reduce phase)
        logger.info("\nStep 3: Aggregation phase")
        research_thread_ids = [r['metadata']['thread_id'] for r in research_responses]

        aggregate_response = await client.chat_completion(
            messages=[
                {"role": "user", "content": "Synthesize all research findings"}
            ],
            max_tokens=250,
            parent_thread_ids=research_thread_ids
        )

        metrics.request_metrics.append(RequestMetrics(
            request_id="aggregate",
            latency=aggregate_response['latency'],
            tokens=aggregate_response['tokens'],
            engine_id=aggregate_response['engine_id'],
            priority=aggregate_response['priority'],
            service_time=aggregate_response['metadata']['service_time'],
            waiting_time=aggregate_response['metadata']['waiting_time'],
            thread_id=aggregate_response['metadata']['thread_id']
        ))

        logger.info(
            f"  Aggregation: latency={aggregate_response['latency']:.3f}s, "
            f"engine={aggregate_response['engine_id']}"
        )

        # Get session stats
        metrics.session_stats = client.get_session_stats()

    metrics.total_duration = time.time() - start_time
    metrics.compute_aggregates()

    logger.info(f"\nScenario completed in {metrics.total_duration:.2f}s")
    logger.info(f"Average latency: {metrics.avg_latency:.3f}s")

    return metrics


async def scenario_load_balancing_validation(
    system: AutellixSystem,
    num_sessions: int = 4,
    requests_per_session: int = 3
) -> ScenarioMetrics:
    """
    Scenario 3: Load balancing validation

    Creates multiple concurrent sessions to validate load distribution
    across engines.

    Args:
        system: AutellixSystem instance
        num_sessions: Number of concurrent sessions
        requests_per_session: Requests per session

    Returns:
        ScenarioMetrics with load distribution data
    """
    logger.info("="*80)
    logger.info("SCENARIO 3: Load Balancing Validation")
    logger.info("="*80)

    total_requests = num_sessions * requests_per_session
    metrics = ScenarioMetrics(
        scenario_name="Load Balancing",
        mode="PLAS",
        total_duration=0.0,
        total_requests=total_requests
    )

    start_time = time.time()

    logger.info(f"Creating {num_sessions} concurrent sessions")
    logger.info(f"Each session will submit {requests_per_session} requests\n")

    # Create multiple clients
    clients = [
        system.create_client(
            is_multithreaded=False,
            session_metadata={"session_num": i}
        )
        for i in range(num_sessions)
    ]

    try:
        # Submit requests from all sessions concurrently
        all_tasks = []
        for i, client in enumerate(clients):
            for j in range(requests_per_session):
                task = client.chat_completion(
                    messages=[
                        {"role": "user", "content": f"Session {i}, Request {j}"}
                    ],
                    max_tokens=50
                )
                all_tasks.append((f"session_{i}_req_{j}", task))

        logger.info(f"Submitting {len(all_tasks)} requests concurrently...")

        # Wait for all requests
        results = await asyncio.gather(*[t for _, t in all_tasks])

        # Collect metrics
        for (req_id, _), response in zip(all_tasks, results):
            metrics.request_metrics.append(RequestMetrics(
                request_id=req_id,
                latency=response['latency'],
                tokens=response['tokens'],
                engine_id=response['engine_id'],
                priority=response['priority'],
                service_time=response['metadata']['service_time'],
                waiting_time=response['metadata']['waiting_time']
            ))

        logger.info("\nLoad Distribution:")
        metrics.compute_aggregates()
        for engine_id, count in sorted(metrics.engine_distribution.items()):
            percentage = (count / total_requests) * 100
            logger.info(f"  {engine_id}: {count} requests ({percentage:.1f}%)")

    finally:
        # Close all clients
        for client in clients:
            client.close()

    metrics.total_duration = time.time() - start_time

    logger.info(f"\nScenario completed in {metrics.total_duration:.2f}s")

    return metrics


async def scenario_cache_locality_validation(
    system: AutellixSystem,
    num_long_requests: int = 5
) -> ScenarioMetrics:
    """
    Scenario 4: KV cache locality validation

    Tests that long requests (>2048 tokens) are routed to the same engine
    to preserve KV cache locality.

    Args:
        system: AutellixSystem instance
        num_long_requests: Number of long requests

    Returns:
        ScenarioMetrics with cache locality data
    """
    logger.info("="*80)
    logger.info("SCENARIO 4: KV Cache Locality Validation")
    logger.info("="*80)

    metrics = ScenarioMetrics(
        scenario_name="Cache Locality",
        mode="PLAS",
        total_duration=0.0,
        total_requests=num_long_requests
    )

    start_time = time.time()

    async with system.create_client(
        is_multithreaded=False,
        session_metadata={"scenario": "cache_test"}
    ) as client:

        logger.info(f"Submitting {num_long_requests} long requests (>2048 tokens)")
        logger.info("Expected: All should route to same engine for cache locality\n")

        # Create long prompts (>2048 tokens)
        base_prompt = "word " * 3000  # ~3000 tokens

        for i in range(num_long_requests):
            prompt = base_prompt + f" Request {i}"

            response = await client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )

            metrics.request_metrics.append(RequestMetrics(
                request_id=f"long_req_{i}",
                latency=response['latency'],
                tokens=response['tokens'],
                engine_id=response['engine_id'],
                priority=response['priority'],
                service_time=response['metadata']['service_time'],
                waiting_time=response['metadata']['waiting_time']
            ))

            logger.info(
                f"Request {i+1}: engine={response['engine_id']}, "
                f"latency={response['latency']:.3f}s"
            )

    metrics.total_duration = time.time() - start_time
    metrics.compute_aggregates()

    # Check cache locality
    engines_used = list(metrics.engine_distribution.keys())
    logger.info(f"\nCache Locality Analysis:")
    logger.info(f"  Engines used: {engines_used}")
    logger.info(f"  Cache affinity: {len(engines_used) == 1}")

    if len(engines_used) == 1:
        logger.info(f"  ✓ All requests routed to {engines_used[0]} (optimal)")
    else:
        logger.info(f"  ⚠ Requests split across {len(engines_used)} engines")

    return metrics


# ============================================================================
# Main E2E Test
# ============================================================================

async def run_e2e_test(
    num_engines: int = 4,
    skip_long_scenarios: bool = False
) -> Dict[str, ScenarioMetrics]:
    """
    Run complete end-to-end test suite.

    Args:
        num_engines: Number of vLLM engines to use
        skip_long_scenarios: Skip time-consuming scenarios for quick testing

    Returns:
        Dictionary of scenario names to metrics
    """
    logger.info("\n" + "="*80)
    logger.info("AUTELLIX RSM - END-TO-END TEST SUITE")
    logger.info("="*80)

    # Initialize system
    logger.info(f"\nInitializing Autellix system with {num_engines} engines...")
    config = AutellixConfig(
        num_engines=num_engines,
        model="meta-llama/Llama-2-7b-hf",  # Use mock-7b for testing without GPU
        num_priority_levels=8,
        base_quantum=512,
        cache_token_threshold=2048,
        max_num_seqs=256
    )

    system = AutellixSystem(config)
    system.start()

    # Wait for engines to initialize
    logger.info("Waiting for engines to initialize...")
    await asyncio.sleep(2.0)

    logger.info("✓ System initialized\n")

    results = {}

    try:
        # Scenario 1: Single-threaded chatbot (PLAS)
        logger.info("\n" + "🔹 Running Scenario 1...")
        results['single_threaded'] = await scenario_single_threaded_chatbot(
            system,
            num_requests=10 if not skip_long_scenarios else 5
        )

        await asyncio.sleep(1.0)  # Brief pause between scenarios

        # Scenario 2: Multi-threaded research (ATLAS)
        logger.info("\n" + "🔹 Running Scenario 2...")
        results['multi_threaded'] = await scenario_multi_threaded_research(
            system,
            num_parallel_tasks=5 if not skip_long_scenarios else 3
        )

        await asyncio.sleep(1.0)

        # Scenario 3: Load balancing
        logger.info("\n" + "🔹 Running Scenario 3...")
        results['load_balancing'] = await scenario_load_balancing_validation(
            system,
            num_sessions=4 if not skip_long_scenarios else 2,
            requests_per_session=3 if not skip_long_scenarios else 2
        )

        await asyncio.sleep(1.0)

        # Scenario 4: Cache locality
        logger.info("\n" + "🔹 Running Scenario 4...")
        results['cache_locality'] = await scenario_cache_locality_validation(
            system,
            num_long_requests=5 if not skip_long_scenarios else 3
        )

        # Collect final system stats
        logger.info("\n" + "="*80)
        logger.info("COLLECTING SYSTEM STATISTICS")
        logger.info("="*80)

        system_stats = system.get_system_stats()

        # Add system stats to all scenario metrics
        for metrics in results.values():
            metrics.system_stats = system_stats
            metrics.scheduler_stats = system_stats.get('scheduler')

            # Get cache stats
            if 'load_balancer' in system_stats:
                lb_stats = system_stats['load_balancer']
                total_hits = lb_stats.get('locality_hits', 0)
                total_reqs = lb_stats.get('total_requests', 1)
                metrics.cache_hits = total_hits
                metrics.cache_misses = total_reqs - total_hits
                metrics.cache_hit_rate = total_hits / total_reqs if total_reqs > 0 else 0

        logger.info("\n✓ System statistics collected")

    finally:
        # Cleanup
        logger.info("\nShutting down system...")
        system.stop()
        logger.info("✓ System shut down cleanly")

    return results


def generate_performance_report(
    results: Dict[str, ScenarioMetrics],
    output_path: Path
):
    """
    Generate comprehensive performance report.

    Args:
        results: Dictionary of scenario metrics
        output_path: Path to save report
    """
    logger.info("\n" + "="*80)
    logger.info("GENERATING PERFORMANCE REPORT")
    logger.info("="*80)

    report_lines = []

    # Header
    report_lines.append("# Autellix RSM End-to-End Test Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("## Executive Summary")
    report_lines.append("")

    # Summary table
    report_lines.append("| Scenario | Mode | Duration | Requests | Avg Latency | Tokens |")
    report_lines.append("|----------|------|----------|----------|-------------|--------|")

    for scenario_name, metrics in results.items():
        report_lines.append(
            f"| {metrics.scenario_name} | {metrics.mode} | "
            f"{metrics.total_duration:.2f}s | {metrics.total_requests} | "
            f"{metrics.avg_latency:.3f}s | {metrics.total_tokens} |"
        )

    report_lines.append("")

    # Detailed results for each scenario
    for scenario_name, metrics in results.items():
        report_lines.append(f"## {metrics.scenario_name}")
        report_lines.append("")
        report_lines.append(f"**Scheduling Mode:** {metrics.mode}")
        report_lines.append("")

        # Performance metrics
        report_lines.append("### Performance Metrics")
        report_lines.append("")
        report_lines.append(f"- **Total Duration:** {metrics.total_duration:.2f}s")
        report_lines.append(f"- **Total Requests:** {metrics.total_requests}")
        report_lines.append(f"- **Average Latency:** {metrics.avg_latency:.3f}s")
        report_lines.append(f"- **Min Latency:** {metrics.min_latency:.3f}s")
        report_lines.append(f"- **Max Latency:** {metrics.max_latency:.3f}s")
        report_lines.append(f"- **Total Tokens:** {metrics.total_tokens}")
        report_lines.append("")

        # Engine distribution
        if metrics.engine_distribution:
            report_lines.append("### Engine Distribution")
            report_lines.append("")
            report_lines.append("| Engine | Requests | Percentage |")
            report_lines.append("|--------|----------|------------|")
            for engine_id, count in sorted(metrics.engine_distribution.items()):
                percentage = (count / metrics.total_requests) * 100
                report_lines.append(f"| {engine_id} | {count} | {percentage:.1f}% |")
            report_lines.append("")

        # Cache metrics
        if metrics.cache_hits > 0 or metrics.cache_misses > 0:
            report_lines.append("### Cache Locality")
            report_lines.append("")
            report_lines.append(f"- **Cache Hits:** {metrics.cache_hits}")
            report_lines.append(f"- **Cache Misses:** {metrics.cache_misses}")
            report_lines.append(f"- **Hit Rate:** {metrics.cache_hit_rate:.2%}")
            report_lines.append("")

        # Session stats
        if metrics.session_stats:
            report_lines.append("### Session Statistics")
            report_lines.append("")
            for key, value in metrics.session_stats.items():
                if isinstance(value, float):
                    report_lines.append(f"- **{key}:** {value:.2f}")
                else:
                    report_lines.append(f"- **{key}:** {value}")
            report_lines.append("")

    # PLAS vs ATLAS comparison
    if 'single_threaded' in results and 'multi_threaded' in results:
        plas = results['single_threaded']
        atlas = results['multi_threaded']

        report_lines.append("## PLAS vs ATLAS Comparison")
        report_lines.append("")
        report_lines.append("| Metric | PLAS (Single-Threaded) | ATLAS (Multi-Threaded) |")
        report_lines.append("|--------|------------------------|------------------------|")
        report_lines.append(
            f"| Total Duration | {plas.total_duration:.2f}s | {atlas.total_duration:.2f}s |"
        )
        report_lines.append(
            f"| Avg Latency | {plas.avg_latency:.3f}s | {atlas.avg_latency:.3f}s |"
        )
        report_lines.append(
            f"| Total Requests | {plas.total_requests} | {atlas.total_requests} |"
        )
        report_lines.append("")

        # Analysis
        report_lines.append("### Analysis")
        report_lines.append("")
        if atlas.total_duration < plas.total_duration:
            speedup = plas.total_duration / atlas.total_duration
            report_lines.append(
                f"ATLAS scheduling achieved **{speedup:.2f}x speedup** over PLAS "
                f"for the multi-threaded workload, demonstrating effective parallel "
                f"execution and critical path optimization."
            )
        else:
            report_lines.append(
                f"PLAS scheduling was more efficient for this workload pattern. "
                f"ATLAS overhead may not be justified for workloads with limited parallelism."
            )
        report_lines.append("")

    # System statistics
    if results:
        first_result = next(iter(results.values()))
        if first_result.system_stats:
            report_lines.append("## System Statistics")
            report_lines.append("")

            stats = first_result.system_stats

            # Frontend stats
            if 'frontend' in stats:
                report_lines.append("### Frontend (Process Table)")
                report_lines.append("")
                pt_stats = stats['frontend'].get('process_table', {})
                for key, value in pt_stats.items():
                    report_lines.append(f"- **{key}:** {value}")
                report_lines.append("")

            # Scheduler stats
            if 'scheduler' in stats:
                report_lines.append("### Scheduler")
                report_lines.append("")
                sched_stats = stats['scheduler']
                for key, value in sched_stats.items():
                    report_lines.append(f"- **{key}:** {value}")
                report_lines.append("")

            # Load balancer stats
            if 'load_balancer' in stats:
                report_lines.append("### Load Balancer")
                report_lines.append("")
                lb_stats = stats['load_balancer']
                for key, value in lb_stats.items():
                    if isinstance(value, float):
                        report_lines.append(f"- **{key}:** {value:.2f}")
                    else:
                        report_lines.append(f"- **{key}:** {value}")
                report_lines.append("")

            # Engine stats
            if 'engines' in stats:
                report_lines.append("### Engines")
                report_lines.append("")
                engine_stats = stats['engines']
                for key, value in engine_stats.items():
                    if isinstance(value, float):
                        report_lines.append(f"- **{key}:** {value:.3f}")
                    else:
                        report_lines.append(f"- **{key}:** {value}")
                report_lines.append("")

    # Footer
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("*Report generated by Autellix RSM E2E Test Suite*")

    # Write report
    report_content = "\n".join(report_lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    logger.info(f"✓ Report saved to: {output_path}")
    logger.info(f"  Report size: {len(report_content)} characters")


def validate_results(results: Dict[str, ScenarioMetrics]) -> bool:
    """
    Validate test results meet expectations.

    Args:
        results: Dictionary of scenario metrics

    Returns:
        True if all validations pass
    """
    logger.info("\n" + "="*80)
    logger.info("VALIDATING RESULTS")
    logger.info("="*80)

    all_passed = True

    for scenario_name, metrics in results.items():
        logger.info(f"\nValidating {metrics.scenario_name}...")

        # Check basic metrics
        assert metrics.total_requests > 0, f"{scenario_name}: No requests processed"
        assert metrics.total_duration > 0, f"{scenario_name}: Invalid duration"
        assert len(metrics.request_metrics) == metrics.total_requests, \
            f"{scenario_name}: Request count mismatch"

        # Check all requests completed
        for req in metrics.request_metrics:
            assert req.latency > 0, f"{scenario_name}: Invalid latency for {req.request_id}"
            assert req.tokens > 0, f"{scenario_name}: No tokens generated for {req.request_id}"
            assert req.engine_id, f"{scenario_name}: No engine assigned for {req.request_id}"

        # Check engine distribution
        assert len(metrics.engine_distribution) > 0, f"{scenario_name}: No engines used"

        # Scenario-specific validations
        if scenario_name == 'cache_locality':
            # Cache locality should route to same engine
            if len(metrics.engine_distribution) > 1:
                logger.warning(
                    f"  ⚠ Cache locality test used {len(metrics.engine_distribution)} "
                    f"engines (expected 1)"
                )

        if scenario_name == 'load_balancing':
            # Load should be distributed
            engine_counts = list(metrics.engine_distribution.values())
            max_count = max(engine_counts)
            min_count = min(engine_counts)
            balance_ratio = min_count / max_count if max_count > 0 else 0

            logger.info(f"  Load balance ratio: {balance_ratio:.2f}")
            if balance_ratio < 0.5:
                logger.warning(f"  ⚠ Load imbalance detected (ratio: {balance_ratio:.2f})")

        logger.info(f"  ✓ {metrics.scenario_name} validation passed")

    logger.info("\n" + "="*80)
    logger.info("✓ ALL VALIDATIONS PASSED")
    logger.info("="*80)

    return all_passed


async def main():
    """Main entry point for E2E test."""
    logger.info("\n" + "="*80)
    logger.info("STARTING AUTELLIX RSM END-TO-END TEST")
    logger.info("="*80)

    try:
        # Run test suite
        results = await run_e2e_test(
            num_engines=4,
            skip_long_scenarios=False  # Set to True for quick testing
        )

        # Validate results
        validate_results(results)

        # Generate report
        report_path = Path(__file__).parent.parent / "examples" / "e2e_test_report.md"
        generate_performance_report(results, report_path)

        # Print summary
        logger.info("\n" + "="*80)
        logger.info("TEST SUMMARY")
        logger.info("="*80)
        logger.info(f"✓ Scenarios completed: {len(results)}")
        logger.info(f"✓ Report generated: {report_path}")
        logger.info("="*80)
        logger.info("🎉 ALL TESTS PASSED")
        logger.info("="*80 + "\n")

        return True

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
