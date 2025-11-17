"""
Example: Using Local LLM for UDF Optimization

This example demonstrates how to use the UDF optimizer with a local LLM server
instead of cloud APIs like Gemini.

Prerequisites:
1. Start a local LLM server (vLLM, Ollama, llama.cpp, etc.)
2. Install required packages: pip install openai pyyaml
3. Configure config/config.yaml with your local server details

Example server setups:

# vLLM
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

# Ollama
ollama serve  # Runs on port 11434

# llama.cpp
./server -m model.gguf --port 8000
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm_client import LLMConfig, create_llm_client, LocalLLMClient
from core.executor import DependencyAnalyzer, StepExecutor, load_plan_from_json
from core.workflow_types import Plan, Step, StepType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_basic_local_llm():
    """Example 1: Basic local LLM usage"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Local LLM Usage")
    print("="*70)

    # Create config for local LLM
    config = LLMConfig(
        backend="local",
        local_api_base="http://localhost:8000/v1",  # vLLM default
        local_model="meta-llama/Llama-3.1-8B-Instruct",  # Change to your model
        temperature=0.7,
        max_tokens=512
    )

    # Create client
    client = create_llm_client(config)

    # Test simple generation
    print("\nTesting simple text generation...")
    result = client.generate_sync(
        prompt="What is the capital of France? Answer in one sentence.",
        system_prompt="You are a helpful assistant."
    )

    print(f"\nResponse: {result}")


def example_2_with_ollama():
    """Example 2: Using Ollama (different port/model naming)"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Using Ollama")
    print("="*70)

    # Ollama configuration (different defaults)
    config = LLMConfig(
        backend="local",
        local_api_base="http://localhost:11434/v1",  # Ollama default port
        local_model="llama3.1:8b",  # Ollama model format
        temperature=0.7
    )

    client = create_llm_client(config)

    print("\nTesting with Ollama...")
    result = client.generate_sync(
        prompt="Explain quantum computing in 2 sentences.",
        system_prompt="You are a physics teacher."
    )

    print(f"\nResponse: {result}")


def example_3_dependency_analysis():
    """Example 3: Dependency analysis with local LLM"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Dependency Analysis with Local LLM")
    print("="*70)

    # Method 1: Using config.yaml (recommended)
    print("\n--- Method 1: Load from config.yaml ---")

    # Make sure config.yaml has backend: "local"
    analyzer = DependencyAnalyzer()

    # Create a simple test plan
    plan = Plan(
        title="Test Research Plan",
        locale="en",
        thought="This is a test plan to demonstrate dependency analysis",
        steps=[
            Step(
                title="Research topic A",
                description="Find information about topic A",
                step_type=StepType.RESEARCH,
                need_search=True,
                step_id="step-1"
            ),
            Step(
                title="Research topic B",
                description="Find information about topic B",
                step_type=StepType.RESEARCH,
                need_search=True,
                step_id="step-2"
            ),
            Step(
                title="Combine results",
                description="Analyze and combine the research findings",
                step_type=StepType.PROCESSING,
                need_search=False,
                step_id="step-3"
            )
        ],
        has_enough_context=False
    )

    # Analyze dependencies
    print("\nAnalyzing plan with local LLM...")
    batches = analyzer.analyze_plan(plan)

    print(f"\nCreated {len(batches)} batches:")
    for batch in batches:
        print(f"  {batch}")

    # Method 2: Custom LLM client (advanced)
    print("\n--- Method 2: Custom LLM client ---")

    custom_config = LLMConfig(
        backend="local",
        local_api_base="http://localhost:8000/v1",
        local_model="your-specific-model",
        temperature=0.3  # Lower for more reliable dependency analysis
    )

    custom_client = create_llm_client(custom_config)
    custom_analyzer = DependencyAnalyzer(llm_client=custom_client)

    # Analyze with custom client
    batches_custom = custom_analyzer.analyze_plan(plan)

    print(f"\nCustom analysis created {len(batches_custom)} batches:")
    for batch in batches_custom:
        print(f"  {batch}")


async def example_4_step_execution():
    """Example 4: Execute steps with local LLM"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Step Execution with Local LLM")
    print("="*70)

    # Create executor (uses config.yaml by default)
    executor = StepExecutor()

    # Create test step
    step = Step(
        title="Research Python async/await",
        description="Explain how async/await works in Python with examples",
        step_type=StepType.RESEARCH,
        need_search=True,
        step_id="test-step"
    )

    # Execute step
    print("\nExecuting research step with local LLM...")
    result = await executor.execute_step(step, context=[])

    print(f"\nExecution result (first 500 chars):")
    print(result[:500])
    print("...")


async def example_5_full_workflow():
    """Example 5: Complete workflow with local LLM"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Full Workflow with Local LLM")
    print("="*70)

    # 1. Create analyzer and executor
    analyzer = DependencyAnalyzer()
    executor = StepExecutor()

    # 2. Load or create plan
    plan = Plan(
        title="AI Research Workflow",
        locale="en",
        thought="Research different AI topics and synthesize findings",
        steps=[
            Step(
                title="Research neural networks",
                description="Explain neural networks basics",
                step_type=StepType.RESEARCH,
                need_search=True,
                step_id="step-1"
            ),
            Step(
                title="Research transformers",
                description="Explain transformer architecture",
                step_type=StepType.RESEARCH,
                need_search=True,
                step_id="step-2"
            ),
            Step(
                title="Compare approaches",
                description="Compare neural networks and transformers",
                step_type=StepType.PROCESSING,
                need_search=False,
                step_id="step-3"
            )
        ],
        has_enough_context=False
    )

    # 3. Analyze dependencies
    print("\n=== Step 1: Dependency Analysis ===")
    batches = analyzer.analyze_plan(plan)

    print(f"\nIdentified {len(batches)} execution batches:")
    for batch in batches:
        mode = "PARALLEL" if batch.parallel else "SEQUENTIAL"
        print(f"  Batch {batch.batch_id} ({mode}): "
              f"{len(batch.step_indices)} steps - {batch.description}")

    # 4. Execute batches
    print("\n=== Step 2: Executing Batches ===")

    all_results = {}

    for batch in batches:
        print(f"\n--- Executing Batch {batch.batch_id} "
              f"({'parallel' if batch.parallel else 'sequential'}) ---")

        # Get steps in this batch
        batch_steps = [plan.steps[i] for i in batch.step_indices]

        if batch.parallel:
            # Execute in parallel
            tasks = []
            for step in batch_steps:
                # Get context from previous results
                context = list(all_results.values())
                task = executor.execute_step(step, context)
                tasks.append(task)

            # Wait for all
            results = await asyncio.gather(*tasks)

            # Store results
            for step, result in zip(batch_steps, results):
                all_results[step.step_id] = result
                print(f"  ✓ {step.title}: {len(result)} chars")

        else:
            # Execute sequentially
            for step in batch_steps:
                context = list(all_results.values())
                result = await executor.execute_step(step, context)
                all_results[step.step_id] = result
                print(f"  ✓ {step.title}: {len(result)} chars")

    # 5. Show final results
    print("\n=== Step 3: Results ===")
    for step_id, result in all_results.items():
        print(f"\n{step_id}:")
        print(f"  {result[:200]}...")


def example_6_comparing_backends():
    """Example 6: Compare different LLM backends"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Comparing LLM Backends")
    print("="*70)

    # Test prompt
    test_prompt = "What is 2 + 2? Answer with just the number."

    backends_to_test = [
        ("Local vLLM", LLMConfig(
            backend="local",
            local_api_base="http://localhost:8000/v1",
            local_model="llama3.1:8b"
        )),
        ("Ollama", LLMConfig(
            backend="local",
            local_api_base="http://localhost:11434/v1",
            local_model="llama3.1:8b"
        )),
    ]

    print("\nTesting same prompt on different backends:\n")

    for name, config in backends_to_test:
        print(f"--- {name} ---")
        try:
            client = create_llm_client(config)
            result = client.generate_sync(
                prompt=test_prompt,
                system_prompt="You are a helpful assistant.",
                max_tokens=50
            )
            print(f"Response: {result.strip()}\n")
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("LOCAL LLM EXAMPLES FOR UDF OPTIMIZER")
    print("="*70)

    print("\nPrerequisites:")
    print("1. Start a local LLM server (vLLM, Ollama, etc.)")
    print("2. Configure config/config.yaml with your server details")
    print("3. Install dependencies: pip install openai pyyaml")

    print("\nExamples:")
    print("1. Basic local LLM usage")
    print("2. Using Ollama")
    print("3. Dependency analysis")
    print("4. Step execution")
    print("5. Full workflow")
    print("6. Comparing backends")

    choice = input("\nSelect example (1-6, or 'all'): ").strip()

    if choice == '1':
        example_1_basic_local_llm()
    elif choice == '2':
        example_2_with_ollama()
    elif choice == '3':
        example_3_dependency_analysis()
    elif choice == '4':
        asyncio.run(example_4_step_execution())
    elif choice == '5':
        asyncio.run(example_5_full_workflow())
    elif choice == '6':
        example_6_comparing_backends()
    elif choice.lower() == 'all':
        try:
            example_1_basic_local_llm()
        except Exception as e:
            print(f"\nExample 1 failed: {e}")

        try:
            example_2_with_ollama()
        except Exception as e:
            print(f"\nExample 2 failed: {e}")

        try:
            example_3_dependency_analysis()
        except Exception as e:
            print(f"\nExample 3 failed: {e}")

        try:
            asyncio.run(example_4_step_execution())
        except Exception as e:
            print(f"\nExample 4 failed: {e}")

        try:
            asyncio.run(example_5_full_workflow())
        except Exception as e:
            print(f"\nExample 5 failed: {e}")

        try:
            example_6_comparing_backends()
        except Exception as e:
            print(f"\nExample 6 failed: {e}")
    else:
        print("Invalid choice")

    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
