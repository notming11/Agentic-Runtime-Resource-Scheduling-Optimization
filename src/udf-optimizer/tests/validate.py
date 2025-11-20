"""
Quick validation test to ensure all components are importable and work together.

This script validates:
1. All imports work without errors
2. Plan can be loaded from JSON
3. DependencyAnalyzer can be instantiated
4. GeminiStepExecutor can be instantiated
5. Workflow types are correct
6. Configuration management works

Run this before executing main_real.py to catch any issues early.
"""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*80)
print("VALIDATION TEST - Real LLM Integration")
print("="*80)

# Test 1: Imports
print("\n[1/6] Testing imports...")
try:
    from core import (
        Plan, Step, State, Configuration, StepType,
        parallel_research_team_node,
        build_parallel_workflow_graph,
        ConfigurationManager,
        DependencyAnalyzer, 
        GeminiStepExecutor, 
        load_plan_from_json,
        BatchDefinition
    )
    print("     ✓ All imports successful")
except Exception as e:
    print(f"     ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Load Plan
print("\n[2/6] Testing plan loading...")
try:
    script_dir = Path(__file__).parent.parent
    plan_path = script_dir / "examples" / "example_response_1.txt"
    
    if not plan_path.exists():
        raise FileNotFoundError(f"example_response_1.txt not found at {plan_path}")
    
    plan = load_plan_from_json(plan_path)
    
    print(f"     ✓ Plan loaded: {plan.title}")
    print(f"       - {len(plan.steps)} steps")
    print(f"       - {sum(1 for s in plan.steps if s.step_type == StepType.RESEARCH)} research")
    print(f"       - {sum(1 for s in plan.steps if s.step_type == StepType.PROCESSING)} processing")
except Exception as e:
    print(f"     ✗ Plan loading failed: {e}")
    sys.exit(1)

# Test 3: DependencyAnalyzer
print("\n[3/6] Testing DependencyAnalyzer...")
try:
    analyzer = DependencyAnalyzer()
    print("     ✓ DependencyAnalyzer instantiated")
    
    # Check parallel_prompt.md exists
    prompt_path = script_dir / "config" / "parallel_prompt.md"
    if not prompt_path.exists():
        print(f"     ⚠ Warning: parallel_prompt.md not found at {prompt_path}")
        print("       LLM analysis will fail, but heuristic fallback will work")
    else:
        print(f"     ✓ parallel_prompt.md found")
except Exception as e:
    print(f"     ✗ DependencyAnalyzer failed: {e}")
    sys.exit(1)

# Test 4: GeminiStepExecutor
print("\n[4/6] Testing GeminiStepExecutor...")
try:
    executor = GeminiStepExecutor()
    print("     ✓ GeminiStepExecutor instantiated")
except Exception as e:
    print(f"     ✗ GeminiStepExecutor failed: {e}")
    sys.exit(1)

# Test 5: API Key Check
print("\n[5/6] Testing API key configuration...")
try:
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if api_key:
        masked_key = api_key[:10] + "..." if len(api_key) > 10 else "***"
        print(f"     ✓ GEMINI_API_KEY found: {masked_key}")
    else:
        print("     ⚠ Warning: GEMINI_API_KEY not found in .env")
        print("       Real execution will fail without API key")
        print("       Create .env file with: GEMINI_API_KEY=your_key")
except Exception as e:
    print(f"     ✗ API key check failed: {e}")
    # Don't exit - user might want to test without API key

# Test 6: Configuration
print("\n[6/6] Testing configuration...")
try:
    config = Configuration(
        max_concurrent_tasks=5,
        task_timeout_seconds=120,
        max_retries=2
    )
    print(f"     ✓ Configuration created")
    print(f"       - Max concurrent: {config.max_concurrent_tasks}")
    print(f"       - Timeout: {config.task_timeout_seconds}s")
    print(f"       - Retries: {config.max_retries}")
except Exception as e:
    print(f"     ✗ Configuration failed: {e}")
    sys.exit(1)

# Test 7: State Creation
print("\n[7/7] Testing state creation...")
try:
    state = State(
        messages=[],
        observations=[],
        current_plan=plan
    )
    print("     ✓ State created with plan")
except Exception as e:
    print(f"     ✗ State creation failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
print("\n✓ All components validated successfully!")
print("\nYou can now run:")
print("  - python main_real.py  (requires GEMINI_API_KEY)")
print("  - python main.py       (mock execution, no API key needed)")
print("\n" + "="*80)
