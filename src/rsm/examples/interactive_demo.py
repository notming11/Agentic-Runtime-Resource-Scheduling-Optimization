"""
Interactive Demo: See Autellix Frontend in Action

This script demonstrates the process table and session manager
with detailed output showing what's happening internally.
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frontend import GlobalProcessTable, SessionManager, AutellixClient


def demo_process_table():
    """Demo 1: Process Table - See PLAS and ATLAS in action"""
    print("=" * 70)
    print("DEMO 1: Process Table - PLAS vs ATLAS")
    print("=" * 70)

    pt = GlobalProcessTable()

    # Single-threaded program (PLAS)
    print("\n📊 Creating single-threaded program (PLAS scheduling)...")
    pid_single = "chatbot_program"
    pt.create_program(pid_single, is_multithreaded=False)
    print(f"   ✓ Created program: {pid_single}")

    # Simulate 3 sequential LLM calls
    print("\n🔄 Simulating 3 sequential LLM calls...")
    for i in range(3):
        call_id = f"call_{i}"
        thread_id = f"thread_{i}"

        # Add call
        thread = pt.add_llm_call(pid_single, call_id, thread_id, prefill_tokens=100)
        print(f"\n   Call {i+1} added:")
        print(f"      Thread ID: {thread_id}")
        print(f"      Initial priority: {thread.priority:.2f}s")

        # Simulate execution
        time.sleep(0.1)  # Simulate some work
        service_time = 0.5 + i * 0.3  # Increasing service times
        waiting_time = 0.2

        # Update metrics
        pt.update_program_metrics(pid_single, call_id, service_time, waiting_time)

        program = pt.get_program(pid_single)
        print(f"      Service time: {service_time:.2f}s")
        print(f"      CUMULATIVE service time: {program.service_time:.2f}s ← PLAS")
        print(f"      Total waiting time: {program.waiting_time:.2f}s")
        print(f"      Starvation ratio: {program.get_starvation_ratio():.2f}")

    # Multi-threaded program (ATLAS)
    print("\n\n📊 Creating multi-threaded program (ATLAS scheduling)...")
    pid_multi = "mcts_program"
    pt.create_program(pid_multi, is_multithreaded=True)
    print(f"   ✓ Created program: {pid_multi}")

    print("\n🌳 Simulating DAG execution (map-reduce):")
    print("""
       Root (1.0s)
          ↓
    ┌─────┼─────┐
    A     B     C
   (2.0s)(3.0s)(1.5s)
    └─────┼─────┘
          ↓
       Final
    """)

    # Root node
    print("\n   Step 1: Root node")
    root_thread = pt.add_llm_call(pid_multi, "root_call", "root_thread", 100)
    root_thread.service_time = 1.0
    pt.update_program_metrics(pid_multi, "root_call", 1.0, 0.1)
    program = pt.get_program(pid_multi)
    print(f"      Root service time: 1.0s")
    print(f"      Critical path: {program.service_time:.2f}s")

    # Parallel branches
    print("\n   Step 2: Parallel branches (A, B, C)")
    branches = [
        ("thread_A", "call_A", 2.0),
        ("thread_B", "call_B", 3.0),
        ("thread_C", "call_C", 1.5),
    ]

    for thread_id, call_id, svc_time in branches:
        thread = pt.add_llm_call(
            pid_multi, call_id, thread_id, 100,
            parent_thread_ids=["root_thread"]
        )
        thread.service_time = svc_time
        program.update_service_time_multi_threaded(thread_id, ["root_thread"])

        critical_path = program.service_time
        print(f"      {thread_id}: {svc_time}s → Critical path: {critical_path:.2f}s")

    print(f"\n   🎯 ATLAS Critical Path: {program.service_time:.2f}s (1.0 + 3.0)")
    print(f"      (Longest path: Root → B → Final)")

    # Stats
    print("\n" + "=" * 70)
    print("📈 Process Table Statistics:")
    stats = pt.get_stats()
    print(f"   Total programs: {stats['total_programs']}")
    print(f"   Active calls: {stats['total_active_calls']}")

    for pid, prog_stats in stats['programs'].items():
        print(f"\n   Program: {pid}")
        print(f"      Service time: {prog_stats['service_time']:.2f}s")
        print(f"      Waiting time: {prog_stats['waiting_time']:.2f}s")
        print(f"      Active threads: {prog_stats['active_threads']}")
        print(f"      Starvation ratio: {prog_stats['starvation_ratio']:.2f}")


def demo_session_manager():
    """Demo 2: Session Manager - See sessions in action"""
    print("\n\n" + "=" * 70)
    print("DEMO 2: Session Manager - Lifecycle & Statistics")
    print("=" * 70)

    pt = GlobalProcessTable()
    sm = SessionManager(pt)

    print("\n🚀 Starting 3 sessions...")
    sessions = []
    for i in range(3):
        session_id = sm.start_session(
            is_multithreaded=(i == 2),  # Last one is multithreaded
            metadata={"user": f"user_{i}", "task": f"task_{i}"}
        )
        sessions.append(session_id)
        print(f"   ✓ Session {i+1}: {session_id[:20]}...")

    print(f"\n📊 Active sessions: {len(sm.list_active_sessions())}")

    # Simulate some LLM calls
    print("\n🔄 Simulating LLM calls for session 1...")
    session_id = sessions[0]

    for i in range(3):
        thread_id = sm.register_llm_call(
            session_id=session_id,
            call_id=f"call_{i}",
            prefill_tokens=100,
            engine_id="engine_0"
        )
        print(f"   Call {i+1} registered: {thread_id}")

        # Simulate completion
        time.sleep(0.05)
        sm.complete_llm_call(
            session_id=session_id,
            thread_id=thread_id,
            service_time=0.5,
            waiting_time=0.1
        )
        print(f"      ✓ Completed")

    # Get session stats
    print("\n📈 Session 1 Statistics:")
    stats = sm.get_session_stats(session_id)
    print(f"   Session ID: {stats['session_id'][:20]}...")
    print(f"   State: {stats['state']}")
    print(f"   Duration: {stats['duration']:.2f}s")
    print(f"   Total calls: {stats['total_calls']}")
    print(f"   Completed calls: {stats['completed_calls']}")
    print(f"   Service time: {stats['service_time']:.2f}s")
    print(f"   Waiting time: {stats['waiting_time']:.2f}s")

    # End sessions
    print("\n🛑 Ending sessions...")
    for i, session_id in enumerate(sessions):
        sm.end_session(session_id)
        print(f"   ✓ Session {i+1} ended")

    # Global stats
    print("\n📊 Global Statistics:")
    global_stats = sm.get_global_stats()
    print(f"   Total sessions created: {global_stats['total_sessions_created']}")
    print(f"   Total sessions completed: {global_stats['total_sessions_completed']}")
    print(f"   Active sessions: {global_stats['active_sessions']}")


def demo_client_api():
    """Demo 3: Client API - See what gets tracked"""
    print("\n\n" + "=" * 70)
    print("DEMO 3: Client API - Internal Tracking")
    print("=" * 70)

    print("\n🚀 Creating AutellixClient...")
    client = AutellixClient(is_multithreaded=False)
    print(f"   ✓ Session started: {client.session_id[:20]}...")

    print("\n📤 Making LLM calls (with mock responses)...")
    for i in range(3):
        print(f"\n   Call {i+1}:")
        response = client.chat_completion(
            messages=[{"role": "user", "content": f"Test message {i}"}],
            model="llama-3.1-8b"
        )

        # Show what was tracked
        metadata = response["autellix_metadata"]
        print(f"      ✓ Call ID: {metadata['thread_id']}")
        print(f"      ✓ Session ID: {metadata['session_id'][:20]}...")
        print(f"      ✓ Response: {response['choices'][0]['message']['content']}")

    # Get stats
    print("\n📈 Session Statistics:")
    stats = client.get_session_stats()
    print(f"   Total calls: {stats['total_calls']}")
    print(f"   Active threads: {stats['active_threads']}")
    print(f"   Duration: {stats['duration']:.2f}s")

    # Access the underlying process table
    print("\n🔍 Underlying Process Table Entry:")
    program = client.process_table.get_program(client.session_id)
    print(f"   Program ID: {program.pid[:20]}...")
    print(f"   Total calls made: {program.total_calls}")
    print(f"   Threads in memory: {list(program.threads.keys())}")

    client.close()
    print(f"\n✓ Session closed")


def demo_anti_starvation():
    """Demo 4: Anti-Starvation Mechanism"""
    print("\n\n" + "=" * 70)
    print("DEMO 4: Anti-Starvation Detection")
    print("=" * 70)

    pt = GlobalProcessTable()

    print("\n📊 Creating two programs...")
    pid_fast = "fast_program"
    pid_slow = "slow_program"

    pt.create_program(pid_fast, is_multithreaded=False)
    pt.create_program(pid_slow, is_multithreaded=False)

    print(f"   ✓ Fast program: {pid_fast}")
    print(f"   ✓ Slow program: {pid_slow}")

    # Fast program - executes quickly
    print("\n⚡ Fast program execution:")
    pt.add_llm_call(pid_fast, "call_1", "thread_1", 100)
    pt.update_program_metrics(pid_fast, "call_1", service_time=1.0, waiting_time=0.2)

    fast_prog = pt.get_program(pid_fast)
    print(f"   Service time: {fast_prog.service_time:.2f}s")
    print(f"   Waiting time: {fast_prog.waiting_time:.2f}s")
    print(f"   Starvation ratio: {fast_prog.get_starvation_ratio():.2f}")

    # Slow program - lots of waiting
    print("\n🐌 Slow program execution (getting starved):")
    pt.add_llm_call(pid_slow, "call_1", "thread_1", 100)
    pt.update_program_metrics(pid_slow, "call_1", service_time=1.0, waiting_time=5.0)

    slow_prog = pt.get_program(pid_slow)
    print(f"   Service time: {slow_prog.service_time:.2f}s")
    print(f"   Waiting time: {slow_prog.waiting_time:.2f}s")
    print(f"   Starvation ratio: {slow_prog.get_starvation_ratio():.2f}")

    # Check for starvation
    beta = 2.0  # Starvation threshold from Autellix paper
    print(f"\n⚠️  Checking starvation (threshold β = {beta}):")
    print(f"   Fast program: {fast_prog.get_starvation_ratio():.2f} < {beta} ✓ OK")

    if slow_prog.get_starvation_ratio() >= beta:
        print(f"   Slow program: {slow_prog.get_starvation_ratio():.2f} >= {beta} ⚠️  STARVED!")
        print(f"   → Would be PROMOTED to highest priority queue (Q1)")
    else:
        print(f"   Slow program: {slow_prog.get_starvation_ratio():.2f} < {beta} ✓ OK")


if __name__ == "__main__":
    print("\n")
    print("█" * 70)
    print("█  AUTELLIX RSM FRONTEND - INTERACTIVE DEMO")
    print("█  Partition 1: Process Table & Session Management")
    print("█" * 70)

    # Run all demos
    demo_process_table()
    demo_session_manager()
    demo_client_api()
    demo_anti_starvation()

    print("\n\n" + "=" * 70)
    print("✅ ALL DEMOS COMPLETED!")
    print("=" * 70)
    print("\nWhat you just saw:")
    print("  ✓ PLAS scheduling (cumulative service time)")
    print("  ✓ ATLAS scheduling (critical path calculation)")
    print("  ✓ Session lifecycle management")
    print("  ✓ Process table statistics")
    print("  ✓ Anti-starvation detection")
    print("\nNote: Backend (scheduler, load balancer) not implemented yet.")
    print("      Responses are mocked, but all tracking/metrics work correctly!")
    print("=" * 70 + "\n")
