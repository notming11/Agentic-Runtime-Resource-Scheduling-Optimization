"""
Basic Usage Examples for Autellix RSM Frontend

This file demonstrates how to use the Autellix stateful API for both
single-threaded and multi-threaded agentic programs.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frontend import AutellixClient, autellix_session, AutellixOpenAIAdapter


def example_single_threaded_chatbot():
    """
    Example 1: Single-threaded chatbot (uses PLAS scheduling)

    This represents a simple conversational agent where each message
    follows sequentially.
    """
    print("=" * 60)
    print("Example 1: Single-Threaded Chatbot (PLAS Scheduling)")
    print("=" * 60)

    # Create client for single-threaded program
    client = AutellixClient(
        backend_url="http://localhost:8000",
        is_multithreaded=False,  # Use PLAS scheduling
        session_metadata={"app": "chatbot", "user": "demo"}
    )

    print(f"Session started: {client.session_id}")

    # Simulate a conversation
    conversation = [
        "Hello! How are you?",
        "What's the weather like today?",
        "Tell me a joke.",
        "Goodbye!"
    ]

    for i, message in enumerate(conversation, 1):
        print(f"\nTurn {i}: {message}")

        response = client.chat_completion(
            messages=[
                {"role": "user", "content": message}
            ],
            model="llama-3.1-8b",
            temperature=0.7
        )

        print(f"Response: {response['choices'][0]['message']['content']}")

    # Get session statistics
    stats = client.get_session_stats()
    print("\n" + "-" * 60)
    print("Session Statistics:")
    print(f"  Total calls: {stats['total_calls']}")
    print(f"  Completed calls: {stats['completed_calls']}")
    print(f"  Service time: {stats['service_time']:.2f}s")
    print(f"  Waiting time: {stats['waiting_time']:.2f}s")
    print(f"  Starvation ratio: {stats['starvation_ratio']:.2f}")

    # Close session
    client.close()
    print(f"\nSession {client.session_id} closed.")


def example_multi_threaded_react():
    """
    Example 2: Multi-threaded ReAct agent (uses ATLAS scheduling)

    This represents a ReAct agent that can make parallel tool calls.
    """
    print("\n" + "=" * 60)
    print("Example 2: Multi-Threaded ReAct Agent (ATLAS Scheduling)")
    print("=" * 60)

    # Use context manager for automatic cleanup
    with autellix_session(is_multithreaded=True) as client:
        print(f"Session started: {client.session_id}")

        # Initial reasoning step
        print("\nStep 1: Reasoning")
        reasoning_response = client.chat_completion(
            messages=[
                {"role": "user", "content": "Research the capitals of France, Germany, and Italy"}
            ],
            model="llama-3.1-8b"
        )
        reasoning_thread_id = reasoning_response["autellix_metadata"]["thread_id"]
        print(f"Thread ID: {reasoning_thread_id}")

        # Parallel tool calls (map-reduce pattern)
        print("\nStep 2: Parallel Tool Calls")
        tool_calls = [
            {"messages": [{"role": "user", "content": "What is the capital of France?"}]},
            {"messages": [{"role": "user", "content": "What is the capital of Germany?"}]},
            {"messages": [{"role": "user", "content": "What is the capital of Italy?"}]},
        ]

        responses = client.parallel_chat_completion(
            requests=tool_calls,
            parent_thread_id=reasoning_thread_id  # All depend on reasoning step
        )

        for i, response in enumerate(responses, 1):
            thread_id = response["autellix_metadata"]["thread_id"]
            content = response["choices"][0]["message"]["content"]
            print(f"  Tool call {i} (thread {thread_id}): {content}")

        # Aggregation step (depends on all tool calls)
        print("\nStep 3: Aggregation")
        thread_ids = [r["autellix_metadata"]["thread_id"] for r in responses]
        final_response = client.chat_completion(
            messages=[
                {"role": "user", "content": "Summarize the capitals"}
            ],
            model="llama-3.1-8b",
            parent_thread_ids=thread_ids  # Depends on all tool calls
        )
        print(f"Final summary: {final_response['choices'][0]['message']['content']}")

        # Get session statistics
        stats = client.get_session_stats()
        print("\n" + "-" * 60)
        print("Session Statistics:")
        print(f"  Total calls: {stats['total_calls']}")
        print(f"  Active threads: {stats['active_threads']}")
        print(f"  Service time (critical path): {stats['service_time']:.2f}s")
        print(f"  Waiting time: {stats['waiting_time']:.2f}s")

    print(f"\nSession automatically closed.")


def example_openai_adapter():
    """
    Example 3: Using OpenAI-compatible adapter

    This shows how to use Autellix with minimal code changes
    for existing OpenAI-based applications.
    """
    print("\n" + "=" * 60)
    print("Example 3: OpenAI-Compatible Adapter")
    print("=" * 60)

    # Use the OpenAI-compatible adapter
    with AutellixOpenAIAdapter(backend_url="http://localhost:8000") as client:
        # Same interface as OpenAI client
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ],
            temperature=0.7
        )

        print(f"Response: {response['choices'][0]['message']['content']}")

    print("Session automatically closed.")


def example_process_table_direct():
    """
    Example 4: Direct process table manipulation

    This shows how to work directly with the process table
    for advanced use cases.
    """
    print("\n" + "=" * 60)
    print("Example 4: Direct Process Table Access")
    print("=" * 60)

    from frontend import GlobalProcessTable

    # Create process table
    process_table = GlobalProcessTable()

    # Create a program
    pid = "program_123"
    process_table.create_program(pid, is_multithreaded=True)
    print(f"Created program: {pid}")

    # Add some LLM calls
    for i in range(3):
        thread_id = f"thread_{i}"
        call_id = f"call_{i}"
        thread = process_table.add_llm_call(
            pid=pid,
            call_id=call_id,
            thread_id=thread_id,
            prefill_tokens=100,
            engine_id="engine_0"
        )
        print(f"Added call {i}: thread_id={thread.thread_id}, priority={thread.priority}")

    # Simulate call completion
    program = process_table.get_program(pid)
    if program:
        process_table.update_program_metrics(
            pid=pid,
            call_id="call_0",
            service_time=1.5,
            waiting_time=0.3
        )
        print(f"\nUpdated metrics for call_0")
        print(f"  Program service time: {program.service_time:.2f}s")
        print(f"  Program waiting time: {program.waiting_time:.2f}s")

    # Get statistics
    stats = process_table.get_stats()
    print("\n" + "-" * 60)
    print("Process Table Statistics:")
    print(f"  Total programs: {stats['total_programs']}")
    print(f"  Total active calls: {stats['total_active_calls']}")
    print(f"  Total completed calls: {stats['total_completed_calls']}")

    # Cleanup
    process_table.remove_program(pid)
    print(f"\nRemoved program: {pid}")


if __name__ == "__main__":
    # Run all examples
    example_single_threaded_chatbot()
    example_multi_threaded_react()
    example_openai_adapter()
    example_process_table_direct()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
