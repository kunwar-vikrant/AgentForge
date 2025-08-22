import sys
import json
from llm_providers import get_llm_response  

def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <llm_provider> <use_case_description>")
        sys.exit(1)
    
    provider = sys.argv[1].lower()  # 'openai', 'grok', or 'ollama'
    use_case = ' '.join(sys.argv[2:])
    
    # Step 1: Planning Agent
    if not use_case.strip():
        print("Error: Use case description is empty.")
        sys.exit(1)

    planning_prompt = (
        """
        You are an expert AI architect.
        Your task is to design a robust structure for an AI agent with the following goal:
        """
        f"{use_case.strip()}\n"
        """
        Requirements:
        - Specify all required tools, libraries, and dependencies (with versions if relevant).
        - Outline the logic flow and agentic patterns (e.g., tool calling, memory, planning).
        - Highlight any edge cases or failure handling.
        - Format your response as a clear, step-by-step plan with bullet points or numbered steps.
        """
    )

    max_retries = 2
    for attempt in range(1, max_retries + 1):
        try:
            plan = get_llm_response(provider, planning_prompt)
            if plan and len(plan.strip()) > 10:
                break
            else:
                print(f"Warning: Received empty or too short plan (attempt {attempt}). Retrying...")
        except Exception as e:
            print(f"Error during planning step (attempt {attempt}): {e}")
            if attempt == max_retries:
                sys.exit(1)
    else:
        print("Failed to generate a valid agent plan after retries.")
        sys.exit(1)

    print("\n=== Agent Plan ===\n")
    print(plan.strip())
    
    # Step 2: Code Generation Agent
    code_gen_prompt = f"Generate Python code for the agent based on this plan: {plan}. Use agentic patterns like tool calling if possible."
    agent_code = get_llm_response(provider, code_gen_prompt)
    
    # Step 3: Testing Agent
    test_prompt = f"Write a simple test for this agent code: {agent_code}. Suggest fixes if needed."
    test_result = get_llm_response(provider, test_prompt)
    print("Test Suggestions:\n", test_result)
    
    # Output the generated agent
    with open("custom_agent.py", "w") as f:
        f.write(agent_code)
    print("Generated agent saved to custom_agent.py")

if __name__ == "__main__":
    main()