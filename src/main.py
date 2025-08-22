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
    planning_prompt = (
        f"You are an expert AI architect.\n"
        f"Task: Outline a robust structure for an AI agent that will: {use_case}.\n"
        "Be specific about required tools, logic flow, and dependencies.\n"
        "Format your response as a clear, step-by-step plan."
    )
    try:
        plan = get_llm_response(provider, planning_prompt)
    except Exception as e:
        print(f"Error during planning step: {e}")
        sys.exit(1)
    print("\n=== Agent Plan ===\n", plan.strip() if plan else "No plan generated.")
    
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