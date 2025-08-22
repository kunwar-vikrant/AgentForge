import sys
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from llm_providers import get_llm_response
from config import get_config

def setup_logging(config):
    """Configure logging based on configuration."""
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if config.save_logs:
        handlers.append(logging.FileHandler('agent_forge.log'))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)

def validate_inputs(provider: str, use_case: str, config) -> tuple[str, str]:
    """Validate and normalize input parameters."""
    provider = provider.lower().strip()
    use_case = use_case.strip()
    
    supported_providers = {'openai', 'grok', 'ollama'}
    if provider not in supported_providers:
        raise ValueError(f"Unsupported provider '{provider}'. Supported: {', '.join(supported_providers)}")
    
    if not use_case or len(use_case) < 10:
        raise ValueError("Use case description must be at least 10 characters long.")
    
    return provider, use_case

def setup_output_directory(config) -> Path:
    """Create output directory with optional timestamp."""
    base_path = Path(config.output_base_dir)
    
    if config.create_timestamped_dirs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = base_path / timestamp
    else:
        output_path = base_path
    
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

def main():
    # Load configuration
    config = get_config()
    logger = setup_logging(config)
    
    if len(sys.argv) < 3:
        print("Usage: python main.py <llm_provider> <use_case_description>")
        print("Supported providers: openai, grok, ollama")
        sys.exit(1)
    
    try:
        provider, use_case = validate_inputs(sys.argv[1], ' '.join(sys.argv[2:]), config)
        output_path = setup_output_directory(config)
        logger.info(f"Validated inputs - Provider: {provider}, Use case length: {len(use_case)}")
        logger.info(f"Created output directory: {output_path}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Setup error: {e}")
        sys.exit(1)
    
    # Step 1: Planning Agent
    logger.info("Starting planning phase...")
    planning_prompt = (
        """
        You are an expert AI architect.
        Your task is to design a robust structure for an AI agent with the following goal:
        """
        f"{use_case}\n"
        """
        Requirements:
        - Specify all required tools, libraries, and dependencies (with versions if relevant).
        - Outline the logic flow and agentic patterns (e.g., tool calling, memory, planning).
        - Highlight any edge cases or failure handling.
        - Format your response as a clear, step-by-step plan with bullet points or numbered steps.
        - Include testing and validation strategies.
        """
    )

    plan = None
    for attempt in range(1, config.max_retries + 1):
        try:
            logger.info(f"Planning attempt {attempt}/{config.max_retries}")
            plan = get_llm_response(provider, planning_prompt)
            
            if plan and len(plan.strip()) >= config.min_plan_length:
                logger.info(f"Successfully generated plan ({len(plan)} characters)")
                break
            else:
                logger.warning(f"Plan too short or empty (attempt {attempt}). Length: {len(plan) if plan else 0}")
                
        except Exception as e:
            logger.error(f"Planning attempt {attempt} failed: {e}")
            if attempt == config.max_retries:
                raise RuntimeError(f"Failed to generate plan after {config.max_retries} attempts") from e
    
    if not plan:
        raise RuntimeError("Failed to generate a valid agent plan")

    print("\n=== Agent Plan ===\n")
    print(plan.strip())
    
    # Save plan to file
    plan_file = output_path / "agent_plan.txt"
    plan_file.write_text(plan, encoding='utf-8')
    logger.info(f"Plan saved to {plan_file}")
    
    # Step 2: Code Generation Agent
    logger.info("Starting code generation phase...")
    code_gen_prompt = f"""
    Generate Python code for the agent based on this plan:
    
    {plan}
    
    Requirements:
    - Use agentic patterns like tool calling if possible.
    - Include proper error handling and logging.
    - Add docstrings and type hints.
    - Make the code modular and testable.
    - Include configuration management.
    """
    
    agent_code = None
    for attempt in range(1, config.max_retries + 1):
        try:
            logger.info(f"Code generation attempt {attempt}/{config.max_retries}")
            agent_code = get_llm_response(provider, code_gen_prompt)
            
            if agent_code and len(agent_code.strip()) > config.min_code_length:
                logger.info(f"Successfully generated code ({len(agent_code)} characters)")
                break
            else:
                logger.warning(f"Generated code too short (attempt {attempt})")
                
        except Exception as e:
            logger.error(f"Code generation attempt {attempt} failed: {e}")
            if attempt == config.max_retries:
                raise RuntimeError(f"Failed to generate code after {config.max_retries} attempts") from e
    
    if not agent_code:
        raise RuntimeError("Failed to generate agent code")
    
    # Step 3: Testing Agent
    logger.info("Starting test generation phase...")
    test_prompt = f"""
    Write comprehensive tests for this agent code:
    
    {agent_code}
    
    Requirements:
    - Include unit tests and integration tests.
    - Test error handling and edge cases.
    - Suggest improvements or fixes if needed.
    - Use pytest framework.
    """
    
    test_result = None
    for attempt in range(1, config.max_retries + 1):
        try:
            logger.info(f"Test generation attempt {attempt}/{config.max_retries}")
            test_result = get_llm_response(provider, test_prompt)
            
            if test_result and len(test_result.strip()) > 50:
                logger.info(f"Successfully generated tests ({len(test_result)} characters)")
                break
            else:
                logger.warning(f"Generated tests too short (attempt {attempt})")
                
        except Exception as e:
            logger.error(f"Test generation attempt {attempt} failed: {e}")
            if attempt == config.max_retries:
                logger.warning(f"Failed to generate tests after {config.max_retries} attempts: {e}")
                test_result = "# Test generation failed - please write tests manually"
                break
    
    print("\n=== Test Suggestions ===\n")
    print(test_result if test_result else "No tests generated")
    
    # Output the generated files
    try:
        agent_file = output_path / "custom_agent.py"
        agent_file.write_text(agent_code, encoding='utf-8')
        logger.info(f"Agent code saved to {agent_file}")
        
        test_file = output_path / "test_agent.py"
        test_file.write_text(test_result, encoding='utf-8')
        logger.info(f"Tests saved to {test_file}")
        
        # Create a summary file
        summary = f"""
# Agent Generation Summary

**Generated on:** {datetime.now().isoformat()}
**Provider:** {provider}
**Use Case:** {use_case}

## Files Created:
- agent_plan.txt: Detailed agent architecture plan
- custom_agent.py: Generated agent implementation
- test_agent.py: Test suite for the agent

## Plan Length: {len(plan)} characters
## Code Length: {len(agent_code)} characters
## Test Length: {len(test_result) if test_result else 0} characters
"""
        
        summary_file = output_path / "README.md"
        summary_file.write_text(summary, encoding='utf-8')
        
        print(f"\n=== Generation Complete ===")
        print(f"All files saved to: {output_path}")
        print(f"- Agent plan: {plan_file}")
        print(f"- Agent code: {agent_file}")
        print(f"- Tests: {test_file}")
        print(f"- Summary: {summary_file}")
        
    except Exception as e:
        logger.error(f"Failed to save files: {e}")
        print(f"Error saving files: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logger = None
    try:
        main()
    except KeyboardInterrupt:
        if logger:
            logger.info("Process interrupted by user")
        print("\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        # Try to get logger if possible
        try:
            from config import get_config
            config = get_config()
            logger = setup_logging(config)
            logger.error(f"Unexpected error: {e}", exc_info=True)
        except:
            pass
        print(f"An unexpected error occurred: {e}")
        print("Check agent_forge.log for detailed error information.")
        sys.exit(1)