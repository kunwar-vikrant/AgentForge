import sys
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from llm_providers import get_llm_response
from config import get_config


def sanitize_generated_code(text: str) -> str:
    """Remove markdown fences and trailing commentary from LLM code output.

    Rules:
    - Strip leading/trailing whitespace
    - Remove starting ``` (any language) line if present
    - Remove ending ``` line
    - If a section begins with a heading like 'Key improvements' after code, drop it
    - Return cleaned code
    """
    if not text:
        return text
    original = text
    lines = text.strip().splitlines()

    # Remove leading fence line(s)
    while lines and lines[0].strip().startswith("```"):
        lines = lines[1:]

    # Identify if there is an early closing fence followed by commentary; trim from that fence.
    cleaned_lines = []
    fence_closed = False
    for idx, line in enumerate(lines):
        if line.strip().startswith("```"):
            fence_closed = True
            # Stop collecting further code; ignore remainder
            break
        # Stop if an explanation/commentary section begins
        low = line.lower().strip()
        if low.startswith("**explanation") or low.startswith("explanation:"):
            break
        cleaned_lines.append(line)

    # Fallback: if nothing collected (edge case), keep original
    if not cleaned_lines:
        cleaned_lines = [l for l in lines if not l.strip().startswith("```")]

    # Remove trailing commentary markers inside collected lines
    for i, line in enumerate(cleaned_lines):
        low = line.lower()
        if low.startswith("key improvements"):
            cleaned_lines = cleaned_lines[:i]
            break

    cleaned = "\n".join(cleaned_lines).rstrip() + "\n"
    return cleaned

def extract_dependencies(code: str) -> list[str]:
    """Heuristically extract third-party dependencies from generated code.

    Strategy:
    - Parse import lines starting with 'import' or 'from'
    - Map top-level module names to probable PyPI packages
    - Exclude known stdlib modules (coarse list)
    - Return sorted unique package names
    """
    if not code:
        return []
    import re
    stdlib = {
        'sys','os','re','json','typing','logging','pathlib','datetime','math','asyncio','subprocess',
        'functools','itertools','collections','dataclasses','time','random','statistics','heapq','queue','threading',
        'uuid','shutil','tempfile','inspect','textwrap','argparse','enum','types','base64','hashlib','hmac','http',
        'urllib','html','signal','glob','pprint','contextlib','copy','csv','gzip','zipfile','tarfile','io'
    }
    mapping = {
        'spacy': 'spacy',
        'transformers': 'transformers',
        'torch': 'torch',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'chromadb': 'chromadb',
        'langchain': 'langchain',
        'openai': 'openai',
        'requests': 'requests',
        'aiohttp': 'aiohttp',
    }
    pkgs = set()
    for line in code.splitlines():
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        m = re.match(r'^(from|import)\s+([a-zA-Z0-9_\.]+)', line)
        if not m:
            continue
        mod = m.group(2).split('.')[0]
        if mod in mapping and mod not in stdlib:
            pkgs.add(mapping[mod])
        elif mod not in stdlib and mod not in mapping:
            # naive heuristic: likely third-party if not stdlib and not lowercase single letter
            if len(mod) > 1 and mod.isalpha():
                pkgs.add(mod)
    return sorted(pkgs)

def finalize_agent_code(raw_code: str, use_case: str) -> str:
    """Post-process raw LLM code to ensure it's directly runnable without edits.

    Guarantees:
    - Removes markdown fences / trailing commentary (delegates to sanitize_generated_code)
    - Ensures there is a main entrypoint block (if __name__ == "__main__": ...)
    - Adds a lightweight main() if absent, attempting to call an obvious run function
    - Injects a USE_CASE constant for runtime reference
    - Avoids duplicating existing guards
    """
    code = sanitize_generated_code(raw_code)
    # Strip model reasoning / chain-of-thought style tags (<think>...</think>)
    import re
    if '<think>' in code.lower():
        code = re.sub(r'<think>.*?</think>', '', code, flags=re.IGNORECASE | re.DOTALL)
    # Remove leftover lone <think> or </think> lines
    filtered = []
    for line in code.splitlines():
        low = line.strip().lower()
        if low.startswith('<think') or low.startswith('</think'):
            continue
        filtered.append(line)
    code = '\n'.join(filtered)
    # Ensure USE_CASE constant present
    if 'USE_CASE =' not in code:
        header = f'"""Auto-generated agent. Do not edit manually unless necessary.\nUse case: {use_case}\n"""\n\nUSE_CASE = {json.dumps(use_case)}\n\n'
    else:
        header = ''

    # Detect existing main guard
    # Detect main guard (single or double quotes)
    has_main_guard = ('if __name__ == "__main__"' in code) or ("if __name__ == '__main__'" in code)
    has_main_func = 'def main(' in code

    # Heuristics for run function
    run_invocation = None
    if 'def run_agent' in code:
        run_invocation = 'run_agent()'
    elif 'class Agent' in code and '.run(' in code:
        run_invocation = 'agent = Agent()\n    agent.run()'
    elif 'def run(' in code:
        run_invocation = 'run()'

    extra = ''
    if not has_main_func:
        # Provide a minimal main referencing detected run function or noop
        body = f"    print('Starting generated agent for: ' + USE_CASE)\n"
        if run_invocation:
            body += f"    {run_invocation}\n"
        extra += f"\ndef main():\n{body}\n"

    if not has_main_guard:
        guard_body = '    main()' if not run_invocation else '    main()'
        extra += f"\nif __name__ == '__main__':\n{guard_body}\n"

    finalized = header + code.rstrip() + '\n' + extra
    return finalized

def evaluate_generated_code(code: str, min_lines: int = 40) -> dict:
    """Heuristically evaluate generated code quality.

    Returns dict with keys:
    - ok: bool overall pass
    - issues: list of textual issue descriptions
    - stats: supporting metrics
    Criteria (fail if any true):
    - Too short (< min_lines non-empty lines)
    - Contains 'pass' placeholder lines > 2
    - Contains TODO / FIXME markers
    - Comment+docstring line ratio > 0.55 (too much instruction vs code)
    - Contains phrases suggesting instructions not code (e.g., 'Below is', 'The following steps')
    - Missing any function definitions besides main (>=1 required)
    """
    import re
    lines = [l for l in code.splitlines() if l.strip()]
    non_empty = len(lines)
    issue_list: list[str] = []
    pass_lines = sum(1 for l in lines if re.match(r'^\s*pass\s*$', l))
    todo_markers = sum(1 for l in lines if re.search(r'TODO|FIXME|placeholder', l, re.IGNORECASE))
    comment_lines = 0
    in_doc = False
    doc_lines = 0
    for l in code.splitlines():
        stripped = l.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            # Toggle docstring state; count line
            doc_lines += 1
            if stripped.count('"""') == 1 or stripped.count("'''") == 1:
                in_doc = not in_doc
            continue
        if in_doc:
            doc_lines += 1
        if stripped.startswith('#'):
            comment_lines += 1
    comment_ratio = (comment_lines + doc_lines) / max(1, non_empty)
    instr_markers = sum(1 for l in lines if re.search(r'Below is|The following steps|You should|Step 1:', l))
    think_blocks = ('<think>' in code.lower()) or ('</think>' in code.lower())
    func_defs = sum(1 for l in lines if re.match(r'^\s*def\s+\w+\(', l))

    if non_empty < min_lines:
        issue_list.append(f"Code too short ({non_empty} < {min_lines} lines)")
    if pass_lines > 2:
        issue_list.append(f"Too many placeholder 'pass' lines ({pass_lines})")
    if todo_markers > 0:
        issue_list.append(f"Found TODO/FIXME markers ({todo_markers})")
    if comment_ratio > 0.55:
        issue_list.append(f"Excessive commentary ratio {comment_ratio:.2f} > 0.55")
    if instr_markers > 2:
        issue_list.append("Appears to contain instructional prose instead of code")
    if func_defs < 2:  # expect at least one real function plus maybe main
        issue_list.append(f"Insufficient function definitions ({func_defs})")
    if think_blocks:
        issue_list.append("Contains <think> reasoning block (remove chain-of-thought from final code)")

    return {
        'ok': len(issue_list) == 0,
        'issues': issue_list,
        'stats': {
            'non_empty': non_empty,
            'pass_lines': pass_lines,
            'todo_markers': todo_markers,
            'comment_ratio': comment_ratio,
            'instr_markers': instr_markers,
            'func_defs': func_defs,
            'think_blocks': think_blocks,
        }
    }

def generate_fallback_agent(use_case: str) -> str:
    """Produce a deterministic minimal functional agent for the given use case.

    Focused on the trading risk summarizer scenario but generic enough for other tasks.
    Uses only stdlib; provides:
    - Data loading (CSV) with schema inference
    - Simple anomaly detection (z-score threshold)
    - Classification heuristic
    - Report building & email sending stub (prints instead of actual SMTP unless env vars provided)
    """
    use_case_json = json.dumps(use_case)
    # Use doubled braces where we want braces to appear in generated code for f-strings executed at runtime.
    template = (
        '"""Fallback Generated Agent\n'
        f'Use case: {use_case}\n'
        'This file was produced by the fallback generator because LLM outputs did not meet quality criteria.\n'
        '"""\n'
        'from __future__ import annotations\n'
        'import csv, statistics, os, smtplib, ssl, sys\n'
        'from pathlib import Path\n'
        'from typing import List, Dict, Any\n\n'
        f'USE_CASE = {use_case_json}\n\n'
        'def load_rows(csv_path: str) -> List[Dict[str, Any]]:\n'
        '    """Load CSV rows into list of dicts. Expects headers including value/volatility if present."""\n'
        '    p = Path(csv_path)\n'
        '    if not p.exists():\n'
        '        raise FileNotFoundError(f"CSV not found: {csv_path}")\n'
        '    with p.open(\'r\', newline=\'\') as f:\n'
        '        reader = csv.DictReader(f)\n'
        '        return [r for r in reader]\n\n'
        'def compute_stats(rows: List[Dict[str, Any]], value_key: str=\'value\') -> Dict[str, float]:\n'
        '    vals = []\n'
        '    for r in rows:\n'
        '        try:\n'
        '            vals.append(float(r.get(value_key, \"nan\")))\n'
        '        except ValueError:\n'
        '            continue\n'
        '    if not vals:\n'
        '        return {\'mean\': 0.0, \'stdev\': 0.0}\n'
        '    if len(vals) < 2:\n'
        '        return {\'mean\': vals[0], \'stdev\': 0.0}\n'
        '    return {\'mean\': statistics.mean(vals), \'stdev\': statistics.pstdev(vals)}\n\n'
        'def detect_anomalies(rows: List[Dict[str, Any]], value_key: str=\'value\', z_thresh: float=2.0) -> List[Dict[str, Any]]:\n'
        '    stats = compute_stats(rows, value_key)\n'
        '    mean, stdev = stats[\'mean\'], stats[\'stdev\']\n'
        '    anomalies = []\n'
        '    if stdev == 0:\n'
        '        return anomalies\n'
        '    for r in rows:\n'
        '        try:\n'
        '            v = float(r.get(value_key, \"nan\"))\n'
        '        except ValueError:\n'
        '            continue\n'
        '        z = (v - mean) / stdev if stdev else 0\n'
        '        if abs(z) >= z_thresh:\n'
        '            row_copy = dict(r)\n'
        '            row_copy[\'_zscore\'] = round(z, 2)\n'
        '            anomalies.append(row_copy)\n'
        '    return anomalies\n\n'
        'def classify(anomaly: Dict[str, Any]) -> str:\n'
        '    z = abs(float(anomaly.get(\'_zscore\', 0)))\n'
        '    if z >= 4: return \'CRITICAL\'\n'
        '    if z >= 3: return \'HIGH\'\n'
        '    if z >= 2.5: return \'MEDIUM\'\n'
        '    return \'LOW\'\n\n'
        'def build_summary(anomalies: List[Dict[str, Any]], top_n: int=5) -> str:\n'
        '    if not anomalies:\n'
        '        return \'No anomalies detected.\'\n'
        '    lines = [\'Anomaly Summary:\']\n'
        '    counts: Dict[str, int] = {}\n'
        '    for a in anomalies:\n'
        '        sev = classify(a)\n'
        '        counts[sev] = counts.get(sev, 0) + 1\n'
        '    lines.append(\'Counts: \'+ \' ,\'.replace(\' \' ,\'\') .join(f"{k}:{v}" for k, v in sorted(counts.items())))\n'
        '    lines.append(\'\\nTop examples:\')\n'
        '    for a in anomalies[:top_n]:\n'
        '        sev = classify(a)\n'
        '        preview = {k: a.get(k) for k in list(a)[:5]}\n'
        '        lines.append("  - sev={s} z={z} raw={p}".format(s=sev, z=a.get(\'_zscore\'), p=preview))\n'
        '    return "\\n".join(lines)\n\n'
        'def maybe_send_email(subject: str, body: str) -> None:\n'
        '    host = os.getenv(\'AGENT_SMTP_HOST\')\n'
        '    user = os.getenv(\'AGENT_SMTP_USER\')\n'
        '    pwd = os.getenv(\'AGENT_SMTP_PASS\')\n'
        '    to_addr = os.getenv(\'AGENT_REPORT_TO\')\n'
        '    if not all([host, user, pwd, to_addr]):\n'
        '        print(\'\\n[Email disabled] Set AGENT_SMTP_HOST/USER/PASS/REPORT_TO to enable email.\\n\')\n'
        '        print(\'Subject:\', subject)\n'
        '        print(body)\n'
        '        return\n'
        '    ctx = ssl.create_default_context()\n'
        '    msg = f"From: {user}\\nTo: {to_addr}\\nSubject: {subject}\\n\\n{body}".encode(\'utf-8\')\n'
        '    with smtplib.SMTP_SSL(host, 465, context=ctx) as s:\n'
        '        s.login(user, pwd)\n'
        '        s.sendmail(user, [to_addr], msg)\n'
        '    print(\'Email sent to\', to_addr)\n\n'
        'def run_pipeline(csv_path: str) -> None:\n'
        '    rows = load_rows(csv_path)\n'
        '    anomalies = detect_anomalies(rows)\n'
        '    summary = build_summary(anomalies)\n'
        '    maybe_send_email(\'Trading Risk Summary\', summary)\n'
        '    print(summary)\n\n'
        'def main():\n'
        '    if len(sys.argv) < 2:\n'
        '        print(\'Usage: python custom_agent.py <data.csv>\')\n'
        '        return\n'
        '    run_pipeline(sys.argv[1])\n\n'
        'if __name__ == "__main__":\n'
        '    main()\n'
    )
    return template

def create_package_assets(output_path: Path, deps: list[str], logger: logging.Logger) -> None:
    """Create helper run scripts and optional Dockerfile for non-technical users."""
    # run.sh
    run_sh = output_path / 'run.sh'
    run_sh.write_text("""#!/usr/bin/env bash
set -e
if [ ! -d .venv ]; then
  python -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip >/dev/null 2>&1
if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
python custom_agent.py "$@"
""", encoding='utf-8')
    run_sh.chmod(0o755)

    # run.bat (Windows)
    run_bat = output_path / 'run.bat'
    run_bat.write_text("""@echo off
IF NOT EXIST .venv (python -m venv .venv)
call .venv\\Scripts\\activate
IF EXIST requirements.txt pip install -r requirements.txt
python custom_agent.py %*
""", encoding='utf-8')

    # Dockerfile
    dockerfile = output_path / 'Dockerfile'
    base = 'python:3.11-slim'
    dockerfile.write_text(f"""FROM {base}
WORKDIR /app
COPY . /app
RUN python -m pip install --upgrade pip && \\
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
CMD ["python", "custom_agent.py"]
""", encoding='utf-8')

    # .env template if environment keys referenced
    env_file = output_path / '.env.example'
    env_keys = ["DATABASE_URL", "MODEL_PATH", "OPENAI_API_KEY"]
    env_file.write_text("\n".join(f"{k}=" for k in env_keys) + "\n", encoding='utf-8')

    logger.info("Created run scripts, Dockerfile, and .env example for generated agent")

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
    
    supported_providers = {'openai', 'grok', 'ollama', 'groq', 'anthropic'}
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

def run_generation(provider: str, use_case: str):
    """Programmatic single-agent generation entrypoint.

    Returns output Path containing generated artifacts.
    """
    config = get_config()
    logger = setup_logging(config)
    provider, use_case = validate_inputs(provider, use_case, config)
    output_path = setup_output_directory(config)
    logger.info(f"Validated inputs - Provider: {provider}, Use case length: {len(use_case)}")
    logger.info(f"Created output directory: {output_path}")
    
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
    quality_attempts = 0
    max_quality_attempts = 3
    last_issues: list[str] = []
    for attempt in range(1, config.max_retries + 1):
        try:
            logger.info(f"Code generation attempt {attempt}/{config.max_retries}")
            base_code = get_llm_response(provider, code_gen_prompt)
            if not base_code or len(base_code.strip()) <= config.min_code_length:
                logger.warning(f"Generated code too short (attempt {attempt})")
                continue
            # Evaluate quality; refine if necessary
            eval_result = evaluate_generated_code(base_code)
            if eval_result['ok']:
                agent_code = base_code
                logger.info(f"Accepted code (chars={len(agent_code)}, lines={eval_result['stats']['non_empty']})")
                break
            else:
                last_issues = eval_result['issues']
                logger.warning(f"Quality issues detected: {last_issues}")
                if quality_attempts + 1 >= max_quality_attempts:
                    logger.warning("Reached max quality refinement attempts; accepting last version with issues.")
                    agent_code = base_code
                    break
                refinement_prompt = f"""You previously generated code for this agent use case: {use_case}\nIssues detected:\n- {'\n- '.join(last_issues)}\nRegenerate ONLY executable Python code (no reasoning, no <think> tags, no prose paragraphs).\nRequirements:\n- Concrete implementations (no 'pass', no TODO/FIXME).\n- >=2 functional helper functions plus main.\n- >=40 non-empty lines of real code.\n- Commentary (comments+docstrings) <= 50% of lines.\n- Provide a main() and if __name__ == '__main__' guard.\n- NO explanation outside code.\nReturn just Python code fenced or unfenced; avoid chain-of-thought.\n"""
                quality_attempts += 1
                logger.info(f"Refinement attempt {quality_attempts}/{max_quality_attempts}")
                try:
                    refined = get_llm_response(provider, refinement_prompt)
                    if refined and len(refined.strip()) > config.min_code_length:
                        refined_eval = evaluate_generated_code(refined)
                        if refined_eval['ok']:
                            agent_code = refined
                            logger.info("Refined code accepted")
                            break
                        else:
                            last_issues = refined_eval['issues']
                            logger.warning(f"Refined code still has issues: {last_issues}")
                            agent_code = refined  # keep latest; may refine again
                    else:
                        logger.warning("Refinement produced too short output; keeping previous version")
                except Exception as ref_e:
                    logger.warning(f"Refinement call failed: {ref_e}")
                    agent_code = base_code
                    break
        except Exception as e:
            logger.error(f"Code generation attempt {attempt} failed: {e}")
            if attempt == config.max_retries and not agent_code:
                raise RuntimeError(f"Failed to generate code after {config.max_retries} attempts") from e

    if not agent_code:
        logger.warning("Falling back to deterministic template due to generation failure")
        agent_code = generate_fallback_agent(use_case)
    else:
        # Final evaluation; if still not ok -> fallback
        final_eval = evaluate_generated_code(agent_code)
        if not final_eval['ok']:
            logger.warning(f"Final code still has issues {final_eval['issues']} -> using fallback template")
            agent_code = generate_fallback_agent(use_case)
        elif last_issues:
            logger.info(f"Proceeding with code despite minor remaining issues: {last_issues}")
    
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
        # 1. Write agent code (finalized) & tests
        agent_file = output_path / "custom_agent.py"
        finalized_code = finalize_agent_code(agent_code, use_case)
        agent_file.write_text(finalized_code, encoding='utf-8')
        logger.info(f"Agent code saved to {agent_file}")

        test_file = output_path / "test_agent.py"
        test_file.write_text(sanitize_generated_code(test_result), encoding='utf-8')
        logger.info(f"Tests saved to {test_file}")

        # 2. Dependencies
        deps = extract_dependencies(agent_code)
        if deps:
            (output_path / 'requirements.txt').write_text('\n'.join(deps) + '\n', encoding='utf-8')
            logger.info(f"Requirements saved ({len(deps)} packages)")
        else:
            logger.info("No third-party dependencies detected")

        # 3. Package assets (best-effort)
        try:
            create_package_assets(output_path, deps, logger)
        except Exception as pkg_e:
            logger.warning(f"Asset creation issue (non-fatal): {pkg_e}")

        # 4. Runtime config JSON
        cfg = {"use_case": use_case, "provider": provider, "generated": datetime.now().isoformat()}
        (output_path / 'agent_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        logger.info("agent_config.json created")

        # 5. Install instructions
        install_instructions = """\n## Quick Start (No Technical Experience Needed)\n\n### Option A: One-Line (macOS/Linux)\n```bash\n./run.sh\n```\n\n### Option B: Windows\n```bat\nrun.bat\n```\n\n### Manual Steps\n```bash\npython -m venv .venv\nsource .venv/bin/activate  # or .venv\\Scripts\\activate on Windows\n{pip_install}\npython custom_agent.py\n```\n\n### Docker (Optional)\n```bash\ndocker build -t generated-agent .\ndocker run --rm generated-agent\n```\n""".format(
            pip_install=("pip install -r requirements.txt" if deps else "# (No third-party dependencies detected)")
        )

        summary = f"""# Agent Generation Summary\n\n**Generated on:** {datetime.now().isoformat()}\n**Provider:** {provider}\n**Use Case:** {use_case}\n\n## Files Created:\n- agent_plan.txt: Detailed agent architecture plan\n- custom_agent.py: Generated agent implementation\n- test_agent.py: Test suite for the agent\n- agent_config.json: Runtime metadata & editable simple config\n\n## Plan Length: {len(plan)} characters\n## Code Length: {len(agent_code)} characters\n## Test Length: {len(test_result) if test_result else 0} characters\n## Dependencies Detected: {', '.join(deps) if deps else 'None'}\n\n## Zero-Edit Guarantee\nThis agent was post-processed to ensure a runnable entrypoint (main()) and a main guard. Execute with the provided scripts or directly via `python custom_agent.py` with no code modifications required.\n{install_instructions}\n"""
        (output_path / "README.md").write_text(summary, encoding='utf-8')

        print("\n=== Generation Complete ===")
        print(f"All files saved to: {output_path}")
        print(f"- Agent plan: {plan_file}")
        print(f"- Agent code: {agent_file}")
        print(f"- Tests: {test_file}")
        print(f"- Summary: {(output_path / 'README.md')}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to save files: {e}")
        print(f"Error saving files: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logger = None
    try:
        if len(sys.argv) < 3:
            print("Usage: python main.py <llm_provider> <use_case_description>")
            print("Supported providers: openai, grok, groq, anthropic, ollama")
            sys.exit(1)
        run_generation(sys.argv[1], ' '.join(sys.argv[2:]))
    except KeyboardInterrupt:
        if logger:
            logger.info("Process interrupted by user")
        print("\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        try:
            from config import get_config
            config = get_config()
            logger = setup_logging(config)
            logger.error(f"Unexpected error: {e}", exc_info=True)
        except Exception:
            pass
        print(f"An unexpected error occurred: {e}")
        print("Check agent_forge.log for detailed error information.")
        sys.exit(1)