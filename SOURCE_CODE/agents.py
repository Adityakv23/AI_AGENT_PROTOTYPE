"""
Multi-agent system: Planner and Executor agents for intelligent task automation
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from dataclasses import dataclass
from enum import Enum

from config import (
    FINE_TUNED_MODEL_PATH,
    AGENT_CONFIG,
    BASE_MODEL_NAME
)
from rag_system import RAGSystem
from evaluation import Evaluator


class AgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class Plan:
    """Structured plan for task execution"""
    steps: List[str]
    reasoning: str
    confidence: float


class PlannerAgent:
    """
    Planning agent that breaks down tasks into actionable steps
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.state = AgentState.IDLE
        self.config = AGENT_CONFIG
    
    def generate_plan(self, task: str, context: Optional[str] = None) -> Plan:
        """Generate an execution plan for the given task"""
        self.state = AgentState.PLANNING
        
        # Heuristic: provide stronger default plans for explain/summarize tasks
        lower_task = task.lower()
        if any(k in lower_task for k in ["explain", "what is", "what are", "summarize", "summary"]):
            steps = [
                "Outline the key concept(s) and motivation",
                "Describe the core architecture/components",
                "Explain how the mechanism works (focus on attention if relevant)",
                "List advantages, limitations, and typical applications",
                "Provide a concise 4–6 sentence explanation"
            ]
            self.state = AgentState.IDLE
            return Plan(steps=steps, reasoning="Template for explanatory task", confidence=0.85)

        # Create planning prompt
        if context:
            prompt = f"""Context: {context}

Task: {task}

Break down this task into specific actionable steps. Provide:
1. A clear reasoning for your approach
2. Sequential steps to complete the task
3. Expected outcomes for each step

Plan:"""
        else:
            prompt = f"""Task: {task}

Break down this task into specific actionable steps.
Provide sequential steps to complete the task.

Plan:"""
        
        # Generate plan
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + 400,
                temperature=self.config['temperature'],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        
        plan_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        plan_text = plan_text[len(prompt):].strip()
        
        # Parse plan into structured format
        plan = self._parse_plan(plan_text)
        self.state = AgentState.IDLE
        
        return plan
    
    def _parse_plan(self, plan_text: str) -> Plan:
        """Parse plan text into structured Plan object"""
        lines = [l.strip() for l in plan_text.split('\n') if l.strip()]
        
        steps = []
        reasoning = []
        confidence = 0.8  # Default confidence
        
        for line in lines:
            # Look for numbered steps
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                step = line.lstrip('0123456789.-• ')
                if step:
                    steps.append(step)
            elif any(keyword in line.lower() for keyword in ['reason', 'approach', 'strategy']):
                reasoning.append(line)
            elif 'confidence' in line.lower():
                try:
                    confidence = float(line.split()[-1])
                except:
                    pass
        
        return Plan(
            steps=steps if steps else ["Execute task"],
            reasoning=' '.join(reasoning) if reasoning else "Standard approach",
            confidence=confidence
        )
    
    def refine_plan(self, plan: Plan, feedback: str) -> Plan:
        """Refine plan based on execution feedback"""
        prompt = f"""Original Plan:
{plan.steps}

Feedback: {feedback}

Revised Plan:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + 300,
                temperature=self.config['temperature'],
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        
        refined_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        refined_plan = self._parse_plan(refined_text[len(prompt):].strip())
        
        return refined_plan


class ExecutorAgent:
    """
    Executor agent that carries out planned actions
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.state = AgentState.IDLE
        self.config = AGENT_CONFIG
        self.rag = RAGSystem()

    def _baseline_summary(self, text: str, max_sentences: int = 3) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        import re
        # Split into sentences conservatively
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            # Fallback to first 40 words
            words = text.split()
            return (" ".join(words[:40]) + ("..." if len(words) > 40 else "")).strip()
        # Heuristic: pick first 2-3 sentences
        return " ".join(sentences[:max_sentences]).strip()

    def _relevance_score(self, context: Optional[str], result: str) -> float:
        if not context or not isinstance(context, str):
            return 1.0  # Nothing to compare against
        import re
        import string
        def tokens(s: str):
            s = s.lower()
            s = s.translate(str.maketrans('', '', string.punctuation))
            return [t for t in s.split() if len(t) > 2]
        ctx = set(tokens(context))
        res = set(tokens(result))
        if not ctx or not res:
            return 0.0
        overlap = len(ctx & res)
        return overlap / max(1, len(res))
    
    def execute_step(self, step: str, context: Optional[str] = None) -> str:
        """Execute a single step and return results"""
        self.state = AgentState.EXECUTING
        
        step_lower = step.lower()
        # Use direct-context summarization when user provided text. Avoid RAG unless explicitly requested
        if "summar" in step_lower or "review" in step_lower:
            if context and isinstance(context, str) and len(context.strip()) > 0:
                instruction = (
                    "Summarize the following text in 5-7 sentences as a cohesive paragraph,"
                    " focusing on the main claim, how it is achieved, and implications."
                    " Avoid meta commentary, commands, or code."
                )
                prompt = (
                    f"Instruction: {instruction}\n\n"
                    f"Text to summarize:\n{context}\n\n"
                    f"Summary:"
                )
                gen_temperature = 0.3
                do_sample = False
            else:
                # No source text provided: write a topic overview paragraph
                topic = step
                instruction = (
                    "Write a 5-7 sentence overview as a cohesive paragraph for the given topic."
                    " Focus on definition, key mechanisms/ideas, importance, and applications."
                    " Avoid meta commentary, commands, links, or code."
                )
                prompt = (
                    f"Instruction: {instruction}\n\n"
                    f"Topic: {topic}\n\n"
                    f"Overview:"
                )
                gen_temperature = 0.3
                do_sample = False
        # Deterministic explanatory style
        elif any(k in step_lower for k in ["explain", "what is", "what are", "describe"]):
            instruction = (
                "You are a precise technical explainer. In 4-6 sentences, provide a clear,"
                " self-contained explanation aimed at an ML student."
                " Avoid mentioning execution, commands, local machines, or JSON."
            )
            base_context = f"\n\nContext:\n{context}\n" if context else "\n"
            prompt = (
                f"Instruction: {instruction}{base_context}\n"
                f"Question: {step}\n"
                f"Answer:"
            )
            gen_temperature = 0.2
            do_sample = False  # greedy for determinism
        else:
    # Default: treat all generic tasks as explanatory summaries
            instruction = (
                "Write a cohesive 5–7 sentence paragraph explaining or summarizing the given topic."
                " Focus on its definition, working principle, significance, and applications."
                " Avoid meta commentary, code, URLs, or examples from repositories."
            )
            prompt = (
                f"Instruction: {instruction}\n\n"
                f"Topic: {step}\n\n"
                f"Answer:"
            )
            gen_temperature = 0.3
            do_sample = False

        
        # Generate execution
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens=min(320, self.config.get('max_tokens', 640)),
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
            if do_sample:
                gen_kwargs.update(dict(temperature=gen_temperature, top_p=0.9))
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = result[len(prompt):].strip()
        original_result = result
        
        # Prevent infinite "Result:" repetition and strip irrelevant lines
        if result.startswith("Result:"):
            result = result.replace("Result:", "").strip()
        if result.count("Result:") > 3:
            # Stop at first few instances
            parts = result.split("Result:")
            result = parts[0] if len(parts) > 0 else result

        # Remove box-drawing artifacts
        box_chars = ["╭", "╮", "╰", "╯", "│", "─", "━", "┃"]
        for ch in box_chars:
            result = result.replace(ch, " ")

        # Remove irrelevant operational chatter and step prefixes
        banned_phrases = [
            "local machine", "executing commands", "invoke ", " json", " ascii byte",
            "remote processes", " shell ", " terminal ", "powershell", "cmd.exe",
            "separate tab", "available files", "look at your file", " ./", " -f ",
            "component", "extra functionality", "config", "etc-config",
            "github", "repository", "clone", "pull request", "issue tracker", "link:", "http://", "https://",
            "stack overflow", "stackoverflow", "reddit", "quote:", "citation", "as per", "according to"
        ]
        cleaned_lines: List[str] = []
        for l in result.splitlines():
            ll = l.strip()
            if not ll:
                continue
            if ll.lower().startswith("step "):
                continue
            # Remove markdown-style quotes and obvious quoted blocks
            if ll.startswith(">"):
                continue
            if (ll.startswith('"') and ll.endswith('"')) or (ll.startswith('“') and ll.endswith('”')):
                continue
            if any(p in ll.lower() for p in banned_phrases):
                continue
            cleaned_lines.append(ll)
        result = " ".join(cleaned_lines).strip()

        # Limit to 7 sentences for explanations/summaries and ensure paragraph form
        if any(k in step_lower for k in ["explain", "what is", "what are", "describe", "summarize", "summary"]):
            import re
            sentences = re.split(r"(?<=[.!?])\s+", result)
            result = " ".join(sentences[:7]).strip()

        # Fallback if overly aggressive cleaning empties output
        if not result:
            # Use a trimmed version of the original text without leading Result: clutter
            fallback = original_result
            if fallback.startswith("Result:"):
                fallback = fallback.replace("Result:", "").strip()
            # Collapse whitespace and limit length
            fallback = " ".join(fallback.split())
            if any(k in step_lower for k in ["explain", "what is", "what are", "describe", "summarize", "summary"]):
                import re
                sentences = re.split(r"(?<=[.!?])\s+", fallback)
                result = " ".join(sentences[:6]).strip()
            else:
                result = fallback[:500].strip()

        # If still empty or too short, retry with a safer prompt/decoding
        if len(result) < 20:
            retry_instruction = (
                "Provide a cohesive paragraph (5-7 sentences)."
                " Do not include step markers, code, commands, or references to execution."
            )
            retry_prompt = (
                f"Instruction: {retry_instruction}\n\n"
                f"Question: {step}\n"
                f"Context: {context if context else ''}\n\n"
                f"Answer:"
            )
            retry_inputs = self.tokenizer(retry_prompt, return_tensors="pt")
            with torch.no_grad():
                retry_outputs = self.model.generate(
                    **retry_inputs,
                    max_new_tokens=240,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
            retry_text = self.tokenizer.decode(retry_outputs[0], skip_special_tokens=True)
            retry_text = retry_text[len(retry_prompt):].strip()
            if retry_text:
                result = retry_text

        # If summarizing/explaining with provided context, ensure relevance; otherwise fallback to baseline
        if any(k in step_lower for k in ["explain", "what is", "what are", "describe", "summarize", "summary"]) and isinstance(context, str) and context.strip():
            score = self._relevance_score(context, result)
            if score < 0.15:  # too off-topic
                result = self._baseline_summary(context, max_sentences=3)

        # Final safeguard: if original output mentioned stackoverflow or quotes/links heavily, fallback
        lower_orig = (original_result or "").lower()
        if any(k in lower_orig for k in ["stackoverflow", "stack overflow", "http://", "https://", ">", "quote:"]):
            # Prefer context-based baseline for summarize/explain, else keep cleaned result
            if isinstance(context, str) and context.strip():
                result = self._baseline_summary(context, max_sentences=5)

        # Final guardrails: drop highly numeric/meta lines and ensure at least one sentence
        def is_mostly_numeric(s: str) -> bool:
            digits = sum(c.isdigit() for c in s)
            letters = sum(c.isalpha() for c in s)
            return digits > 0 and digits >= letters

        if result:
            parts = [p.strip() for p in result.splitlines() if p.strip()]
            parts = [p for p in parts if not is_mostly_numeric(p) and not p.endswith("%")]
            result = " ".join(parts).strip()

        # If no sentence-ending punctuation and we have context, synthesize a paragraph from context
        if (not any(ch in result for ch in ".!?")) and isinstance(context, str) and context.strip():
            words = context.split()
            snippet = " ".join(words[:80]).strip()
            if not snippet.endswith(('.', '!', '?')):
                snippet += "."
            result = snippet
        
        self.state = AgentState.IDLE
        return result
    
    def evaluate_result(self, step: str, result: str, expected: Optional[str] = None) -> float:
        """Evaluate quality of execution result"""
        self.state = AgentState.EVALUATING
        
        # Simple quality scoring based on length, keywords, etc.
        quality_score = 0.5  # Base score
        
        # Length check
        if 50 <= len(result.split()) <= 500:
            quality_score += 0.2
        
        # Check for complete sentences
        if '.' in result or '!' in result or '?' in result:
            quality_score += 0.2
        
        # Check if result addresses the step
        step_keywords = set(step.lower().split())
        result_keywords = set(result.lower().split())
        overlap = len(step_keywords & result_keywords)
        if overlap > 0:
            quality_score += 0.1 * min(overlap / len(step_keywords), 1.0)
        
        self.state = AgentState.IDLE
        return min(quality_score, 1.0)


class AIAgent:
    """
    Main AI agent coordinating Planner and Executor
    """
    
    def __init__(self, use_finetuned: bool = True):
        self.config = AGENT_CONFIG
        self.use_finetuned = use_finetuned
        
        # Load model
        self._load_model()
        
        # Initialize agents
        self.planner = PlannerAgent(self.model, self.tokenizer)
        self.executor = ExecutorAgent(self.model, self.tokenizer)
        
        # Initialize evaluator
        self.evaluator = Evaluator()
        
        print("✓ AI Agent initialized with multi-agent architecture")
        print("  - Planner Agent: Breaks down tasks into steps")
        print("  - Executor Agent: Carries out planned actions")

    def _detect_intent(self, task: str, context: Optional[str]) -> str:
            """Very simple intent classifier with summarization fallback"""
            t = (task or "").lower().strip()

            # Explicit intent keywords
            if any(k in t for k in ["summarize", "summarise", "summary"]):
                return "summarize"
            if any(k in t for k in ["explain", "what is", "what are", "describe"]):
                return "explain"
            if any(k in t for k in ["compare", "contrast"]):
                return "compare"

            # If only a short topic/abstract was given, assume summarization intent
            if len(t.split()) < 30 and not context:
                return "summarize"

            # Fallback: general intent, but default to summarize-like behavior
            return "summarize"

    
    def _load_model(self):
        """Load the model (fine-tuned or base)"""
        if self.use_finetuned and FINE_TUNED_MODEL_PATH.exists():
            print(f"Loading fine-tuned model from {FINE_TUNED_MODEL_PATH}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(FINE_TUNED_MODEL_PATH))
            self.model = AutoModelForCausalLM.from_pretrained(
                str(FINE_TUNED_MODEL_PATH),
                dtype=torch.float16,
                device_map="auto"
            )
        else:
            if self.use_finetuned:
                print("Fine-tuned model not found. Using base model.")
            print(f"Loading base model: {BASE_MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                dtype=torch.float16,
                device_map="auto"
            )
        
        print(f"✓ Model loaded successfully")
    
    def process_task(self, task: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a task using planning and execution
        Returns comprehensive results
        """
        print(f"\n=== Processing Task ===")
        print(f"Task: {task}\n")
        # Heuristic: if user wrote "Summarize this: <text>", extract <text> as context
        if context is None and isinstance(task, str):
            import re
            # Accept: summarize this [optional colon or dash] <text>
            m = re.match(r"^\s*(summari[sz]e\s+this)\s*[:\-]?\s*(.*)$", task, flags=re.IGNORECASE | re.DOTALL)
            if m:
                extracted = (m.group(2) or "").strip()
                if extracted:
                    context = extracted
                    task = "Summarize the provided text concisely"

        intent = self._detect_intent(task, context)

        results = {
            'task': task,
            'intent': intent,
            'plan': None,
            'execution': [],
            'final_result': None,
            'quality_metrics': None
        }
        
        # Step 1: Generate plan
        print("--- Planning Phase ---")
        plan = self.planner.generate_plan(task, context)
        results['plan'] = {
            'steps': plan.steps,
            'reasoning': plan.reasoning,
            'confidence': plan.confidence
        }
        
        print(f"Generated plan with {len(plan.steps)} steps")
        print(f"Confidence: {plan.confidence:.2f}")
        
        # Step 2: Execute plan
        print("\n--- Execution Phase ---")
        execution_results = []
        
        for i, step in enumerate(plan.steps, 1):
            print(f"\nStep {i}: {step}")
            # Guardrails: if summarize without context, generate a topic overview
            if intent == "summarize" and not (isinstance(context, str) and context.strip()):
                result = self.executor.execute_step("Summarize the topic: " + task, None)
            else:
                result = self.executor.execute_step(step, context)
            quality = self.executor.evaluate_result(step, result)
            
            execution_results.append({
                'step': step,
                'result': result,
                'quality': quality
            })
            
            print(f"✓ Quality: {quality:.2f}")
        
        results['execution'] = execution_results
        
        # Step 3: Compile final result
        print("\n--- Compiling Results ---")
        lower_task = (task or "").lower()
        if any(k in lower_task for k in ["explain", "what is", "what are", "summarize", "summary"]):
            # For explanatory/summarization tasks, return the first clean answer only
            final_result = execution_results[0]['result'] if execution_results else ""
        else:
            final_result = "\n\n".join([
                f"Step {i}: {er['result']}"
                for i, er in enumerate(execution_results, 1)
            ])
        results['final_result'] = final_result
        
        print("✓ Task processing complete")
        
        return results
    
    def summarize_academic_paper(self, paper_text: str) -> str:
        """Specialized method for academic paper summarization"""
        task = "Summarize this academic paper concisely"
        
        result = self.process_task(task, paper_text)
        
        # Extract summary from execution results
        if result['execution'] and len(result['execution']) > 0:
            summary = result['execution'][0]['result']
        else:
            summary = result['final_result']
        
        return summary


if __name__ == "__main__":
    # Test the AI agent
    agent = AIAgent(use_finetuned=False)  # Use base model for testing
    
    # Test task
    test_task = "Analyze the impact of attention mechanisms in deep learning"
    
    result = agent.process_task(test_task)
    
    print("\n=== Final Result ===")
    print(result['final_result'])

