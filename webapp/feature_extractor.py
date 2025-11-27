from __future__ import annotations

import difflib
import re
import ast
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from radon.complexity import cc_visit

# Import function extractor
try:
    from function_extractor import FunctionExtractor, FunctionInfo
except ImportError:
    from .function_extractor import FunctionExtractor, FunctionInfo

SAFE_FEATURES = [
    "added_lines",
    "deleted_lines",
    "nloc",
    "complexity",
    "token_count",
    "commit_msg_length",
    "commit_msg_word_count",
    "commit_msg_char_count",
    "total_changes",
    "net_lines",
    "code_churn",
    # Performance-aware metrics
    "sync_constructs_count",
    "max_loop_nesting_depth",
    "nested_loops_count",
]


def _count_added_deleted(before: str, after: str) -> Tuple[int, int]:
    diff = difflib.unified_diff(
        before.splitlines(keepends=False),
        after.splitlines(keepends=False),
        n=0,
    )
    added = deleted = 0
    for line in diff:
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            deleted += 1
    return added, deleted


def _cyclomatic_complexity(code: str) -> float:
    try:
        blocks = cc_visit(code)
    except Exception:
        return 0.0

    if not blocks:
        return 0.0
    complexities = [block.complexity for block in blocks]
    return float(np.mean(complexities))


def _token_count(code: str) -> int:
    return len(re.findall(r"\w+", code))


def _nloc(code: str) -> int:
    return len([line for line in code.splitlines() if line.strip()])


def _count_synchronization_constructs(code: str) -> int:
    """
    Count synchronization constructs in code.
    Detects: locks, mutexes, semaphores, synchronized blocks, etc.
    Works for multiple languages (Python, Java, C++, etc.)
    """
    if not code:
        return 0
    
    # Python-specific patterns
    python_patterns = [
        r'\block\b',  # lock
        r'\bRLock\b',  # RLock
        r'\bSemaphore\b',  # Semaphore
        r'\bCondition\b',  # Condition
        r'\bEvent\b',  # Event
        r'\bBarrier\b',  # Barrier
        r'\bwith\s+.*lock',  # with lock:
        r'\bthreading\.Lock',  # threading.Lock
        r'\bthreading\.RLock',  # threading.RLock
        r'\bmultiprocessing\.Lock',  # multiprocessing.Lock
        r'\bmultiprocessing\.RLock',  # multiprocessing.RLock
        r'\b@lock',  # decorator
    ]
    
    # Java-specific patterns
    java_patterns = [
        r'\bsynchronized\s*\(',  # synchronized(...)
        r'\bsynchronized\s+\w+',  # synchronized method
        r'\bLock\b',  # Lock interface
        r'\bReentrantLock\b',  # ReentrantLock
        r'\bSemaphore\b',  # Semaphore
        r'\bCountDownLatch\b',  # CountDownLatch
        r'\bCyclicBarrier\b',  # CyclicBarrier
        r'\bReadWriteLock\b',  # ReadWriteLock
        r'\b\.lock\(\)',  # .lock()
        r'\b\.unlock\(\)',  # .unlock()
    ]
    
    # C++ patterns
    cpp_patterns = [
        r'\bstd::mutex\b',  # std::mutex
        r'\bstd::lock_guard\b',  # std::lock_guard
        r'\bstd::unique_lock\b',  # std::unique_lock
        r'\bstd::shared_lock\b',  # std::shared_lock
        r'\bstd::condition_variable\b',  # std::condition_variable
        r'\bstd::semaphore\b',  # std::semaphore
        r'\b\.lock\(\)',  # .lock()
        r'\b\.unlock\(\)',  # .unlock()
    ]
    
    # Combine all patterns
    all_patterns = python_patterns + java_patterns + cpp_patterns
    
    count = 0
    for pattern in all_patterns:
        matches = re.findall(pattern, code, re.IGNORECASE)
        count += len(matches)
    
    return count


def _calculate_max_loop_nesting_depth(code: str) -> Tuple[int, int]:
    """
    Calculate maximum loop nesting depth and count of nested loops.
    Returns: (max_depth, nested_loops_count)
    """
    if not code:
        return 0, 0
    
    # Try AST parsing for Python code
    try:
        tree = ast.parse(code)
        max_depth = 0
        nested_count = 0
        
        def visit_node(node, depth=0):
            nonlocal max_depth, nested_count
            if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
                current_depth = depth + 1
                max_depth = max(max_depth, current_depth)
                if current_depth > 1:
                    nested_count += 1
                # Visit children
                for child in ast.iter_child_nodes(node):
                    visit_node(child, current_depth)
            else:
                # Visit children with same depth
                for child in ast.iter_child_nodes(node):
                    visit_node(child, depth)
        
        visit_node(tree)
        return max_depth, nested_count
    except (SyntaxError, ValueError):
        # Fallback to regex-based detection for non-Python code
        return _calculate_loop_nesting_regex(code)


def _calculate_loop_nesting_regex(code: str) -> Tuple[int, int]:
    """
    Fallback method using regex to detect loop nesting.
    Works for Java, C++, JavaScript, etc.
    """
    # Common loop patterns across languages
    loop_patterns = [
        r'\bfor\s*\(',  # for (
        r'\bwhile\s*\(',  # while (
        r'\bdo\s*\{',  # do {
        r'\bforeach\s*\(',  # foreach (
    ]
    
    lines = code.split('\n')
    max_depth = 0
    nested_count = 0
    depth_stack = []
    
    for line in lines:
        # Count opening braces (indicates nesting)
        open_braces = line.count('{')
        close_braces = line.count('}')
        
        # Check for loop keywords
        is_loop = any(re.search(pattern, line, re.IGNORECASE) for pattern in loop_patterns)
        
        if is_loop:
            current_depth = len(depth_stack) + 1
            max_depth = max(max_depth, current_depth)
            if current_depth > 1:
                nested_count += 1
            depth_stack.append('loop')
        
        # Update depth based on braces
        for _ in range(open_braces):
            if depth_stack:
                depth_stack.append('block')
        for _ in range(close_braces):
            if depth_stack:
                depth_stack.pop()
    
    return max_depth, nested_count


def build_feature_vector(
    before_code: str,
    after_code: str,
    commit_message: str,
    process_metrics: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compute the pre-commit metrics used during model training.
    Now includes performance-aware metrics: synchronization constructs and loop nesting depth.

    Parameters
    ----------
    before_code : str
        The code before the change (parent revision). May be empty.
    after_code : str
        The code after the change (current revision).
    commit_message : str
        Commit message associated with the change.
    process_metrics : Optional[Dict]
        Optional process metrics to include.
    """
    added, deleted = _count_added_deleted(before_code or "", after_code or "")
    churn = added + deleted
    net_lines = added - deleted

    # Calculate performance-aware metrics
    sync_count = _count_synchronization_constructs(after_code)
    max_loop_depth, nested_loops = _calculate_max_loop_nesting_depth(after_code)

    stats = {
        "added_lines": float(added),
        "deleted_lines": float(deleted),
        "nloc": float(_nloc(after_code)),
        "complexity": float(_cyclomatic_complexity(after_code)),
        "token_count": float(_token_count(after_code)),
        "commit_msg_length": float(len(commit_message)),
        "commit_msg_word_count": float(len(commit_message.split())),
        "commit_msg_char_count": float(len(commit_message.replace(" ", ""))),
        "total_changes": float(churn),
        "net_lines": float(net_lines),
        "code_churn": float(churn),
        # Performance-aware metrics
        "sync_constructs_count": float(sync_count),
        "max_loop_nesting_depth": float(max_loop_depth),
        "nested_loops_count": float(nested_loops),
    }

    # Add process metrics if provided
    if process_metrics:
        stats.update(process_metrics)
    
    # Use only safe features for DataFrame (process metrics are optional)
    features_df = pd.DataFrame(
        [[stats.get(name, 0.0) for name in SAFE_FEATURES]],
        columns=SAFE_FEATURES,
    )
    return features_df, stats


def build_function_level_features(
    before_code: str,
    after_code: str,
    commit_message: str,
    language: Optional[str] = None,
    filename: Optional[str] = None,
    process_metrics: Optional[Dict] = None,
) -> List[Tuple[pd.DataFrame, Dict[str, float], FunctionInfo]]:
    """
    Extract features at function level.
    
    Parameters
    ----------
    before_code : str
        The code before the change (parent revision). May be empty.
    after_code : str
        The code after the change (current revision).
    commit_message : str
        Commit message associated with the change.
    language : Optional[str]
        Programming language (auto-detected if not provided).
    filename : Optional[str]
        Filename for language detection.
    process_metrics : Optional[Dict]
        Optional process metrics to include.
    
    Returns
    -------
    List of tuples: (features_df, stats_dict, function_info)
        One entry per function in the code.
    """
    extractor = FunctionExtractor()
    before_funcs, after_funcs = extractor.extract_functions_from_diff(
        before_code, after_code, language, filename
    )
    
    results = []
    
    # Process each function in after_code
    for after_func in after_funcs:
        # Find corresponding function in before_code
        before_func = None
        for bf in before_funcs:
            if bf.name == after_func.name and bf.class_name == after_func.class_name:
                before_func = bf
                break
        
        before_func_code = before_func.code if before_func else ""
        after_func_code = after_func.code
        
        # Extract features for this function
        features_df, stats = build_feature_vector(
            before_func_code,
            after_func_code,
            commit_message,
            process_metrics
        )
        
        # Add function metadata to stats
        stats['function_name'] = after_func.name
        stats['function_start_line'] = after_func.start_line
        stats['function_end_line'] = after_func.end_line
        stats['is_method'] = after_func.is_method
        stats['class_name'] = after_func.class_name or ""
        
        results.append((features_df, stats, after_func))
    
    # If no functions found, return file-level features as fallback
    if not results:
        features_df, stats = build_feature_vector(
            before_code,
            after_code,
            commit_message,
            process_metrics
        )
        # Create a dummy function info
        dummy_func = FunctionInfo(
            name="<file_level>",
            start_line=1,
            end_line=len(after_code.split('\n')) if after_code else 1,
            code=after_code,
            language=language or "unknown",
            is_method=False,
            class_name=None,
            parameters=[]
        )
        results.append((features_df, stats, dummy_func))
    
    return results

