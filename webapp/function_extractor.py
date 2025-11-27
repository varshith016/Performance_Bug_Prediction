"""
Function-Level Code Extraction
Extracts functions/methods from code for function-level analysis.
"""

import ast
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FunctionInfo:
    """Information about a function/method."""
    name: str
    start_line: int
    end_line: int
    code: str
    language: str
    is_method: bool = False
    class_name: Optional[str] = None
    parameters: List[str] = None


class FunctionExtractor:
    """Extract functions/methods from code."""
    
    def __init__(self):
        self.supported_languages = ['python', 'java', 'javascript', 'cpp', 'c']
    
    def detect_language(self, code: str, filename: Optional[str] = None) -> str:
        """
        Detect programming language from code or filename.
        
        Args:
            code: Source code
            filename: Optional filename with extension
        
        Returns:
            Language name ('python', 'java', etc.)
        """
        if filename:
            ext = filename.split('.')[-1].lower()
            lang_map = {
                'py': 'python',
                'java': 'java',
                'js': 'javascript',
                'jsx': 'javascript',
                'cpp': 'cpp',
                'cc': 'cpp',
                'cxx': 'cpp',
                'c': 'c',
                'h': 'c',
                'hpp': 'cpp',
            }
            if ext in lang_map:
                return lang_map[ext]
        
        # Heuristic detection from code
        if re.search(r'\bdef\s+\w+\s*\(', code):
            return 'python'
        elif re.search(r'\bpublic\s+\w+\s+\w+\s*\(', code) or re.search(r'\bprivate\s+\w+\s+\w+\s*\(', code):
            return 'java'
        elif re.search(r'\bfunction\s+\w+\s*\(', code) or re.search(r'\bconst\s+\w+\s*=\s*\(', code):
            return 'javascript'
        elif re.search(r'\w+\s+\w+\s*\([^)]*\)\s*\{', code):
            return 'cpp'
        
        return 'python'  # Default
    
    def extract_python_functions(self, code: str) -> List[FunctionInfo]:
        """
        Extract functions from Python code using AST.
        
        Args:
            code: Python source code
        
        Returns:
            List of FunctionInfo objects
        """
        functions = []
        lines = code.split('\n')
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    # Get function name
                    func_name = node.name
                    
                    # Get line numbers
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                    
                    # Get function code
                    func_code = '\n'.join(lines[start_line - 1:end_line])
                    
                    # Get parameters
                    params = [arg.arg for arg in node.args.args]
                    
                    # Check if it's a method (has 'self' or 'cls' as first param)
                    is_method = len(params) > 0 and params[0] in ('self', 'cls')
                    
                    # Get class name if it's a method
                    class_name = None
                    if is_method:
                        for parent in ast.walk(tree):
                            if isinstance(parent, ast.ClassDef):
                                for item in parent.body:
                                    if item == node or (hasattr(item, 'name') and item.name == func_name):
                                        class_name = parent.name
                                        break
                    
                    functions.append(FunctionInfo(
                        name=func_name,
                        start_line=start_line,
                        end_line=end_line,
                        code=func_code,
                        language='python',
                        is_method=is_method,
                        class_name=class_name,
                        parameters=params
                    ))
        except (SyntaxError, ValueError) as e:
            # If AST parsing fails, try regex-based extraction
            return self._extract_functions_regex(code, 'python')
        
        return functions
    
    def extract_java_functions(self, code: str) -> List[FunctionInfo]:
        """
        Extract methods from Java code using regex.
        
        Args:
            code: Java source code
        
        Returns:
            List of FunctionInfo objects
        """
        functions = []
        lines = code.split('\n')
        
        # Pattern to match Java methods
        # Matches: [modifiers] return_type method_name(params) { ... }
        pattern = r'(?:public|private|protected|static|\s)+[\w\s\[\]<>]+\s+(\w+)\s*\([^)]*\)\s*\{'
        
        for match in re.finditer(pattern, code, re.MULTILINE):
            method_name = match.group(1)
            start_pos = match.start()
            start_line = code[:start_pos].count('\n') + 1
            
            # Find matching closing brace
            brace_count = 0
            pos = match.end() - 1
            end_pos = pos
            
            while pos < len(code):
                if code[pos] == '{':
                    brace_count += 1
                elif code[pos] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = pos
                        break
                pos += 1
            
            end_line = code[:end_pos + 1].count('\n') + 1
            func_code = '\n'.join(lines[start_line - 1:end_line])
            
            # Extract parameters from method signature
            sig_match = re.search(r'\(([^)]*)\)', match.group(0))
            params = []
            if sig_match:
                param_str = sig_match.group(1)
                params = [p.strip().split()[-1] for p in param_str.split(',') if p.strip()]
            
            # Check if it's a constructor (same name as class)
            is_method = True  # All Java functions are methods
            class_name = None
            
            # Try to find class name
            class_match = re.search(r'class\s+(\w+)', code[:start_pos])
            if class_match:
                class_name = class_match.group(1)
            
            functions.append(FunctionInfo(
                name=method_name,
                start_line=start_line,
                end_line=end_line,
                code=func_code,
                language='java',
                is_method=is_method,
                class_name=class_name,
                parameters=params
            ))
        
        return functions
    
    def _extract_functions_regex(self, code: str, language: str) -> List[FunctionInfo]:
        """
        Fallback regex-based function extraction.
        
        Args:
            code: Source code
            language: Programming language
        
        Returns:
            List of FunctionInfo objects
        """
        functions = []
        lines = code.split('\n')
        
        if language == 'python':
            # Python function pattern
            pattern = r'^\s*def\s+(\w+)\s*\([^)]*\)\s*:'
        elif language in ['java', 'cpp', 'c']:
            # C-style function pattern
            pattern = r'[\w\s\[\]<>]+\s+(\w+)\s*\([^)]*\)\s*\{'
        else:
            # Generic function pattern
            pattern = r'function\s+(\w+)\s*\([^)]*\)\s*\{'
        
        for match in re.finditer(pattern, code, re.MULTILINE):
            func_name = match.group(1)
            start_line = code[:match.start()].count('\n') + 1
            
            # Simple heuristic: find next function or end of file
            next_match = None
            for next_m in re.finditer(pattern, code[match.end():], re.MULTILINE):
                next_match = next_m
                break
            
            if next_match:
                end_line = code[:match.end() + next_match.start()].count('\n') + 1
            else:
                end_line = len(lines)
            
            func_code = '\n'.join(lines[start_line - 1:end_line])
            
            functions.append(FunctionInfo(
                name=func_name,
                start_line=start_line,
                end_line=end_line,
                code=func_code,
                language=language,
                is_method=False,
                class_name=None,
                parameters=[]
            ))
        
        return functions
    
    def extract_functions(self, code: str, language: Optional[str] = None, filename: Optional[str] = None) -> List[FunctionInfo]:
        """
        Extract functions from code.
        
        Args:
            code: Source code
            language: Optional language (auto-detected if not provided)
            filename: Optional filename for language detection
        
        Returns:
            List of FunctionInfo objects
        """
        if not code or not code.strip():
            return []
        
        if language is None:
            language = self.detect_language(code, filename)
        
        if language == 'python':
            return self.extract_python_functions(code)
        elif language == 'java':
            return self.extract_java_functions(code)
        else:
            # Fallback to regex for other languages
            return self._extract_functions_regex(code, language)
    
    def extract_functions_from_diff(
        self,
        before_code: str,
        after_code: str,
        language: Optional[str] = None,
        filename: Optional[str] = None
    ) -> Tuple[List[FunctionInfo], List[FunctionInfo]]:
        """
        Extract functions from both before and after code.
        
        Args:
            before_code: Code before change
            after_code: Code after change
            language: Optional language
            filename: Optional filename
        
        Returns:
            Tuple of (before_functions, after_functions)
        """
        before_funcs = self.extract_functions(before_code, language, filename) if before_code else []
        after_funcs = self.extract_functions(after_code, language, filename) if after_code else []
        
        return before_funcs, after_funcs

