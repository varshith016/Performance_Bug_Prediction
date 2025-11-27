"""
SZZ (Sliwerski-Zimmermann-Zeller) Algorithm Service
Identifies bug-inducing commits by analyzing bug-fix commits and tracing back to their origins.
"""

import re
import logging
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime
from collections import defaultdict

# Import Git service
try:
    from git_service import GitService
except ImportError:
    # Handle relative import
    from .git_service import GitService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SZZService:
    """
    Service for identifying bug-inducing commits using the SZZ algorithm.
    """
    
    # Keywords that indicate bug-fix commits (general)
    BUG_FIX_KEYWORDS = [
        'fix', 'fixes', 'fixed', 'fixing',
        'bug', 'bugs',
        'error', 'errors',
        'issue', 'issues',
        'defect', 'defects',
        'patch', 'patches',
        'resolve', 'resolves', 'resolved',
        'correct', 'corrects', 'corrected',
        'repair', 'repairs', 'repaired',
        'solve', 'solves', 'solved',
        'close', 'closes', 'closed',
        'address', 'addresses', 'addressed'
    ]
    
    # Performance-specific keywords that indicate performance bug fixes
    PERFORMANCE_KEYWORDS = [
        'performance', 'performances',
        'slow', 'slower', 'slowly', 'slowness',
        'latency', 'latencies',
        'bottleneck', 'bottlenecks',
        'optimize', 'optimizes', 'optimized', 'optimizing', 'optimization', 'optimisations',
        'throughput', 'throughputs',
        'speed', 'speeds', 'speeding', 'speedup', 'speed-up',
        'efficiency', 'efficient', 'inefficient', 'inefficiency',
        'memory leak', 'memory leaks', 'leak', 'leaks', 'leaking',
        'cpu usage', 'cpu utilization', 'cpu intensive',
        'deadlock', 'deadlocks',
        'race condition', 'race conditions',
        'timeout', 'timeouts', 'timing out',
        'hang', 'hangs', 'hanging', 'hung',
        'stall', 'stalls', 'stalling', 'stalled',
        'lag', 'lags', 'lagging', 'laggy',
        'response time', 'response times',
        'scalability', 'scalable', 'scaling',
        'concurrency', 'concurrent',
        'thread', 'threads', 'threading', 'threaded',
        'async', 'asynchronous', 'asynchronously',
        'blocking', 'blocked', 'blocks',
        'wait', 'waits', 'waiting',
        'cache', 'caching', 'cached',
        'garbage collection', 'gc', 'garbage collector',
    ]
    
    # Issue reference patterns (e.g., #123, JIRA-456, GH-789)
    ISSUE_PATTERNS = [
        r'#\d+',  # GitHub style: #123
        r'[A-Z]+-\d+',  # JIRA style: JIRA-456
        r'GH-\d+',  # GitHub issue: GH-789
        r'issue\s*#?\s*\d+',  # issue #123
        r'bug\s*#?\s*\d+',  # bug #123
    ]
    
    def __init__(self, git_service: Optional[GitService] = None, jira_service: Optional[object] = None):
        """
        Initialize SZZ Service.
        
        Args:
            git_service: Optional GitService instance. If None, creates a new one.
            jira_service: Optional JiraService instance for issue tracker integration.
        """
        self.git_service = git_service or GitService()
        self.jira_service = jira_service
    
    def is_bug_fix_commit(self, commit_message: str) -> bool:
        """
        Determine if a commit is a bug-fix commit based on commit message.
        Now includes performance-specific keyword detection and Jira integration.
        
        Args:
            commit_message: Commit message text.
        
        Returns:
            True if commit appears to be a bug fix (including performance bugs).
        """
        if not commit_message:
            return False
        
        message_lower = commit_message.lower()
        
        # Check for general bug-fix keywords
        for keyword in self.BUG_FIX_KEYWORDS:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, message_lower, re.IGNORECASE):
                return True
        
        # Check for performance-specific keywords
        # These are often indicators of performance bug fixes
        for keyword in self.PERFORMANCE_KEYWORDS:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, message_lower, re.IGNORECASE):
                return True
        
        # Check for issue references
        for pattern in self.ISSUE_PATTERNS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                return True
        
        # Check for performance-related issue patterns
        performance_issue_patterns = [
            r'perf\s*#?\s*\d+',  # perf #123
            r'performance\s*#?\s*\d+',  # performance #123
            r'slow\s*#?\s*\d+',  # slow #123
            r'latency\s*#?\s*\d+',  # latency #123
        ]
        for pattern in performance_issue_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                return True
        
        # Enhanced detection using Jira (if available)
        if self.jira_service:
            try:
                # Extract Jira issue keys from commit message
                issue_keys = self.jira_service.extract_issue_keys(commit_message)
                for issue_key in issue_keys:
                    issue_info = self.jira_service.get_issue(issue_key)
                    if issue_info and self.jira_service.enhance_bug_fix_detection(commit_message, issue_info):
                        logger.info(f"Jira issue {issue_key} indicates bug fix")
                        return True
            except Exception as e:
                logger.warning(f"Error checking Jira for commit: {e}")
        
        return False
    
    def identify_bug_fix_commits(
        self,
        repo_path: str,
        max_commits: Optional[int] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        progress_callback: Optional[callable] = None
    ) -> List[Dict]:
        """
        Identify bug-fix commits in a repository.
        
        Args:
            repo_path: Path to the repository.
            max_commits: Maximum number of commits to analyze.
            since: Start date for commit range.
            until: End date for commit range.
            progress_callback: Optional callback function(progress, message) for progress updates.
        
        Returns:
            List of bug-fix commit dictionaries.
        """
        logger.info(f"Identifying bug-fix commits in {repo_path}")
        
        if progress_callback:
            progress_callback(10.0, "Fetching commit history...")
        
        # Get commit history
        commits = self.git_service.get_commit_history(
            repo_path,
            max_count=max_commits,
            since=since,
            until=until
        )
        
        logger.info(f"Analyzing {len(commits)} commits for bug-fix patterns...")
        
        if progress_callback:
            progress_callback(11.0, f"Analyzing {len(commits)} commits...")
        
        bug_fix_commits = []
        total_commits = len(commits)
        last_progress_update = 0
        jira_issues_found = 0
        jira_issues_used = 0
        
        for idx, commit in enumerate(commits):
            # Update progress every 5% of commits or every 50 commits, whichever is more frequent
            if progress_callback and total_commits > 0:
                progress_pct = (idx / total_commits) * 100
                if progress_pct - last_progress_update >= 5 or idx % 50 == 0:
                    # Map to 10-15% range (within SZZ's overall 10-25% range)
                    progress = 10.0 + (idx / total_commits) * 5.0  # 10-15%
                    jira_status = ""
                    if self.jira_service:
                        jira_status = f" (Jira: {jira_issues_found} issues found, {jira_issues_used} used)"
                    progress_callback(progress, f"Checking commit {idx+1}/{total_commits}{jira_status}...")
                    last_progress_update = progress_pct
            
            # Track Jira usage
            if self.jira_service:
                try:
                    issue_keys = self.jira_service.extract_issue_keys(commit['message'])
                    if issue_keys:
                        jira_issues_found += len(issue_keys)
                        logger.debug(f"Found Jira issue keys in commit {commit['hash'][:7]}: {issue_keys}")
                except:
                    pass
            
            if self.is_bug_fix_commit(commit['message']):
                commit_info = self.git_service.get_commit_info(
                    repo_path,
                    commit['hash']
                )
                commit_info['is_bug_fix'] = True
                
                # Check if Jira helped identify this
                if self.jira_service:
                    try:
                        issue_keys = self.jira_service.extract_issue_keys(commit['message'])
                        if issue_keys:
                            for issue_key in issue_keys:
                                issue_info = self.jira_service.get_issue(issue_key)
                                if issue_info and self.jira_service.enhance_bug_fix_detection(commit['message'], issue_info):
                                    commit_info['jira_issue'] = issue_key
                                    commit_info['jira_enhanced'] = True
                                    jira_issues_used += 1
                                    break
                    except:
                        pass
                
                bug_fix_commits.append(commit_info)
        
        logger.info(f"Found {len(bug_fix_commits)} bug-fix commits out of {len(commits)} total commits")
        if self.jira_service:
            logger.info(f"Jira integration: Found {jira_issues_found} issue keys, used {jira_issues_used} for bug detection")
        
        if progress_callback:
            progress_callback(15.0, f"Found {len(bug_fix_commits)} bug-fix commits")
        
        return bug_fix_commits
    
    def find_bug_inducing_commits(
        self,
        repo_path: str,
        bug_fix_commit_hash: str
    ) -> List[Dict]:
        """
        Find bug-inducing commits for a given bug-fix commit.
        
        This uses a simplified SZZ approach:
        1. Get files changed in bug-fix commit
        2. For each file, find when the buggy lines were introduced
        3. Identify the commits that introduced those lines
        
        Args:
            repo_path: Path to the repository.
            bug_fix_commit_hash: Hash of the bug-fix commit.
        
        Returns:
            List of bug-inducing commit dictionaries.
        """
        logger.info(f"Finding bug-inducing commits for bug-fix: {bug_fix_commit_hash[:7]}")
        
        try:
            repo = self.git_service.get_repo(repo_path)
            bug_fix_commit = repo.commit(bug_fix_commit_hash)
            
            if not bug_fix_commit.parents:
                logger.warning(f"Bug-fix commit {bug_fix_commit_hash[:7]} has no parents")
                return []
            
            parent_commit = bug_fix_commit.parents[0]
            
            # Get diff between parent and bug-fix commit
            diff = parent_commit.diff(bug_fix_commit)
            
            bug_inducing_commits = []
            files_analyzed = set()
            
            for item in diff:
                file_path = item.a_path if item.a_path else item.b_path
                
                if file_path in files_analyzed:
                    continue
                files_analyzed.add(file_path)
                
                # Get the blame for this file at the parent commit
                # This tells us when each line was last modified
                try:
                    # Use git blame to find when lines were introduced
                    blame_output = repo.git.blame(
                        '-w',  # Ignore whitespace
                        '-M',  # Detect moved lines
                        parent_commit.hexsha,
                        '--',
                        file_path
                    )
                    
                    # Parse blame output to find commits
                    blame_commits = self._parse_blame_output(blame_output)
                    
                    # Get diff to see which lines were changed in bug-fix
                    file_diff = item.diff
                    if isinstance(file_diff, bytes):
                        file_diff = file_diff.decode('utf-8', errors='ignore')
                    elif not isinstance(file_diff, str):
                        file_diff = str(file_diff)
                    changed_lines = self._extract_changed_lines(file_diff)
                    
                    # Find commits that introduced the changed lines
                    for line_num in changed_lines:
                        if line_num in blame_commits:
                            commit_hash = blame_commits[line_num]
                            if commit_hash and commit_hash != parent_commit.hexsha:
                                # This commit introduced the buggy line
                                if commit_hash not in [c['hash'] for c in bug_inducing_commits]:
                                    commit_info = self.git_service.get_commit_info(
                                        repo_path,
                                        commit_hash
                                    )
                                    commit_info['bug_fix_commit'] = bug_fix_commit_hash
                                    commit_info['file_path'] = file_path
                                    commit_info['line_number'] = line_num
                                    bug_inducing_commits.append(commit_info)
                
                except Exception as e:
                    logger.warning(f"Error analyzing file {file_path}: {e}")
                    continue
            
            logger.info(f"Found {len(bug_inducing_commits)} potential bug-inducing commits")
            return bug_inducing_commits
            
        except Exception as e:
            logger.error(f"Error finding bug-inducing commits: {e}")
            return []
    
    def _parse_blame_output(self, blame_output: str) -> Dict[int, str]:
        """
        Parse git blame output to extract commit hashes for each line.
        
        Args:
            blame_output: Raw git blame output.
        
        Returns:
            Dictionary mapping line numbers to commit hashes.
        """
        line_commits = {}
        lines = blame_output.split('\n')
        
        for line in lines:
            if not line.strip():
                continue
            
            # Git blame format: commit_hash (author date line_num) content
            match = re.match(r'^([a-f0-9]{40})\s+', line)
            if match:
                commit_hash = match.group(1)
                # Extract line number (approximate - blame format varies)
                # For simplicity, we'll use line position
                line_num = len(line_commits) + 1
                line_commits[line_num] = commit_hash
        
        return line_commits
    
    def _extract_changed_lines(self, diff_output: str) -> Set[int]:
        """
        Extract line numbers that were changed in a diff.
        
        Args:
            diff_output: Unified diff output.
        
        Returns:
            Set of line numbers that were changed.
        """
        changed_lines = set()
        lines = diff_output.split('\n')
        current_line = 0
        
        for line in lines:
            if line.startswith('@@'):
                # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
                match = re.search(r'\+(\d+)', line)
                if match:
                    current_line = int(match.group(1))
            elif line.startswith('+') or line.startswith('-'):
                if line.startswith('+'):
                    changed_lines.add(current_line)
                current_line += 1
        
        return changed_lines
    
    def label_commits(
        self,
        repo_path: str,
        bug_inducing_commits: List[Dict],
        all_commits: Optional[List[Dict]] = None
    ) -> Dict[str, str]:
        """
        Label commits as 'BUGGY' or 'CLEAN' based on SZZ analysis.
        
        Args:
            repo_path: Path to the repository.
            bug_inducing_commits: List of bug-inducing commits from SZZ.
            all_commits: Optional list of all commits to label. If None, only labels bug-inducing commits.
        
        Returns:
            Dictionary mapping commit hashes to labels ('BUGGY' or 'CLEAN').
        """
        labels = {}
        
        # Mark bug-inducing commits as BUGGY
        bug_inducing_hashes = {c['hash'] for c in bug_inducing_commits}
        for commit_hash in bug_inducing_hashes:
            labels[commit_hash] = 'BUGGY'
        
        # If all_commits provided, mark others as CLEAN
        if all_commits:
            for commit in all_commits:
                commit_hash = commit.get('hash') or commit.get('hexsha', '')
                if commit_hash and commit_hash not in labels:
                    labels[commit_hash] = 'CLEAN'
        
        logger.info(f"Labeled {len([l for l in labels.values() if l == 'BUGGY'])} commits as BUGGY")
        logger.info(f"Labeled {len([l for l in labels.values() if l == 'CLEAN'])} commits as CLEAN")
        
        return labels
    
    def analyze_repository(
        self,
        repo_path: str,
        max_commits: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """
        Complete SZZ analysis of a repository.
        
        Args:
            repo_path: Path to the repository.
            max_commits: Maximum number of commits to analyze.
            progress_callback: Optional callback function(progress, message) for progress updates.
        
        Returns:
            Dictionary with analysis results:
            - bug_fix_commits: List of bug-fix commits
            - bug_inducing_commits: List of bug-inducing commits
            - labels: Dictionary mapping commit hashes to labels
            - statistics: Analysis statistics
        """
        logger.info(f"Starting SZZ analysis for repository: {repo_path}")
        
        # Progress range: 10-25% (SZZ analysis takes 10-25% of total training progress)
        if progress_callback:
            progress_callback(10.0, "Identifying bug-fix commits...")
        
        # Step 1: Identify bug-fix commits (10-15%)
        # The identify_bug_fix_commits function will handle its own progress (10-15% range)
        bug_fix_commits = self.identify_bug_fix_commits(
            repo_path,
            max_commits=max_commits,
            progress_callback=progress_callback
        )
        
        if progress_callback:
            progress_callback(15.0, f"Found {len(bug_fix_commits)} bug-fix commits. Finding bug-inducing commits...")
        
        # Step 2: Find bug-inducing commits for each bug-fix (15-22%)
        all_bug_inducing = []
        total_bug_fixes = len(bug_fix_commits)
        for idx, bug_fix in enumerate(bug_fix_commits):
            if progress_callback and total_bug_fixes > 0:
                progress = 15.0 + (idx / total_bug_fixes) * 7.0  # 15-22%
                progress_callback(progress, f"Analyzing bug-fix {idx+1}/{total_bug_fixes}...")
            
            bug_inducing = self.find_bug_inducing_commits(
                repo_path,
                bug_fix['hash']
            )
            all_bug_inducing.extend(bug_inducing)
        
        # Remove duplicates
        seen_hashes = set()
        unique_bug_inducing = []
        for commit in all_bug_inducing:
            if commit['hash'] not in seen_hashes:
                seen_hashes.add(commit['hash'])
                unique_bug_inducing.append(commit)
        
        if progress_callback:
            progress_callback(22.0, "Labeling commits...")
        
        # Step 3: Label commits (22-25%)
        all_commits = self.git_service.get_commit_history(
            repo_path,
            max_count=max_commits
        )
        labels = self.label_commits(
            repo_path,
            unique_bug_inducing,
            all_commits
        )
        
        if progress_callback:
            progress_callback(25.0, "SZZ analysis complete!")
        
        # Calculate statistics
        total_commits = len(all_commits)
        bug_fix_count = len(bug_fix_commits)
        bug_inducing_count = len(unique_bug_inducing)
        buggy_count = len([l for l in labels.values() if l == 'BUGGY'])
        clean_count = len([l for l in labels.values() if l == 'CLEAN'])
        
        # Count Jira-enhanced commits
        jira_enhanced_count = len([c for c in bug_fix_commits if c.get('jira_enhanced', False)])
        
        statistics = {
            'total_commits': total_commits,
            'bug_fix_commits': bug_fix_count,
            'bug_inducing_commits': bug_inducing_count,
            'buggy_labeled': buggy_count,
            'clean_labeled': clean_count,
            'bug_fix_ratio': bug_fix_count / total_commits if total_commits > 0 else 0,
            'bug_inducing_ratio': bug_inducing_count / total_commits if total_commits > 0 else 0,
            'jira_enabled': self.jira_service is not None,
            'jira_enhanced_commits': jira_enhanced_count
        }
        
        return {
            'bug_fix_commits': bug_fix_commits,
            'bug_inducing_commits': unique_bug_inducing,
            'labels': labels,
            'statistics': statistics
        }
    
    def create_labeled_dataset(
        self,
        repo_path: str,
        analysis_results: Dict,
        output_path: Optional[str] = None,
        function_level: bool = False
    ) -> str:
        """
        Create a labeled dataset from SZZ analysis results.
        Supports both file-level and function-level labeling.
        
        Args:
            repo_path: Path to the repository.
            analysis_results: Results from analyze_repository().
            output_path: Optional path to save CSV. If None, returns as string.
            function_level: If True, extract features at function level. Default: False (file level).
        
        Returns:
            Path to saved CSV file or CSV content as string.
        """
        import pandas as pd
        try:
            from feature_extractor import build_feature_vector, build_function_level_features
        except ImportError:
            from .feature_extractor import build_feature_vector, build_function_level_features
        
        logger.info(f"Creating labeled dataset from SZZ analysis (function_level={function_level})...")
        
        rows = []
        labels = analysis_results['labels']
        
        # Get all commits
        repo = self.git_service.get_repo(repo_path)
        all_commits = self.git_service.get_commit_history(repo_path)
        
        for commit_info in all_commits:
            commit_hash = commit_info['hash']
            label = labels.get(commit_hash, 'CLEAN')
            
            # Get commit details
            try:
                commit = repo.commit(commit_hash)
                commit_message = commit.message.strip()
                
                # Get changed files
                changed_files = []
                if commit.parents:
                    diff = commit.parents[0].diff(commit)
                    for item in diff:
                        file_path = item.a_path if item.a_path else item.b_path
                        changed_files.append(file_path)
                
                # For each changed file, extract features
                for file_path in changed_files[:5]:  # Limit to first 5 files per commit
                    try:
                        # Get file before and after
                        if commit.parents:
                            parent_hash = commit.parents[0].hexsha
                            before_content = self.git_service.get_file_at_commit(
                                repo_path, file_path, parent_hash
                            )
                        else:
                            before_content = ""
                        
                        after_content = self.git_service.get_file_at_commit(
                            repo_path, file_path, commit_hash
                        )
                        
                        # Detect language from file extension
                        language = None
                        if file_path:
                            ext = file_path.split('.')[-1].lower()
                            lang_map = {
                                'py': 'python',
                                'java': 'java',
                                'js': 'javascript',
                                'cpp': 'cpp',
                                'c': 'c',
                            }
                            language = lang_map.get(ext, 'python')
                        
                        if function_level:
                            # Extract features at function level
                            function_features = build_function_level_features(
                                before_content,
                                after_content,
                                commit_message,
                                language=language,
                                filename=file_path
                            )
                            
                            # Create one row per function
                            for features_df, stats, func_info in function_features:
                                row = {
                                    'file_path': file_path,
                                    'function_name': func_info.name,
                                    'function_start_line': func_info.start_line,
                                    'function_end_line': func_info.end_line,
                                    'is_method': func_info.is_method,
                                    'class_name': func_info.class_name or '',
                                    'commit_hash': commit_hash,
                                    'commit_message': commit_message,
                                    'author': commit_info.get('author', ''),
                                    'date': commit_info.get('date', ''),
                                    'label': label,
                                    **stats
                                }
                                rows.append(row)
                        else:
                            # Extract features at file level (original behavior)
                            features_df, stats = build_feature_vector(
                                before_content,
                                after_content,
                                commit_message
                            )
                            
                            # Create row
                            row = {
                                'file_path': file_path,
                                'function_name': '',  # Empty for file-level
                                'function_start_line': 0,
                                'function_end_line': 0,
                                'is_method': False,
                                'class_name': '',
                                'commit_hash': commit_hash,
                                'commit_message': commit_message,
                                'author': commit_info.get('author', ''),
                                'date': commit_info.get('date', ''),
                                'label': label,
                                **stats
                            }
                            rows.append(row)
                    
                    except Exception as e:
                        logger.warning(f"Error processing file {file_path} in commit {commit_hash[:7]}: {e}")
                        continue
            
            except Exception as e:
                logger.warning(f"Error processing commit {commit_hash[:7]}: {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved labeled dataset to {output_path} ({len(rows)} rows, function_level={function_level})")
            return output_path
        else:
            return df.to_csv(index=False)

