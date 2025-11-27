"""
Git Integration Service
Provides functionality to clone repositories, checkout commits, and extract file versions.
"""

import os
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

# Try to import GitPython, handle gracefully if not installed
try:
    import git
    from git import Repo, InvalidGitRepositoryError, GitCommandError
    HAS_GITPYTHON = True
except ImportError:
    HAS_GITPYTHON = False
    print("GitPython not installed: pip install GitPython")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GitService:
    """
    Service for Git repository operations.
    Handles cloning, checkout, and file extraction from Git repositories.
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize Git Service.
        
        Args:
            base_dir: Base directory for storing cloned repositories.
                     If None, uses a temporary directory.
        """
        if base_dir is None:
            # Use a directory in the project root for cloned repos
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_dir = os.path.join(project_root, "cloned_repos")
        
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        
        if not HAS_GITPYTHON:
            raise ImportError(
                "GitPython is required for Git operations. "
                "Install it with: pip install GitPython"
            )
    
    def clone_repository(
        self, 
        repo_url: str, 
        repo_name: Optional[str] = None,
        branch: Optional[str] = None,
        depth: Optional[int] = None
    ) -> str:
        """
        Clone a Git repository.
        
        Args:
            repo_url: URL of the repository (e.g., https://github.com/user/repo.git)
            repo_name: Optional name for the cloned directory. 
                      If None, extracts from URL.
            branch: Optional branch to checkout after cloning. If None, uses default.
            depth: Optional shallow clone depth (number of commits).
        
        Returns:
            Path to the cloned repository directory.
        
        Raises:
            GitCommandError: If cloning fails.
        """
        # Extract repo name from URL if not provided
        if repo_name is None:
            repo_name = self._extract_repo_name(repo_url)
        
        repo_path = os.path.join(self.base_dir, repo_name)
        
        # Remove existing clone if it exists
        if os.path.exists(repo_path):
            logger.info(f"Removing existing clone at {repo_path}")
            try:
                shutil.rmtree(repo_path)
            except PermissionError as e:
                logger.warning(f"Could not remove existing clone (permission denied): {e}")
                logger.warning(f"Attempting to use existing clone at {repo_path}")
                # Try to use existing clone instead
                if os.path.exists(os.path.join(repo_path, '.git')):
                    logger.info(f"Using existing clone at {repo_path}")
                    # Check if it's shallow and try to unshallow for better commit history
                    try:
                        repo = Repo(repo_path)
                        is_shallow = repo.git.rev_parse('--is-shallow-repository').strip()
                        if is_shallow == 'true':
                            logger.info("Existing repository is shallow, fetching more history...")
                            try:
                                repo.git.fetch('--unshallow')
                                logger.info("Successfully unshallowed existing repository")
                            except Exception as unshallow_error:
                                logger.warning(f"Could not unshallow existing repository: {unshallow_error}")
                    except Exception as e:
                        logger.debug(f"Could not check/unshallow existing repository: {e}")
                    return repo_path
                else:
                    raise Exception(f"Cannot use existing directory {repo_path} - it's not a valid git repository")
            except Exception as e:
                logger.warning(f"Error removing existing clone: {e}")
                # Try to continue anyway
                if os.path.exists(os.path.join(repo_path, '.git')):
                    logger.info(f"Using existing clone at {repo_path}")
                    # Check if it's shallow and try to unshallow for better commit history
                    try:
                        repo = Repo(repo_path)
                        is_shallow = repo.git.rev_parse('--is-shallow-repository').strip()
                        if is_shallow == 'true':
                            logger.info("Existing repository is shallow, fetching more history...")
                            try:
                                repo.git.fetch('--unshallow')
                                logger.info("Successfully unshallowed existing repository")
                            except Exception as unshallow_error:
                                logger.warning(f"Could not unshallow existing repository: {unshallow_error}")
                    except Exception as e:
                        logger.debug(f"Could not check/unshallow existing repository: {e}")
                    return repo_path
                raise
        
        logger.info(f"Cloning repository: {repo_url}")
        logger.info(f"Target directory: {repo_path}")
        
        try:
            # Clone with optional depth for shallow clones
            clone_kwargs = {}
            if depth:
                clone_kwargs['depth'] = depth
            
            repo = Repo.clone_from(repo_url, repo_path, **clone_kwargs)
            
            # Checkout specific branch if provided
            if branch:
                logger.info(f"Checking out branch: {branch}")
                repo.git.checkout(branch)
            
            logger.info(f"Successfully cloned repository to {repo_path}")
            
            # If this was a shallow clone, we might need to unshallow it
            # Check if it's shallow and unshallow if needed for commit history
            try:
                is_shallow = repo.git.rev_parse('--is-shallow-repository').strip()
                if is_shallow == 'true':
                    logger.info("Repository is shallow, fetching full history...")
                    try:
                        repo.git.fetch('--unshallow')
                        logger.info("Successfully unshallowed repository")
                    except Exception as unshallow_error:
                        logger.warning(f"Could not unshallow repository: {unshallow_error}")
                        logger.warning("Will use available commits from shallow clone")
            except Exception as e:
                logger.debug(f"Could not check if repository is shallow: {e}")
            
            return repo_path
            
        except GitCommandError as e:
            logger.error(f"Failed to clone repository: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during clone: {e}")
            raise
    
    def get_repo(self, repo_path: str) -> Repo:
        """
        Get a Git Repo object for an existing repository.
        
        Args:
            repo_path: Path to the repository.
        
        Returns:
            Git Repo object.
        
        Raises:
            InvalidGitRepositoryError: If path is not a valid Git repository.
        """
        try:
            return Repo(repo_path)
        except InvalidGitRepositoryError:
            raise ValueError(f"Path {repo_path} is not a valid Git repository")
    
    def checkout_commit(self, repo_path: str, commit_hash: str) -> None:
        """
        Checkout a specific commit in the repository.
        
        Args:
            repo_path: Path to the repository.
            commit_hash: Commit hash (full or short).
        
        Raises:
            GitCommandError: If checkout fails.
        """
        repo = self.get_repo(repo_path)
        
        try:
            logger.info(f"Checking out commit: {commit_hash}")
            repo.git.checkout(commit_hash)
            logger.info(f"Successfully checked out commit {commit_hash}")
        except GitCommandError as e:
            logger.error(f"Failed to checkout commit {commit_hash}: {e}")
            raise
    
    def get_file_at_commit(
        self, 
        repo_path: str, 
        file_path: str, 
        commit_hash: Optional[str] = None
    ) -> str:
        """
        Get the contents of a file at a specific commit.
        
        Args:
            repo_path: Path to the repository.
            file_path: Relative path to the file within the repository.
            commit_hash: Optional commit hash. If None, uses current HEAD.
        
        Returns:
            File contents as a string.
        
        Raises:
            FileNotFoundError: If file doesn't exist at that commit.
            GitCommandError: If Git operation fails.
        """
        repo = self.get_repo(repo_path)
        
        try:
            if commit_hash:
                # Get file content from specific commit
                file_content = repo.git.show(f"{commit_hash}:{file_path}")
            else:
                # Get file content from current HEAD
                file_path_abs = os.path.join(repo_path, file_path)
                if not os.path.exists(file_path_abs):
                    raise FileNotFoundError(f"File {file_path} not found in repository")
                
                with open(file_path_abs, 'r', encoding='utf-8', errors='ignore') as f:
                    file_content = f.read()
            
            return file_content
            
        except GitCommandError as e:
            logger.error(f"Failed to get file {file_path} at commit {commit_hash}: {e}")
            raise FileNotFoundError(f"File {file_path} not found at commit {commit_hash}")
    
    def get_file_before_after_commits(
        self,
        repo_path: str,
        file_path: str,
        before_commit: str,
        after_commit: str
    ) -> Tuple[str, str]:
        """
        Get file contents at two different commits (before and after).
        
        Args:
            repo_path: Path to the repository.
            file_path: Relative path to the file.
            before_commit: Commit hash for the "before" version.
            after_commit: Commit hash for the "after" version.
        
        Returns:
            Tuple of (before_content, after_content).
        """
        before_content = self.get_file_at_commit(repo_path, file_path, before_commit)
        after_content = self.get_file_at_commit(repo_path, file_path, after_commit)
        
        return before_content, after_content
    
    def get_commit_history(
        self,
        repo_path: str,
        file_path: Optional[str] = None,
        max_count: Optional[int] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get commit history for the repository or a specific file.
        
        Args:
            repo_path: Path to the repository.
            file_path: Optional file path to get history for specific file.
            max_count: Maximum number of commits to return.
            since: Optional start date for commit range.
            until: Optional end date for commit range.
        
        Returns:
            List of commit dictionaries with keys: hash, author, date, message.
        """
        repo = self.get_repo(repo_path)
        
        # Get commit log
        commits = []
        try:
            # Build git log command arguments
            # GitPython's git.log() accepts arguments as strings
            log_args = ['--pretty=format:%H|%an|%ad|%s', '--date=iso']
            
            # Add max count
            if max_count:
                log_args.extend(['-n', str(max_count)])
            
            # Add date filters
            if since:
                if isinstance(since, datetime):
                    since_str = since.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    since_str = str(since)
                log_args.extend(['--since', since_str])
            
            if until:
                if isinstance(until, datetime):
                    until_str = until.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    until_str = str(until)
                log_args.extend(['--until', until_str])
            
            # Add file path if specified (must come after --)
            if file_path:
                log_args.append('--')
                log_args.append(file_path)
            
            logger.info(f"Running git log with args: {log_args}")
            logger.info(f"Requesting {max_count or 'all'} commits")
            
            # Call git log
            log_output = repo.git.log(*log_args)
            
            # Handle empty output
            if not log_output or not log_output.strip():
                logger.warning("Git log returned empty output")
                return commits
            
            # Parse output line by line
            lines = log_output.strip().split('\n') if log_output else []
            logger.info(f"Git log returned {len(lines)} lines of output")
            
            for line in lines:
                if not line.strip():
                    continue
                
                parts = line.split('|', 3)
                if len(parts) == 4:
                    commit_hash, author, date_str, message = parts
                    try:
                        # Parse date - handle different formats
                        try:
                            date_obj = datetime.fromisoformat(date_str.replace(' ', 'T'))
                        except ValueError:
                            # Try alternative format
                            from dateutil import parser
                            date_obj = parser.parse(date_str)
                        
                        commits.append({
                            'hash': commit_hash.strip(),
                            'author': author.strip(),
                            'date': date_obj,
                            'message': message.strip()
                        })
                    except Exception as e:
                        logger.warning(f"Error parsing commit line: {line[:50]}... Error: {e}")
                        continue
            
            logger.info(f"Successfully parsed {len(commits)} commits from git log")
            
            # If we got fewer commits than requested and repository is shallow, try to fetch more
            if max_count and len(commits) < max_count:
                try:
                    is_shallow = repo.git.rev_parse('--is-shallow-repository').strip()
                    if is_shallow == 'true':
                        logger.info(f"Repository is shallow and only has {len(commits)} commits, but {max_count} requested. Attempting to fetch more...")
                        try:
                            repo.git.fetch('--unshallow')
                            logger.info("Successfully unshallowed, retrying git log...")
                            # Retry getting commits
                            log_output_retry = repo.git.log(*log_args)
                            if log_output_retry and log_output_retry.strip():
                                commits = []  # Reset commits list
                                lines_retry = log_output_retry.strip().split('\n') if log_output_retry else []
                                for line in lines_retry:
                                    if not line.strip():
                                        continue
                                    parts = line.split('|', 3)
                                    if len(parts) == 4:
                                        commit_hash, author, date_str, message = parts
                                        try:
                                            try:
                                                date_obj = datetime.fromisoformat(date_str.replace(' ', 'T'))
                                            except ValueError:
                                                from dateutil import parser
                                                date_obj = parser.parse(date_str)
                                            commits.append({
                                                'hash': commit_hash.strip(),
                                                'author': author.strip(),
                                                'date': date_obj,
                                                'message': message.strip()
                                            })
                                        except Exception as e:
                                            logger.warning(f"Error parsing commit line: {line[:50]}... Error: {e}")
                                            continue
                                logger.info(f"After unshallow, successfully parsed {len(commits)} commits from git log")
                        except Exception as fetch_error:
                            logger.warning(f"Could not fetch more commits: {fetch_error}")
                except Exception as check_error:
                    logger.debug(f"Could not check if repository is shallow: {check_error}")
            
        except GitCommandError as e:
            logger.error(f"Git command error getting commit history: {e}")
            logger.error(f"Command was: git log {' '.join(log_args)}")
        except Exception as e:
            logger.error(f"Unexpected error getting commit history: {e}", exc_info=True)
        
        return commits
    
    def get_commit_info(self, repo_path: str, commit_hash: str) -> Dict:
        """
        Get detailed information about a specific commit.
        
        Args:
            repo_path: Path to the repository.
            commit_hash: Commit hash.
        
        Returns:
            Dictionary with commit information.
        """
        repo = self.get_repo(repo_path)
        
        try:
            commit = repo.commit(commit_hash)
            
            # Get changed files
            changed_files = []
            if commit.parents:
                diff = commit.diff(commit.parents[0])
                for item in diff:
                    changed_files.append({
                        'path': item.a_path if item.a_path else item.b_path,
                        'change_type': item.change_type,
                        'added_lines': item.diff.count('\n+') if hasattr(item, 'diff') else 0,
                        'deleted_lines': item.diff.count('\n-') if hasattr(item, 'diff') else 0
                    })
            
            return {
                'hash': commit.hexsha,
                'short_hash': commit.hexsha[:7],
                'author': commit.author.name,
                'author_email': commit.author.email,
                'date': commit.committed_datetime,
                'message': commit.message.strip(),
                'changed_files': changed_files,
                'stats': {
                    'total': commit.stats.total,
                    'insertions': commit.stats.total.get('insertions', 0),
                    'deletions': commit.stats.total.get('deletions', 0)
                }
            }
        except Exception as e:
            logger.error(f"Error getting commit info for {commit_hash}: {e}")
            raise
    
    def get_changed_files_in_commit(
        self,
        repo_path: str,
        commit_hash: str
    ) -> List[Dict]:
        """
        Get list of files changed in a specific commit.
        
        Args:
            repo_path: Path to the repository.
            commit_hash: Commit hash.
        
        Returns:
            List of dictionaries with file change information.
        """
        commit_info = self.get_commit_info(repo_path, commit_hash)
        return commit_info.get('changed_files', [])
    
    def cleanup_repo(self, repo_path: str, ignore_errors: bool = True) -> None:
        """
        Remove a cloned repository from disk.
        
        Args:
            repo_path: Path to the repository to remove.
            ignore_errors: If True, log warnings instead of raising exceptions (useful for Windows file locking).
        """
        if os.path.exists(repo_path):
            logger.info(f"Cleaning up repository: {repo_path}")
            try:
                shutil.rmtree(repo_path)
            except PermissionError as e:
                if ignore_errors:
                    logger.warning(f"Could not remove repository (permission denied - file may be locked): {e}")
                    logger.warning("Repository will remain on disk. This is OK and won't affect functionality.")
                else:
                    raise
            except Exception as e:
                if ignore_errors:
                    logger.warning(f"Could not remove repository: {e}")
                else:
                    raise
        else:
            logger.warning(f"Repository path does not exist: {repo_path}")
    
    def _extract_repo_name(self, repo_url: str) -> str:
        """
        Extract repository name from URL.
        
        Args:
            repo_url: Repository URL.
        
        Returns:
            Repository name.
        """
        # Remove .git suffix if present
        repo_url = repo_url.rstrip('/').rstrip('.git')
        
        # Extract name from URL
        if '/' in repo_url:
            repo_name = repo_url.split('/')[-1]
        else:
            repo_name = repo_url
        
        # Sanitize name (remove invalid characters)
        repo_name = "".join(c for c in repo_name if c.isalnum() or c in ('-', '_'))
        
        return repo_name
    
    def is_valid_repo_url(self, repo_url: str) -> bool:
        """
        Check if a URL appears to be a valid Git repository URL.
        
        Args:
            repo_url: URL to validate.
        
        Returns:
            True if URL appears valid.
        """
        valid_prefixes = ['http://', 'https://', 'git@', 'git://']
        return any(repo_url.startswith(prefix) for prefix in valid_prefixes) or \
               repo_url.endswith('.git')


# Convenience function for quick operations
def clone_and_get_file(
    repo_url: str,
    file_path: str,
    commit_hash: Optional[str] = None,
    branch: Optional[str] = None
) -> str:
    """
    Convenience function to clone a repo and get a file in one step.
    
    Args:
        repo_url: Repository URL.
        file_path: Path to file within repository.
        commit_hash: Optional commit hash.
        branch: Optional branch to checkout.
    
    Returns:
        File contents.
    """
    service = GitService()
    repo_path = service.clone_repository(repo_url, branch=branch)
    
    try:
        if commit_hash:
            service.checkout_commit(repo_path, commit_hash)
        
        file_content = service.get_file_at_commit(repo_path, file_path, commit_hash)
        return file_content
    finally:
        # Cleanup
        service.cleanup_repo(repo_path)

