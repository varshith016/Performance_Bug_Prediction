"""
Process Metrics Service
Calculates process metrics from Git history (churn, developer count, commit frequency, etc.)
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter

# Import Git service
try:
    from git_service import GitService
except ImportError:
    from .git_service import GitService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessMetricsService:
    """
    Service for calculating process metrics from Git history.
    """
    
    def __init__(self, git_service: Optional[GitService] = None):
        """
        Initialize Process Metrics Service.
        
        Args:
            git_service: Optional GitService instance. If None, creates a new one.
        """
        self.git_service = git_service or GitService()
    
    def calculate_churn(
        self,
        repo_path: str,
        file_path: str,
        time_window_days: int = 30,
        commit_hash: Optional[str] = None
    ) -> Dict:
        """
        Calculate code churn for a file within a time window.
        
        Churn = sum of all additions and deletions in commits affecting the file.
        
        Args:
            repo_path: Path to the repository.
            file_path: Path to the file within repository.
            time_window_days: Number of days to look back.
            commit_hash: Optional commit hash to calculate churn up to.
        
        Returns:
            Dictionary with churn metrics.
        """
        try:
            repo = self.git_service.get_repo(repo_path)
            
            # Get commit history for the file
            commits = self.git_service.get_commit_history(
                repo_path,
                file_path=file_path,
                max_count=1000
            )
            
            if not commits:
                return {
                    'churn': 0,
                    'additions': 0,
                    'deletions': 0,
                    'net_churn': 0,
                    'commit_count': 0
                }
            
            # Filter commits within time window
            if commit_hash:
                target_commit = repo.commit(commit_hash)
                cutoff_date = target_commit.committed_datetime - timedelta(days=time_window_days)
            else:
                cutoff_date = datetime.now() - timedelta(days=time_window_days)
            
            total_additions = 0
            total_deletions = 0
            commit_count = 0
            
            for commit_info in commits:
                commit_date = commit_info.get('date')
                if isinstance(commit_date, str):
                    try:
                        commit_date = datetime.fromisoformat(commit_date.replace(' ', 'T'))
                    except:
                        continue
                
                if commit_date and commit_date >= cutoff_date:
                    # Get commit details
                    try:
                        commit = repo.commit(commit_info['hash'])
                        if commit.parents:
                            diff = commit.parents[0].diff(commit)
                            for item in diff:
                                if (item.a_path == file_path or item.b_path == file_path):
                                    stats = item.diff_stats
                                    total_additions += stats.get('insertions', 0)
                                    total_deletions += stats.get('deletions', 0)
                                    commit_count += 1
                                    break
                    except Exception as e:
                        logger.warning(f"Error processing commit {commit_info['hash'][:7]}: {e}")
                        continue
            
            churn = total_additions + total_deletions
            net_churn = total_additions - total_deletions
            
            return {
                'churn': churn,
                'additions': total_additions,
                'deletions': total_deletions,
                'net_churn': net_churn,
                'commit_count': commit_count,
                'time_window_days': time_window_days
            }
            
        except Exception as e:
            logger.error(f"Error calculating churn for {file_path}: {e}")
            return {
                'churn': 0,
                'additions': 0,
                'deletions': 0,
                'net_churn': 0,
                'commit_count': 0
            }
    
    def count_developers(
        self,
        repo_path: str,
        file_path: str,
        time_window_days: int = 90
    ) -> Dict:
        """
        Count number of unique developers who modified a file.
        
        Args:
            repo_path: Path to the repository.
            file_path: Path to the file.
            time_window_days: Number of days to look back.
        
        Returns:
            Dictionary with developer metrics.
        """
        try:
            commits = self.git_service.get_commit_history(
                repo_path,
                file_path=file_path,
                max_count=1000
            )
            
            if not commits:
                return {
                    'developer_count': 0,
                    'unique_developers': [],
                    'most_active_developer': None
                }
            
            cutoff_date = datetime.now() - timedelta(days=time_window_days)
            developers = []
            
            for commit_info in commits:
                commit_date = commit_info.get('date')
                if isinstance(commit_date, str):
                    try:
                        commit_date = datetime.fromisoformat(commit_date.replace(' ', 'T'))
                    except:
                        continue
                
                if commit_date and commit_date >= cutoff_date:
                    author = commit_info.get('author', '')
                    if author:
                        developers.append(author)
            
            developer_counter = Counter(developers)
            unique_developers = list(developer_counter.keys())
            most_active = developer_counter.most_common(1)[0][0] if developer_counter else None
            
            return {
                'developer_count': len(unique_developers),
                'unique_developers': unique_developers,
                'most_active_developer': most_active,
                'developer_commits': dict(developer_counter)
            }
            
        except Exception as e:
            logger.error(f"Error counting developers for {file_path}: {e}")
            return {
                'developer_count': 0,
                'unique_developers': [],
                'most_active_developer': None
            }
    
    def calculate_commit_frequency(
        self,
        repo_path: str,
        file_path: str,
        time_window_days: int = 30
    ) -> Dict:
        """
        Calculate commit frequency for a file.
        
        Args:
            repo_path: Path to the repository.
            file_path: Path to the file.
            time_window_days: Number of days to analyze.
        
        Returns:
            Dictionary with frequency metrics.
        """
        try:
            commits = self.git_service.get_commit_history(
                repo_path,
                file_path=file_path,
                max_count=1000
            )
            
            if not commits:
                return {
                    'commit_frequency': 0.0,
                    'commits_per_week': 0.0,
                    'total_commits': 0,
                    'time_window_days': time_window_days
                }
            
            cutoff_date = datetime.now() - timedelta(days=time_window_days)
            recent_commits = []
            
            for commit_info in commits:
                commit_date = commit_info.get('date')
                if isinstance(commit_date, str):
                    try:
                        commit_date = datetime.fromisoformat(commit_date.replace(' ', 'T'))
                    except:
                        continue
                
                if commit_date and commit_date >= cutoff_date:
                    recent_commits.append(commit_date)
            
            total_commits = len(recent_commits)
            weeks = time_window_days / 7.0
            commits_per_week = total_commits / weeks if weeks > 0 else 0.0
            commit_frequency = total_commits / time_window_days if time_window_days > 0 else 0.0
            
            return {
                'commit_frequency': commit_frequency,
                'commits_per_week': commits_per_week,
                'total_commits': total_commits,
                'time_window_days': time_window_days
            }
            
        except Exception as e:
            logger.error(f"Error calculating commit frequency for {file_path}: {e}")
            return {
                'commit_frequency': 0.0,
                'commits_per_week': 0.0,
                'total_commits': 0,
                'time_window_days': time_window_days
            }
    
    def get_file_age(
        self,
        repo_path: str,
        file_path: str
    ) -> Dict:
        """
        Get the age of a file (when it was first created).
        
        Args:
            repo_path: Path to the repository.
            file_path: Path to the file.
        
        Returns:
            Dictionary with age metrics.
        """
        try:
            commits = self.git_service.get_commit_history(
                repo_path,
                file_path=file_path,
                max_count=1
            )
            
            if not commits:
                return {
                    'file_age_days': 0,
                    'first_commit_date': None,
                    'first_commit_hash': None
                }
            
            # The last commit in history (oldest) is when file was created
            first_commit = commits[-1]
            first_commit_date = first_commit.get('date')
            
            if isinstance(first_commit_date, str):
                try:
                    first_commit_date = datetime.fromisoformat(first_commit_date.replace(' ', 'T'))
                except:
                    first_commit_date = None
            
            if first_commit_date:
                age_days = (datetime.now() - first_commit_date).days
            else:
                age_days = 0
            
            return {
                'file_age_days': age_days,
                'first_commit_date': first_commit_date.isoformat() if first_commit_date else None,
                'first_commit_hash': first_commit.get('hash')
            }
            
        except Exception as e:
            logger.error(f"Error getting file age for {file_path}: {e}")
            return {
                'file_age_days': 0,
                'first_commit_date': None,
                'first_commit_hash': None
            }
    
    def get_all_process_metrics(
        self,
        repo_path: str,
        file_path: str,
        commit_hash: Optional[str] = None,
        time_window_days: int = 30
    ) -> Dict:
        """
        Get all process metrics for a file.
        
        Args:
            repo_path: Path to the repository.
            file_path: Path to the file.
            commit_hash: Optional commit hash.
            time_window_days: Time window for metrics.
        
        Returns:
            Dictionary with all process metrics.
        """
        logger.info(f"Calculating process metrics for {file_path}")
        
        churn_metrics = self.calculate_churn(
            repo_path,
            file_path,
            time_window_days=time_window_days,
            commit_hash=commit_hash
        )
        
        developer_metrics = self.count_developers(
            repo_path,
            file_path,
            time_window_days=time_window_days * 3  # Longer window for developers
        )
        
        frequency_metrics = self.calculate_commit_frequency(
            repo_path,
            file_path,
            time_window_days=time_window_days
        )
        
        age_metrics = self.get_file_age(repo_path, file_path)
        
        # Combine all metrics
        all_metrics = {
            **churn_metrics,
            'developer_count': developer_metrics['developer_count'],
            'most_active_developer': developer_metrics['most_active_developer'],
            **frequency_metrics,
            **age_metrics
        }
        
        return all_metrics
    
    def add_process_metrics_to_features(
        self,
        repo_path: str,
        file_path: str,
        existing_features: Dict,
        commit_hash: Optional[str] = None
    ) -> Dict:
        """
        Add process metrics to existing feature dictionary.
        
        Args:
            repo_path: Path to the repository.
            file_path: Path to the file.
            existing_features: Existing feature dictionary.
            commit_hash: Optional commit hash.
        
        Returns:
            Updated feature dictionary with process metrics.
        """
        process_metrics = self.get_all_process_metrics(
            repo_path,
            file_path,
            commit_hash=commit_hash
        )
        
        # Add process metrics to features
        updated_features = existing_features.copy()
        updated_features.update({
            'churn_30d': process_metrics.get('churn', 0),
            'additions_30d': process_metrics.get('additions', 0),
            'deletions_30d': process_metrics.get('deletions', 0),
            'net_churn_30d': process_metrics.get('net_churn', 0),
            'commit_count_30d': process_metrics.get('commit_count', 0),
            'developer_count_90d': process_metrics.get('developer_count', 0),
            'commit_frequency_30d': process_metrics.get('commit_frequency', 0.0),
            'commits_per_week': process_metrics.get('commits_per_week', 0.0),
            'file_age_days': process_metrics.get('file_age_days', 0)
        })
        
        return updated_features

