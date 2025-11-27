"""
Jira Issue Tracker Integration Service
Connects to Jira API to fetch issue information and link with Git commits.
"""

import re
import logging
import requests
from typing import Dict, List, Optional, Set
from datetime import datetime
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JiraService:
    """
    Service for integrating with Jira issue tracker.
    """
    
    def __init__(
        self,
        jira_url: Optional[str] = None,
        username: Optional[str] = None,
        api_token: Optional[str] = None,
        project_key: Optional[str] = None
    ):
        """
        Initialize Jira Service.
        
        Args:
            jira_url: Base URL of Jira instance (e.g., 'https://yourcompany.atlassian.net')
            username: Jira username/email
            api_token: Jira API token (from https://id.atlassian.com/manage-profile/security/api-tokens)
            project_key: Optional project key to filter issues (e.g., 'PROJ')
        """
        self.jira_url = jira_url
        self.username = username
        self.api_token = api_token
        self.project_key = project_key
        self.session = None
        
        if jira_url and username and api_token:
            self._setup_session()
    
    def _setup_session(self):
        """Setup authenticated session with Jira."""
        self.session = requests.Session()
        self.session.auth = (self.username, self.api_token)
        self.session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
    
    def extract_issue_keys(self, text: str) -> Set[str]:
        """
        Extract Jira issue keys from text (commit messages, etc.).
        
        Args:
            text: Text to search for issue keys
        
        Returns:
            Set of issue keys (e.g., {'PROJ-123', 'PROJ-456'})
        """
        if not text:
            return set()
        
        # Pattern: PROJECT-KEY-123 or PROJ-123
        pattern = r'\b([A-Z][A-Z0-9_]+-\d+)\b'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        # Also check for lowercase versions and convert
        lowercase_matches = re.findall(r'\b([a-z][a-z0-9_]+-\d+)\b', text)
        matches.extend([m.upper() for m in lowercase_matches])
        
        return set(matches)
    
    def get_issue(self, issue_key: str) -> Optional[Dict]:
        """
        Fetch issue information from Jira.
        
        Args:
            issue_key: Jira issue key (e.g., 'PROJ-123')
        
        Returns:
            Dictionary with issue information or None if not found
        """
        if not self.session or not self.jira_url:
            logger.warning("Jira not configured. Cannot fetch issue.")
            return None
        
        try:
            url = urljoin(self.jira_url, f'/rest/api/3/issue/{issue_key}')
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                issue_data = response.json()
                return self._parse_issue(issue_data)
            elif response.status_code == 404:
                logger.warning(f"Issue {issue_key} not found")
                return None
            else:
                logger.error(f"Error fetching issue {issue_key}: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Exception fetching issue {issue_key}: {e}")
            return None
    
    def _parse_issue(self, issue_data: Dict) -> Dict:
        """
        Parse Jira issue JSON response.
        
        Args:
            issue_data: Raw Jira API response
        
        Returns:
            Parsed issue dictionary
        """
        fields = issue_data.get('fields', {})
        
        # Extract key information
        issue_type = fields.get('issuetype', {}).get('name', 'Unknown')
        summary = fields.get('summary', '')
        description = fields.get('description', '')
        status = fields.get('status', {}).get('name', 'Unknown')
        priority = fields.get('priority', {}).get('name', 'Unknown')
        
        # Check for performance-related labels/components
        labels = fields.get('labels', [])
        components = [c.get('name', '') for c in fields.get('components', [])]
        
        # Check if it's a performance issue
        is_performance_issue = self._is_performance_issue(
            summary, description, labels, components, issue_type
        )
        
        return {
            'key': issue_data.get('key', ''),
            'summary': summary,
            'description': description,
            'issue_type': issue_type,
            'status': status,
            'priority': priority,
            'labels': labels,
            'components': components,
            'is_performance_issue': is_performance_issue,
            'created': fields.get('created', ''),
            'updated': fields.get('updated', ''),
            'resolved': fields.get('resolutiondate', ''),
        }
    
    def _is_performance_issue(self, summary: str, description: str, labels: List[str], 
                             components: List[str], issue_type: str) -> bool:
        """
        Determine if an issue is performance-related.
        
        Args:
            summary: Issue summary
            description: Issue description
            labels: Issue labels
            components: Issue components
            issue_type: Issue type
        
        Returns:
            True if issue appears to be performance-related
        """
        performance_keywords = [
            'performance', 'slow', 'latency', 'bottleneck', 'optimize', 'optimization',
            'speed', 'throughput', 'efficiency', 'memory leak', 'cpu usage',
            'deadlock', 'race condition', 'timeout', 'hang', 'stall', 'lag',
            'scalability', 'concurrency', 'thread', 'async', 'blocking'
        ]
        
        text_to_check = f"{summary} {description}".lower()
        text_to_check += " " + " ".join(labels).lower()
        text_to_check += " " + " ".join(components).lower()
        text_to_check += " " + issue_type.lower()
        
        for keyword in performance_keywords:
            if keyword in text_to_check:
                return True
        
        return False
    
    def get_issues_from_commits(self, commit_messages: List[str]) -> Dict[str, Dict]:
        """
        Extract and fetch Jira issues referenced in commit messages.
        
        Args:
            commit_messages: List of commit messages
        
        Returns:
            Dictionary mapping issue keys to issue information
        """
        issues = {}
        
        for commit_msg in commit_messages:
            issue_keys = self.extract_issue_keys(commit_msg)
            for issue_key in issue_keys:
                if issue_key not in issues:
                    issue_info = self.get_issue(issue_key)
                    if issue_info:
                        issues[issue_key] = issue_info
        
        return issues
    
    def enhance_bug_fix_detection(self, commit_message: str, issue_info: Optional[Dict] = None) -> bool:
        """
        Enhance bug-fix detection using Jira issue information.
        
        Args:
            commit_message: Commit message
            issue_info: Optional Jira issue information
        
        Returns:
            True if commit appears to be a bug fix (enhanced with Jira data)
        """
        # Basic check: if issue is marked as bug and resolved
        if issue_info:
            issue_type = issue_info.get('issue_type', '').lower()
            status = issue_info.get('status', '').lower()
            
            # If it's a bug issue and resolved/fixed, likely a bug fix
            if 'bug' in issue_type and status in ['resolved', 'closed', 'done', 'fixed']:
                return True
            
            # If it's a performance issue and resolved, likely a performance bug fix
            if issue_info.get('is_performance_issue', False) and status in ['resolved', 'closed', 'done']:
                return True
        
        return False
    
    def search_issues(
        self,
        jql: Optional[str] = None,
        project: Optional[str] = None,
        issue_type: Optional[str] = None,
        status: Optional[str] = None,
        max_results: int = 50
    ) -> List[Dict]:
        """
        Search for issues using JQL (Jira Query Language).
        
        Args:
            jql: JQL query string
            project: Project key to filter
            issue_type: Issue type to filter
            status: Status to filter
            max_results: Maximum number of results
        
        Returns:
            List of issue dictionaries
        """
        if not self.session or not self.jira_url:
            logger.warning("Jira not configured. Cannot search issues.")
            return []
        
        # Build JQL query
        if not jql:
            conditions = []
            if project:
                conditions.append(f"project = {project}")
            if issue_type:
                conditions.append(f"issuetype = {issue_type}")
            if status:
                conditions.append(f"status = {status}")
            
            jql = " AND ".join(conditions) if conditions else "ORDER BY created DESC"
        
        try:
            url = urljoin(self.jira_url, '/rest/api/3/search')
            params = {
                'jql': jql,
                'maxResults': max_results,
                'fields': 'summary,description,issuetype,status,priority,labels,components,created,updated,resolutiondate'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                issues = []
                for issue_data in data.get('issues', []):
                    parsed_issue = self._parse_issue(issue_data)
                    issues.append(parsed_issue)
                return issues
            else:
                logger.error(f"Error searching issues: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Exception searching issues: {e}")
            return []

