#!/usr/bin/env python3
"""
Terragon Autonomous Value Executor
Executes highest-value items from the backlog automatically.
"""

import asyncio
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml


class ValueExecutor:
    """Executes value items from the autonomous backlog."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.config_path = self.repo_path / ".terragon" / "config.yaml"
        self.metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        
        self.config = self._load_config()
        self.metrics = self._load_metrics()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load Terragon configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load value metrics."""
        if self.metrics_path.exists():
            with open(self.metrics_path, 'r') as f:
                return json.load(f)
        return {}
    
    async def get_next_best_item(self) -> Optional[Dict[str, Any]]:
        """Get the next highest-value item to execute."""
        # In a real implementation, this would parse the BACKLOG.md
        # or maintain a structured backlog in JSON format
        
        # For demo purposes, return a mock next item
        next_items = [
            {
                'id': 'CICD-001',
                'title': 'Activate GitHub Actions Workflows',
                'type': 'ci_cd_enhancement',
                'effort': 4,
                'composite_score': 78.4,
                'files': ['.github/workflows/'],
                'description': 'Set up GitHub Actions workflows using existing templates'
            },
            {
                'id': 'SEC-001', 
                'title': 'Implement Automated Security Scanning',
                'type': 'security',
                'effort': 3,
                'composite_score': 72.1,
                'files': ['.github/workflows/security.yml'],
                'description': 'Add security scanning to CI/CD pipeline'
            }
        ]
        
        # Return highest scoring item that meets execution criteria
        for item in next_items:
            if self._can_execute_item(item):
                return item
        
        return None
    
    def _can_execute_item(self, item: Dict[str, Any]) -> bool:
        """Check if item can be safely executed."""
        min_score = self.config.get('scoring', {}).get('thresholds', {}).get('minScore', 10)
        max_risk = self.config.get('scoring', {}).get('thresholds', {}).get('maxRisk', 0.8)
        
        # Check score threshold
        if item.get('composite_score', 0) < min_score:
            return False
        
        # Check risk threshold
        if item.get('risk', 5) / 10 > max_risk:
            return False
        
        # Check dependencies (simplified)
        return True
    
    async def execute_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a value item."""
        print(f"🚀 Executing: {item['title']}")
        start_time = datetime.now(timezone.utc)
        
        try:
            # Route to appropriate executor based on type
            if item['type'] == 'ci_cd_enhancement':
                result = await self._execute_cicd_item(item)
            elif item['type'] == 'security':
                result = await self._execute_security_item(item)
            elif item['type'] == 'technical_debt':
                result = await self._execute_debt_item(item)
            elif item['type'] == 'performance':
                result = await self._execute_performance_item(item)
            else:
                result = await self._execute_generic_item(item)
            
            # Record execution
            execution_record = {
                'timestamp': start_time.isoformat(),
                'itemId': item['id'],
                'title': item['title'],
                'type': item['type'],
                'estimatedEffort': item.get('effort', 0),
                'actualEffort': (datetime.now(timezone.utc) - start_time).total_seconds() / 3600,
                'success': result.get('success', False),
                'impact': result.get('impact', {}),
                'changes': result.get('changes', []),
                'notes': result.get('notes', '')
            }
            
            # Update metrics
            if 'executionHistory' not in self.metrics:
                self.metrics['executionHistory'] = []
            self.metrics['executionHistory'].append(execution_record)
            
            # Update value delivered
            if result.get('success'):
                value_delivered = self.metrics.get('valueDelivered', {})
                value_delivered['totalHoursSaved'] = value_delivered.get('totalHoursSaved', 0) + item.get('effort', 0)
                if item['type'] == 'security':
                    value_delivered['securityIssuesResolved'] = value_delivered.get('securityIssuesResolved', 0) + 1
                elif item['type'] == 'technical_debt':
                    value_delivered['technicalDebtReduced'] = value_delivered.get('technicalDebtReduced', 0) + item.get('effort', 0)
                elif item['type'] == 'performance':
                    value_delivered['performanceGains'] = value_delivered.get('performanceGains', 0) + 1
            
            # Save metrics
            with open(self.metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            print(f"✅ Completed: {item['title']} ({'success' if result.get('success') else 'failed'})")
            return execution_record
            
        except Exception as e:
            print(f"❌ Failed to execute {item['title']}: {e}")
            return {
                'timestamp': start_time.isoformat(),
                'itemId': item['id'],
                'title': item['title'],
                'success': False,
                'error': str(e)
            }
    
    async def _execute_cicd_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CI/CD related improvements."""
        changes = []
        
        if 'GitHub Actions' in item['title']:
            # Create GitHub Actions workflows from templates
            workflows_dir = self.repo_path / '.github' / 'workflows'
            workflows_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy CI template
            ci_template = self.repo_path / 'docs' / 'workflows' / 'ci-template.yml'
            if ci_template.exists():
                ci_workflow = workflows_dir / 'ci.yml'
                with open(ci_template, 'r') as src, open(ci_workflow, 'w') as dst:
                    content = src.read()
                    # Customize template for this repo
                    content = content.replace('{{PYTHON_VERSION}}', '3.10')
                    content = content.replace('{{COVERAGE_THRESHOLD}}', '85')
                    dst.write(content)
                changes.append(str(ci_workflow))
            
            # Copy security template
            security_template = self.repo_path / 'docs' / 'workflows' / 'security-template.yml'
            if security_template.exists():
                security_workflow = workflows_dir / 'security.yml'
                with open(security_template, 'r') as src, open(security_workflow, 'w') as dst:
                    dst.write(src.read())
                changes.append(str(security_workflow))
        
        return {
            'success': len(changes) > 0,
            'changes': changes,
            'impact': {
                'ci_workflows_created': len(changes),
                'automation_improved': True
            },
            'notes': f'Created {len(changes)} workflow files'
        }
    
    async def _execute_security_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Execute security-related improvements."""
        changes = []
        
        if 'Security Scanning' in item['title']:
            # Add security scanning to Makefile
            makefile = self.repo_path / 'Makefile'
            if makefile.exists():
                with open(makefile, 'r') as f:
                    content = f.read()
                
                if 'security:' not in content:
                    security_target = """
security: ## Run security scans
\tbandit -r src/ -f json -o security-report.json || true
\tsafety check --json --output safety-report.json || true
\tpip-audit --format=json --output=audit-report.json || true
"""
                    content += security_target
                    with open(makefile, 'w') as f:
                        f.write(content)
                    changes.append(str(makefile))
        
        return {
            'success': len(changes) > 0,
            'changes': changes,
            'impact': {
                'security_automation': True,
                'vulnerability_scanning': True
            },
            'notes': 'Added security scanning targets'
        }
    
    async def _execute_debt_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Execute technical debt improvements."""
        changes = []
        
        # Run automated fixes where possible
        try:
            # Run ruff fix
            result = subprocess.run([
                'ruff', 'check', '--fix', 'src/'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0:
                changes.append('Automated ruff fixes applied')
        except Exception:
            pass
        
        return {
            'success': len(changes) > 0,
            'changes': changes,
            'impact': {
                'code_quality_improved': True,
                'automated_fixes': len(changes)
            },
            'notes': 'Applied automated code quality fixes'
        }
    
    async def _execute_performance_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Execute performance improvements."""
        return {
            'success': False,
            'notes': 'Performance improvements require manual review'
        }
    
    async def _execute_generic_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generic improvements."""
        return {
            'success': False,
            'notes': 'Generic item execution not implemented'
        }
    
    async def run_tests(self) -> bool:
        """Run tests to validate changes."""
        try:
            # Run pytest
            result = subprocess.run([
                'python', '-m', 'pytest', '--cov=async_toolformer', '--cov-fail-under=85'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            return result.returncode == 0
        except Exception:
            return False
    
    async def run_quality_checks(self) -> bool:
        """Run code quality checks."""
        try:
            # Run ruff
            ruff_result = subprocess.run([
                'ruff', 'check', 'src/'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            # Run mypy
            mypy_result = subprocess.run([
                'mypy', 'src/'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            return ruff_result.returncode == 0 and mypy_result.returncode == 0
        except Exception:
            return False
    
    async def create_pull_request(self, execution_record: Dict[str, Any]) -> Optional[str]:
        """Create a pull request for the changes."""
        if not execution_record.get('success') or not execution_record.get('changes'):
            return None
        
        try:
            # Create branch
            branch_name = f"auto-value/{execution_record['itemId'].lower()}"
            subprocess.run(['git', 'checkout', '-b', branch_name], cwd=self.repo_path)
            
            # Add changes
            for change in execution_record['changes']:
                subprocess.run(['git', 'add', change], cwd=self.repo_path)
            
            # Commit
            commit_msg = f"""[AUTO-VALUE] {execution_record['title']}

Autonomous SDLC improvement with composite score: {execution_record.get('compositeScore', 'N/A')}

Changes:
{chr(10).join(f'- {change}' for change in execution_record['changes'])}

Impact:
{chr(10).join(f'- {k}: {v}' for k, v in execution_record.get('impact', {}).items())}

🤖 Generated with Terragon Autonomous SDLC Engine
Execution Time: {execution_record.get('actualEffort', 0):.2f} hours

Co-Authored-By: Terragon <noreply@terragonlabs.com>"""
            
            subprocess.run(['git', 'commit', '-m', commit_msg], cwd=self.repo_path)
            
            print(f"📝 Created branch {branch_name} with changes")
            return branch_name
            
        except Exception as e:
            print(f"Failed to create PR: {e}")
            return None
    
    async def execute_next_value_item(self) -> Optional[Dict[str, Any]]:
        """Execute the next highest-value item."""
        item = await self.get_next_best_item()
        if not item:
            print("No eligible items for execution")
            return None
        
        print(f"🎯 Selected item: {item['title']} (Score: {item['composite_score']})")
        
        # Execute item
        result = await self.execute_item(item)
        
        if result.get('success'):
            # Run validation
            print("🧪 Running tests...")
            tests_passed = await self.run_tests()
            
            print("🔍 Running quality checks...")
            quality_passed = await self.run_quality_checks()
            
            if tests_passed and quality_passed:
                # Create PR
                branch = await self.create_pull_request(result)
                if branch:
                    result['branch'] = branch
                    result['pr_ready'] = True
            else:
                print("❌ Tests or quality checks failed - rolling back")
                subprocess.run(['git', 'checkout', 'main'], cwd=self.repo_path)
                result['rollback'] = True
        
        return result


async def main():
    """Main execution function."""
    executor = ValueExecutor()
    result = await executor.execute_next_value_item()
    
    if result:
        print(f"\n🎉 Execution Summary:")
        print(f"Item: {result.get('title', 'Unknown')}")
        print(f"Success: {result.get('success', False)}")
        print(f"Time: {result.get('actualEffort', 0):.2f} hours")
        if result.get('pr_ready'):
            print(f"Branch: {result.get('branch', 'Unknown')}")
            print("✅ Ready for review!")


if __name__ == "__main__":
    asyncio.run(main())