#!/usr/bin/env python3
"""
Terragon Autonomous Value Discovery Engine
Continuously discovers, prioritizes, and executes highest-value SDLC improvements.
"""

import asyncio
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml


class ValueDiscoveryEngine:
    """Discovers and prioritizes technical debt, security issues, and improvements."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.config_path = self.repo_path / ".terragon" / "config.yaml"
        self.metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        self.backlog_path = self.repo_path / "BACKLOG.md"
        
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
    
    async def discover_value_items(self) -> List[Dict[str, Any]]:
        """Discover technical debt and improvement opportunities."""
        items = []
        
        # Git history analysis
        items.extend(await self._analyze_git_history())
        
        # Static analysis
        items.extend(await self._run_static_analysis())
        
        # Security scanning
        items.extend(await self._run_security_scan())
        
        # Performance analysis
        items.extend(await self._analyze_performance())
        
        # Dependencies analysis
        items.extend(await self._analyze_dependencies())
        
        return items
    
    async def _analyze_git_history(self) -> List[Dict[str, Any]]:
        """Analyze git history for TODOs, FIXMEs, and debt markers."""
        items = []
        
        try:
            # Search for TODO/FIXME/HACK patterns
            result = subprocess.run([
                'git', 'grep', '-n', '-E', 'TODO|FIXME|HACK|XXX|DEPRECATED',
                '--', '*.py', '*.js', '*.ts', '*.md'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            for line in result.stdout.splitlines():
                if ':' in line:
                    file_path, line_num, content = line.split(':', 2)
                    items.append({
                        'id': f"git-{hash(line) % 10000}",
                        'title': f"Address {content.strip()[:50]}...",
                        'file': file_path,
                        'line': int(line_num),
                        'type': 'technical_debt',
                        'source': 'git_history',
                        'effort': 3,
                        'impact': 5,
                        'risk': 3
                    })
        except Exception as e:
            print(f"Git analysis failed: {e}")
        
        return items
    
    async def _run_static_analysis(self) -> List[Dict[str, Any]]:
        """Run static analysis tools."""
        items = []
        
        # Run ruff for code quality issues
        try:
            result = subprocess.run([
                'ruff', 'check', '--output-format=json', 'src/'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.stdout:
                ruff_issues = json.loads(result.stdout)
                for issue in ruff_issues:
                    items.append({
                        'id': f"ruff-{issue.get('code', 'unknown')}",
                        'title': f"Fix {issue.get('code')}: {issue.get('message', 'Unknown issue')}",
                        'file': issue.get('filename', 'unknown'),
                        'line': issue.get('location', {}).get('row', 0),
                        'type': 'code_quality',
                        'source': 'static_analysis',
                        'effort': 2,
                        'impact': 4,
                        'risk': 2
                    })
        except Exception as e:
            print(f"Static analysis failed: {e}")
        
        return items
    
    async def _run_security_scan(self) -> List[Dict[str, Any]]:
        """Run security scanning tools."""
        items = []
        
        # Run bandit for security issues
        try:
            result = subprocess.run([
                'bandit', '-r', 'src/', '-f', 'json'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.stdout:
                bandit_report = json.loads(result.stdout)
                for issue in bandit_report.get('results', []):
                    items.append({
                        'id': f"security-{issue.get('test_id', 'unknown')}",
                        'title': f"Security: {issue.get('issue_text', 'Security issue')}",
                        'file': issue.get('filename', 'unknown'),
                        'line': issue.get('line_number', 0),
                        'type': 'security',
                        'source': 'security_scan',
                        'effort': 4,
                        'impact': 8,
                        'risk': 7,
                        'severity': issue.get('issue_severity', 'medium')
                    })
        except Exception as e:
            print(f"Security scan failed: {e}")
        
        return items
    
    async def _analyze_performance(self) -> List[Dict[str, Any]]:
        """Analyze performance bottlenecks."""
        items = []
        
        # Look for common performance anti-patterns
        patterns = [
            ('time.sleep', 'Blocking sleep call'),
            ('requests.get', 'Synchronous HTTP call'),
            ('for.*in.*range.*len', 'Inefficient iteration pattern')
        ]
        
        for pattern, description in patterns:
            try:
                result = subprocess.run([
                    'grep', '-rn', pattern, 'src/'
                ], capture_output=True, text=True, cwd=self.repo_path)
                
                for line in result.stdout.splitlines():
                    if ':' in line:
                        file_path, line_num, content = line.split(':', 2)
                        items.append({
                            'id': f"perf-{hash(line) % 10000}",
                            'title': f"Performance: {description}",
                            'file': file_path,
                            'line': int(line_num),
                            'type': 'performance',
                            'source': 'performance_analysis',
                            'effort': 3,
                            'impact': 6,
                            'risk': 4
                        })
            except Exception:
                continue
        
        return items
    
    async def _analyze_dependencies(self) -> List[Dict[str, Any]]:
        """Analyze dependencies for updates and vulnerabilities."""
        items = []
        
        # Check for outdated dependencies
        try:
            # Run pip-audit if available
            result = subprocess.run([
                'pip-audit', '--format=json'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.stdout:
                audit_report = json.loads(result.stdout)
                for vuln in audit_report.get('vulnerabilities', []):
                    items.append({
                        'id': f"dep-{vuln.get('id', 'unknown')}",
                        'title': f"Update vulnerable dependency: {vuln.get('package', 'unknown')}",
                        'file': 'pyproject.toml',
                        'line': 0,
                        'type': 'dependency_vulnerability',
                        'source': 'dependency_analysis',
                        'effort': 2,
                        'impact': 7,
                        'risk': 6,
                        'cve': vuln.get('id', '')
                    })
        except Exception as e:
            print(f"Dependency analysis failed: {e}")
        
        return items
    
    def calculate_composite_score(self, item: Dict[str, Any]) -> float:
        """Calculate WSJF + ICE + Technical Debt composite score."""
        weights = self.config.get('scoring', {}).get('weights', {}).get('advanced', {})
        
        # WSJF Score
        cost_of_delay = (
            item.get('impact', 5) * 0.4 +
            item.get('urgency', 3) * 0.3 +
            item.get('risk', 3) * 0.2 +
            item.get('opportunity', 3) * 0.1
        )
        job_size = max(item.get('effort', 3), 1)
        wsjf = cost_of_delay / job_size
        
        # ICE Score
        ice = item.get('impact', 5) * item.get('confidence', 7) * (10 - item.get('effort', 3))
        
        # Technical Debt Score
        debt_score = item.get('impact', 5) * item.get('risk', 3)
        
        # Security boost
        security_boost = 2.0 if item.get('type') == 'security' else 1.0
        
        # Composite score
        composite = (
            weights.get('wsjf', 0.5) * wsjf +
            weights.get('ice', 0.1) * (ice / 100) +
            weights.get('technicalDebt', 0.3) * debt_score +
            weights.get('security', 0.1) * security_boost
        ) * 10
        
        return round(composite, 1)
    
    async def prioritize_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize items using composite scoring."""
        for item in items:
            item['composite_score'] = self.calculate_composite_score(item)
        
        return sorted(items, key=lambda x: x['composite_score'], reverse=True)
    
    async def update_backlog(self, items: List[Dict[str, Any]]) -> None:
        """Update the BACKLOG.md file with new discoveries."""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Update metrics
        self.metrics['lastUpdated'] = timestamp
        self.metrics['backlogMetrics']['totalItems'] = len(items)
        
        if items:
            high_priority = len([i for i in items if i['composite_score'] > 60])
            medium_priority = len([i for i in items if 30 <= i['composite_score'] <= 60])
            low_priority = len([i for i in items if i['composite_score'] < 30])
            
            self.metrics['backlogMetrics']['priorityDistribution'] = {
                'critical': len([i for i in items if i['composite_score'] > 80]),
                'high': high_priority,
                'medium': medium_priority,
                'low': low_priority
            }
        
        # Save updated metrics
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"Updated backlog with {len(items)} items at {timestamp}")
    
    async def run_discovery_cycle(self) -> None:
        """Run a complete discovery and prioritization cycle."""
        print("🔍 Starting autonomous value discovery cycle...")
        
        # Discover items
        items = await self.discover_value_items()
        print(f"📋 Discovered {len(items)} potential value items")
        
        # Prioritize items
        prioritized_items = await self.prioritize_items(items)
        print(f"🎯 Prioritized {len(prioritized_items)} items")
        
        # Update backlog
        await self.update_backlog(prioritized_items)
        
        # Show top items
        if prioritized_items:
            print("\n🏆 Top 5 Value Items:")
            for i, item in enumerate(prioritized_items[:5], 1):
                print(f"{i}. [{item['composite_score']:.1f}] {item['title']}")
        
        print("✅ Discovery cycle completed!")


async def main():
    """Main execution function."""
    engine = ValueDiscoveryEngine()
    await engine.run_discovery_cycle()


if __name__ == "__main__":
    asyncio.run(main())