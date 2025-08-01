#!/usr/bin/env python3
"""
Terragon Autonomous Scheduler
Manages continuous value discovery and execution cycles.
"""

import asyncio
import json
import schedule
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
import subprocess
import sys

from autonomous_discovery import ValueDiscoveryEngine
from value_executor import ValueExecutor


class AutonomousScheduler:
    """Manages scheduled autonomous SDLC improvements."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.discovery_engine = ValueDiscoveryEngine(repo_path)
        self.executor = ValueExecutor(repo_path)
        self.metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        
        # Setup schedules based on configuration
        self._setup_schedules()
    
    def _setup_schedules(self):
        """Setup automated schedules for value discovery and execution."""
        
        # Immediate execution after PR merge (triggered externally)
        # Hourly security scans
        schedule.every().hour.do(self._run_security_scan)
        
        # Daily comprehensive analysis
        schedule.every().day.at("02:00").do(self._run_daily_analysis)
        
        # Weekly deep review
        schedule.every().monday.at("03:00").do(self._run_weekly_review)
        
        # Monthly strategic review
        schedule.every().month.do(self._run_monthly_review)
    
    async def _run_security_scan(self):
        """Run focused security vulnerability scan."""
        print(f"🔒 Running hourly security scan at {datetime.now()}")
        
        try:
            # Run security-focused discovery
            items = []
            
            # Security scan
            items.extend(await self.discovery_engine._run_security_scan())
            
            # Dependency vulnerabilities
            items.extend(await self.discovery_engine._analyze_dependencies())
            
            if items:
                prioritized = await self.discovery_engine.prioritize_items(items)
                critical_security = [i for i in prioritized if i.get('type') == 'security' and i.get('composite_score', 0) > 70]
                
                if critical_security:
                    print(f"🚨 Found {len(critical_security)} critical security issues")
                    # Auto-execute critical security fixes
                    for item in critical_security[:1]:  # Execute top 1 to avoid overwhelming
                        await self.executor.execute_item(item)
                else:
                    print("✅ No critical security issues found")
            else:
                print("✅ Security scan clean")
                
        except Exception as e:
            print(f"❌ Security scan failed: {e}")
    
    async def _run_daily_analysis(self):
        """Run comprehensive daily analysis and execution."""
        print(f"📊 Running daily analysis at {datetime.now()}")
        
        try:
            # Full discovery cycle
            await self.discovery_engine.run_discovery_cycle()
            
            # Execute next best value item
            result = await self.executor.execute_next_value_item()
            
            if result and result.get('success'):
                print(f"✅ Daily execution completed: {result.get('title')}")
            else:
                print("ℹ️ No items executed in daily cycle")
                
        except Exception as e:
            print(f"❌ Daily analysis failed: {e}")
    
    async def _run_weekly_review(self):
        """Run deep weekly SDLC assessment."""
        print(f"📈 Running weekly deep review at {datetime.now()}")
        
        try:
            # Comprehensive analysis
            await self.discovery_engine.run_discovery_cycle()
            
            # Execute multiple items (up to 5 hours of work)
            total_effort = 0
            max_weekly_effort = 5  # hours
            
            while total_effort < max_weekly_effort:
                result = await self.executor.execute_next_value_item()
                
                if not result or not result.get('success'):
                    break
                
                effort = result.get('actualEffort', 0)
                total_effort += effort
                
                print(f"✅ Weekly item completed: {result.get('title')} ({effort:.1f}h)")
            
            print(f"📊 Weekly review completed: {total_effort:.1f} hours of improvements")
            
        except Exception as e:
            print(f"❌ Weekly review failed: {e}")
    
    async def _run_monthly_review(self):
        """Run strategic monthly review and recalibration."""
        print(f"🎯 Running monthly strategic review at {datetime.now()}")
        
        try:
            # Load metrics for analysis
            with open(self.metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Analyze execution history
            history = metrics.get('executionHistory', [])
            if len(history) >= 5:  # Need some data for analysis
                # Calculate accuracy metrics
                effort_ratios = []
                impact_scores = []
                
                for record in history[-10:]:  # Last 10 executions
                    estimated = record.get('estimatedEffort', 1)
                    actual = record.get('actualEffort', 1)
                    if estimated > 0 and actual > 0:
                        effort_ratios.append(actual / estimated)
                    
                    if record.get('success'):
                        impact_scores.append(7)  # Placeholder impact scoring
                    else:
                        impact_scores.append(1)
                
                # Update learning metrics
                if effort_ratios:
                    avg_accuracy = sum(effort_ratios) / len(effort_ratios)
                    metrics['learningMetrics']['estimationAccuracy'] = min(1.0, 2.0 - avg_accuracy)
                
                if impact_scores:
                    avg_impact = sum(impact_scores) / len(impact_scores)
                    metrics['learningMetrics']['valuePredictionAccuracy'] = avg_impact / 10
                
                # Increment adaptation cycles
                metrics['learningMetrics']['adaptationCycles'] = metrics['learningMetrics'].get('adaptationCycles', 0) + 1
                
                # Save updated metrics
                with open(self.metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                print(f"📊 Model accuracy: {metrics['learningMetrics']['estimationAccuracy']:.2f}")
                print(f"📈 Value prediction: {metrics['learningMetrics']['valuePredictionAccuracy']:.2f}")
            
            # Run comprehensive discovery
            await self.discovery_engine.run_discovery_cycle()
            
            print("✅ Monthly strategic review completed")
            
        except Exception as e:
            print(f"❌ Monthly review failed: {e}")
    
    async def handle_pr_merge_trigger(self):
        """Handle immediate execution after PR merge."""
        print("🚀 PR merge detected - running immediate value discovery")
        
        try:
            # Quick discovery cycle
            await self.discovery_engine.run_discovery_cycle()
            
            # Execute one high-value item
            result = await self.executor.execute_next_value_item()
            
            if result and result.get('success'):
                print(f"⚡ Immediate execution completed: {result.get('title')}")
            
        except Exception as e:
            print(f"❌ PR merge trigger failed: {e}")
    
    def run_scheduler(self):
        """Run the continuous scheduler."""
        print("🤖 Starting Terragon Autonomous Scheduler")
        print("📅 Scheduled tasks:")
        print("   - Hourly: Security scans")
        print("   - Daily 02:00: Comprehensive analysis + execution")
        print("   - Weekly Monday 03:00: Deep review + multiple executions")
        print("   - Monthly: Strategic review + model recalibration")
        print("   - On PR merge: Immediate value discovery")
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                print("\n👋 Scheduler stopped by user")
                break
            except Exception as e:
                print(f"❌ Scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    async def run_single_cycle(self, cycle_type: str = "daily"):
        """Run a single execution cycle for testing."""
        if cycle_type == "security":
            await self._run_security_scan()
        elif cycle_type == "daily":
            await self._run_daily_analysis()
        elif cycle_type == "weekly":
            await self._run_weekly_review()
        elif cycle_type == "monthly":
            await self._run_monthly_review()
        elif cycle_type == "pr_merge":
            await self.handle_pr_merge_trigger()
        else:
            print(f"Unknown cycle type: {cycle_type}")


async def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Terragon Autonomous Scheduler")
    parser.add_argument("--mode", choices=["scheduler", "single"], default="single",
                       help="Run continuous scheduler or single cycle")
    parser.add_argument("--cycle", choices=["security", "daily", "weekly", "monthly", "pr_merge"], 
                       default="daily", help="Cycle type for single mode")
    
    args = parser.parse_args()
    
    scheduler = AutonomousScheduler()
    
    if args.mode == "scheduler":
        scheduler.run_scheduler()
    else:
        await scheduler.run_single_cycle(args.cycle)


if __name__ == "__main__":
    # Install required package if not available
    try:
        import schedule
    except ImportError:
        print("Installing required package: schedule")
        subprocess.run([sys.executable, "-m", "pip", "install", "schedule"])
        import schedule
    
    asyncio.run(main())