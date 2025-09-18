"""
Severity classification system for kernel analysis suggestions
"""

import re
from typing import Dict, Any, List, Tuple
from ..critic_models import SeverityLevel, Suggestion, CorrectnessCheck, PerformanceMetrics


class SeverityClassifier:
    """Classify suggestion severity based on impact and context"""
    
    def __init__(self):
        self.name = "severity_classifier"
        self.description = "Classify suggestion severity based on impact and context"
        
        # Severity rules and patterns
        self.severity_rules = {
            "correctness": {
                "critical": [
                    "out of bounds", "buffer overflow", "null pointer", "dereference",
                    "race condition", "data corruption", "crash", "segfault"
                ],
                "high": [
                    "bounds check", "synchronization", "memory safety", "type safety",
                    "undefined behavior", "potential crash"
                ],
                "medium": [
                    "warning", "potential issue", "best practice", "style"
                ],
                "low": [
                    "optimization", "performance", "efficiency", "cleanup"
                ]
            },
            "performance": {
                "critical": [
                    "memory leak", "infinite loop", "exponential complexity"
                ],
                "high": [
                    "memory bound", "no tiling", "no vectorization", "poor memory access",
                    "bottleneck", "major performance issue"
                ],
                "medium": [
                    "optimization opportunity", "can be improved", "consider",
                    "performance gain", "efficiency"
                ],
                "low": [
                    "minor optimization", "cleanup", "style improvement"
                ]
            },
            "hardware": {
                "high": [
                    "tensor core", "hardware specific", "major optimization",
                    "significant performance gain"
                ],
                "medium": [
                    "hardware optimization", "platform specific", "moderate gain"
                ],
                "low": [
                    "minor hardware optimization", "cosmetic improvement"
                ]
            }
        }
        
        # Impact scoring factors
        self.impact_factors = {
            "correctness_impact": 3.0,  # Correctness issues are most critical
            "performance_impact": 2.0,  # Performance issues are important
            "hardware_impact": 1.5,     # Hardware optimizations are valuable
            "frequency_factor": 1.2,    # More frequent issues get higher priority
            "context_factor": 1.1       # Context-aware adjustments
        }
    
    def classify_suggestion(self, suggestion: Suggestion, 
                          context: Dict[str, Any]) -> SeverityLevel:
        """Classify a single suggestion's severity"""
        
        # Start with base severity
        base_severity = suggestion.severity
        
        # Apply context-based adjustments
        adjusted_severity = self._apply_context_adjustments(suggestion, context)
        
        # Apply impact-based adjustments
        impact_severity = self._apply_impact_adjustments(suggestion, context)
        
        # Take the most severe classification
        severity_levels = [base_severity, adjusted_severity, impact_severity]
        severity_values = [self._severity_to_value(s) for s in severity_levels]
        max_severity_idx = severity_values.index(max(severity_values))
        
        return severity_levels[max_severity_idx]
    
    def classify_batch_suggestions(self, suggestions: List[Suggestion], 
                                 context: Dict[str, Any]) -> List[Suggestion]:
        """Classify a batch of suggestions"""
        classified_suggestions = []
        
        for suggestion in suggestions:
            classified_severity = self.classify_suggestion(suggestion, context)
            suggestion.severity = classified_severity
            classified_suggestions.append(suggestion)
        
        return classified_suggestions
    
    def _apply_context_adjustments(self, suggestion: Suggestion, 
                                 context: Dict[str, Any]) -> SeverityLevel:
        """Apply context-based severity adjustments"""
        
        # Check for critical context indicators
        critical_indicators = [
            "production", "critical_path", "real_time", "safety_critical"
        ]
        
        if any(indicator in str(context).lower() for indicator in critical_indicators):
            return self._upgrade_severity(suggestion.severity, 2)
        
        # Check for development context
        dev_indicators = [
            "development", "testing", "prototype", "experimental"
        ]
        
        if any(indicator in str(context).lower() for indicator in dev_indicators):
            return self._downgrade_severity(suggestion.severity, 1)
        
        return suggestion.severity
    
    def _apply_impact_adjustments(self, suggestion: Suggestion, 
                                context: Dict[str, Any]) -> SeverityLevel:
        """Apply impact-based severity adjustments"""
        
        # Analyze suggestion content for impact indicators
        content = f"{suggestion.title} {suggestion.message}".lower()
        
        # Check for high-impact keywords
        high_impact_keywords = [
            "critical", "severe", "major", "significant", "substantial",
            "break", "fail", "error", "crash", "corruption"
        ]
        
        if any(keyword in content for keyword in high_impact_keywords):
            return self._upgrade_severity(suggestion.severity, 2)
        
        # Check for medium-impact keywords
        medium_impact_keywords = [
            "important", "considerable", "noticeable", "moderate",
            "warning", "caution", "attention"
        ]
        
        if any(keyword in content for keyword in medium_impact_keywords):
            return self._upgrade_severity(suggestion.severity, 1)
        
        # Check for low-impact keywords
        low_impact_keywords = [
            "minor", "small", "slight", "optional", "nice to have",
            "cleanup", "style", "cosmetic"
        ]
        
        if any(keyword in content for keyword in low_impact_keywords):
            return self._downgrade_severity(suggestion.severity, 1)
        
        return suggestion.severity
    
    def _upgrade_severity(self, current_severity: SeverityLevel, levels: int) -> SeverityLevel:
        """Upgrade severity by specified levels"""
        severity_order = [
            SeverityLevel.LOW,
            SeverityLevel.MEDIUM,
            SeverityLevel.HIGH,
            SeverityLevel.CRITICAL
        ]
        
        try:
            current_index = severity_order.index(current_severity)
            new_index = min(len(severity_order) - 1, current_index + levels)
            return severity_order[new_index]
        except ValueError:
            return current_severity
    
    def _downgrade_severity(self, current_severity: SeverityLevel, levels: int) -> SeverityLevel:
        """Downgrade severity by specified levels"""
        severity_order = [
            SeverityLevel.LOW,
            SeverityLevel.MEDIUM,
            SeverityLevel.HIGH,
            SeverityLevel.CRITICAL
        ]
        
        try:
            current_index = severity_order.index(current_severity)
            new_index = max(0, current_index - levels)
            return severity_order[new_index]
        except ValueError:
            return current_severity
    
    def _severity_to_value(self, severity: SeverityLevel) -> int:
        """Convert severity level to numeric value for comparison"""
        severity_values = {
            SeverityLevel.LOW: 1,
            SeverityLevel.MEDIUM: 2,
            SeverityLevel.HIGH: 3,
            SeverityLevel.CRITICAL: 4
        }
        return severity_values.get(severity, 0)
    
    def analyze_suggestion_patterns(self, suggestions: List[Suggestion]) -> Dict[str, Any]:
        """Analyze patterns in suggestions for insights"""
        
        analysis = {
            "total_suggestions": len(suggestions),
            "severity_distribution": {},
            "category_distribution": {},
            "common_issues": [],
            "priority_areas": []
        }
        
        # Count severity distribution
        for suggestion in suggestions:
            severity = suggestion.severity.value
            analysis["severity_distribution"][severity] = analysis["severity_distribution"].get(severity, 0) + 1
            
            category = suggestion.category
            analysis["category_distribution"][category] = analysis["category_distribution"].get(category, 0) + 1
        
        # Identify common issues
        issue_keywords = {}
        for suggestion in suggestions:
            words = suggestion.title.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    issue_keywords[word] = issue_keywords.get(word, 0) + 1
        
        # Get top common issues
        analysis["common_issues"] = sorted(issue_keywords.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Identify priority areas
        high_severity_categories = [
            cat for cat, count in analysis["category_distribution"].items()
            if any(s.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL] 
                  for s in suggestions if s.category == cat)
        ]
        analysis["priority_areas"] = high_severity_categories
        
        return analysis
    
    def get_severity_explanation(self, severity: SeverityLevel) -> str:
        """Get human-readable explanation of severity level"""
        explanations = {
            SeverityLevel.LOW: "Low priority - Minor optimization or style improvement",
            SeverityLevel.MEDIUM: "Medium priority - Performance improvement or best practice",
            SeverityLevel.HIGH: "High priority - Important optimization or potential issue",
            SeverityLevel.CRITICAL: "Critical priority - Must fix to prevent crashes or errors"
        }
        return explanations.get(severity, "Unknown severity level")
    
    def suggest_priority_order(self, suggestions: List[Suggestion]) -> List[Suggestion]:
        """Suggest optimal priority order for addressing suggestions"""
        
        # Score each suggestion based on multiple factors
        scored_suggestions = []
        
        for suggestion in suggestions:
            score = self._calculate_priority_score(suggestion)
            scored_suggestions.append((score, suggestion))
        
        # Sort by score (highest first)
        scored_suggestions.sort(key=lambda x: x[0], reverse=True)
        
        # Return suggestions in priority order
        return [suggestion for score, suggestion in scored_suggestions]
    
    def _calculate_priority_score(self, suggestion: Suggestion) -> float:
        """Calculate priority score for a suggestion"""
        
        # Base score from severity
        severity_scores = {
            SeverityLevel.LOW: 1.0,
            SeverityLevel.MEDIUM: 2.0,
            SeverityLevel.HIGH: 3.0,
            SeverityLevel.CRITICAL: 4.0
        }
        base_score = severity_scores.get(suggestion.severity, 0.0)
        
        # Category multiplier
        category_multipliers = {
            "correctness": 1.5,  # Correctness issues are most important
            "memory": 1.3,       # Memory issues are very important
            "compute": 1.2,      # Compute optimizations are important
            "hardware": 1.1,     # Hardware optimizations are valuable
            "style": 0.8         # Style issues are less important
        }
        category_multiplier = category_multipliers.get(suggestion.category, 1.0)
        
        # Code snippet bonus (suggestions with code are more actionable)
        code_bonus = 1.2 if suggestion.code_snippet else 1.0
        
        # Calculate final score
        final_score = base_score * category_multiplier * code_bonus
        
        return final_score
