"""
Dual LLM Security Pattern for Prompt Injection Protection
=========================================================

Implements a two-layer LLM architecture to protect against prompt injection
attacks from malicious external content (news, social media, etc.).

Architecture:
- Reader LLM: Processes untrusted external content, NO tool access
- Thinker LLM: Processes sanitized JSON, HAS tool access

The Reader never sees tool schemas, and the Thinker never sees raw external text.
This eliminates the injection vector.

References:
- Simon Willison (2024): Dual LLM Pattern for Prompt Injection
- OWASP (2025): LLM Application Security Guidelines
- Anthropic (2025): Constitutional AI Security

Author: BXMA Quant Team  
Date: January 2026
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable
import hashlib


class ThreatLevel(Enum):
    """Threat level classification for content."""
    SAFE = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class SecurityConfig:
    """Configuration for the security layer."""
    
    # Content filtering
    max_input_length: int = 10000
    strip_html: bool = True
    strip_urls: bool = False
    
    # Injection detection
    detect_injection_patterns: bool = True
    block_code_injection: bool = True
    
    # Rate limiting
    max_requests_per_minute: int = 60
    
    # Logging
    log_all_inputs: bool = True
    log_sanitized_outputs: bool = True
    
    # Blocking
    auto_block_on_detection: bool = True
    quarantine_suspicious: bool = True


@dataclass
class SanitizedInput:
    """
    Sanitized and structured input from the Reader LLM.
    
    This is the ONLY format the Thinker LLM receives.
    """
    
    # Unique identifier for audit trail
    id: str = ""
    
    # Source information
    source_type: str = ""  # "news", "social", "filing", "transcript"
    source_name: str = ""
    original_timestamp: str = ""
    
    # Extracted structured data (NO raw text)
    entities: list[str] = field(default_factory=list)  # Companies, people, etc.
    sentiment: float = 0.0  # -1 to 1
    topics: list[str] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)
    numerical_data: dict[str, float] = field(default_factory=dict)
    
    # Risk signals
    risk_keywords: list[str] = field(default_factory=list)
    urgency: str = "normal"  # "low", "normal", "high", "critical"
    
    # Security metadata
    threat_level: ThreatLevel = ThreatLevel.SAFE
    injection_detected: bool = False
    sanitization_notes: list[str] = field(default_factory=list)
    
    # Processing metadata
    processed_at: datetime = field(default_factory=datetime.now)
    reader_model: str = ""
    confidence: float = 0.0
    
    def to_json(self) -> str:
        """Convert to JSON for Thinker LLM consumption."""
        return json.dumps({
            "id": self.id,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "entities": self.entities,
            "sentiment": self.sentiment,
            "topics": self.topics,
            "key_facts": self.key_facts,
            "numerical_data": self.numerical_data,
            "risk_keywords": self.risk_keywords,
            "urgency": self.urgency,
            "confidence": self.confidence,
        }, indent=2)
    
    def is_safe(self) -> bool:
        """Check if input is safe for processing."""
        return not self.injection_detected and self.threat_level in [ThreatLevel.SAFE, ThreatLevel.LOW]


class InjectionDetector:
    """
    Detects potential prompt injection attempts in text.
    
    Uses pattern matching and heuristics to identify malicious content.
    """
    
    # Known injection patterns
    INJECTION_PATTERNS = [
        r"ignore\s+(previous|above|all)\s+(instructions?|prompts?)",
        r"disregard\s+(everything|all)",
        r"forget\s+(everything|what)",
        r"you\s+are\s+now\s+a",
        r"new\s+instructions?:",
        r"system\s*:\s*",
        r"<\s*system\s*>",
        r"\[\s*INST\s*\]",
        r"```\s*(system|admin|root)",
        r"override\s+(mode|instructions?)",
        r"jailbreak",
        r"DAN\s+mode",
        r"act\s+as\s+(if|though)",
        r"pretend\s+(you|that)",
        r"roleplay\s+as",
    ]
    
    # Suspicious code patterns
    CODE_PATTERNS = [
        r"<script",
        r"javascript:",
        r"eval\s*\(",
        r"exec\s*\(",
        r"import\s+os",
        r"subprocess",
        r"__import__",
        r"\$\{.*\}",  # Template injection
        r"\{\{.*\}\}",  # Jinja injection
    ]
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]
        self.compiled_code_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.CODE_PATTERNS
        ]
    
    def detect(self, text: str) -> tuple[bool, list[str]]:
        """
        Detect injection attempts in text.
        
        Returns:
            Tuple of (injection_detected, list_of_matched_patterns)
        """
        matched = []
        
        # Check injection patterns
        if self.config.detect_injection_patterns:
            for pattern in self.compiled_patterns:
                if pattern.search(text):
                    matched.append(f"injection_pattern: {pattern.pattern}")
        
        # Check code patterns
        if self.config.block_code_injection:
            for pattern in self.compiled_code_patterns:
                if pattern.search(text):
                    matched.append(f"code_pattern: {pattern.pattern}")
        
        # Check for suspicious Unicode
        if self._has_suspicious_unicode(text):
            matched.append("suspicious_unicode")
        
        # Check for hidden text techniques
        if self._has_hidden_text(text):
            matched.append("hidden_text")
        
        return len(matched) > 0, matched
    
    def _has_suspicious_unicode(self, text: str) -> bool:
        """Check for suspicious Unicode characters used in attacks."""
        suspicious_ranges = [
            (0x200B, 0x200F),  # Zero-width characters
            (0x2028, 0x202F),  # Line/paragraph separators
            (0xFEFF, 0xFEFF),  # BOM
            (0xE0000, 0xE007F),  # Tags block
        ]
        
        for char in text:
            code = ord(char)
            for start, end in suspicious_ranges:
                if start <= code <= end:
                    return True
        return False
    
    def _has_hidden_text(self, text: str) -> bool:
        """Check for hidden text techniques."""
        # Check for excessive whitespace that might hide content
        if re.search(r'\s{50,}', text):
            return True
        
        # Check for text that looks like it's trying to be invisible
        if re.search(r'color:\s*white|opacity:\s*0|font-size:\s*0', text, re.IGNORECASE):
            return True
        
        return False


class ReaderLLM:
    """
    Reader LLM: Processes untrusted external content.
    
    This LLM:
    - Has NO access to tools or system functions
    - Only outputs structured JSON
    - Never sees tool schemas or system prompts
    """
    
    def __init__(
        self,
        config: SecurityConfig,
        llm_interface: Callable[[str], str] | None = None,
    ):
        self.config = config
        self.llm = llm_interface or self._default_llm
        self.detector = InjectionDetector(config)
        self.model_name = "reader-llm-v1"
    
    def _default_llm(self, prompt: str) -> str:
        """Default LLM placeholder."""
        return json.dumps({
            "entities": ["Sample Corp"],
            "sentiment": 0.0,
            "topics": ["market"],
            "key_facts": ["Market update"],
            "numerical_data": {},
        })
    
    def _preprocess(self, text: str) -> str:
        """Preprocess text before LLM processing."""
        # Truncate if too long
        if len(text) > self.config.max_input_length:
            text = text[:self.config.max_input_length] + "...[TRUNCATED]"
        
        # Strip HTML if configured
        if self.config.strip_html:
            text = re.sub(r'<[^>]+>', '', text)
        
        # Strip URLs if configured
        if self.config.strip_urls:
            text = re.sub(r'https?://\S+', '[URL]', text)
        
        return text
    
    def process(
        self,
        text: str,
        source_type: str,
        source_name: str,
    ) -> SanitizedInput:
        """
        Process untrusted text into sanitized structured data.
        
        Args:
            text: Raw untrusted text
            source_type: Type of source (news, social, etc.)
            source_name: Name of the source
            
        Returns:
            SanitizedInput with structured, safe data
        """
        # Generate unique ID
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
        input_id = f"{source_type}_{content_hash}"
        
        # Check for injection attempts FIRST
        injection_detected, patterns = self.detector.detect(text)
        
        if injection_detected and self.config.auto_block_on_detection:
            return SanitizedInput(
                id=input_id,
                source_type=source_type,
                source_name=source_name,
                threat_level=ThreatLevel.HIGH,
                injection_detected=True,
                sanitization_notes=[f"Blocked: {p}" for p in patterns],
                reader_model=self.model_name,
            )
        
        # Preprocess
        processed_text = self._preprocess(text)
        
        # Build extraction prompt (NO tool information)
        prompt = f"""Extract structured information from this {source_type} content.
Source: {source_name}

Content:
{processed_text}

Extract the following as JSON:
- entities: List of companies, people, or organizations mentioned
- sentiment: Overall sentiment from -1 (negative) to 1 (positive)
- topics: Main topics discussed
- key_facts: Important factual statements (max 5)
- numerical_data: Any numbers with their context (e.g., "revenue": 1.5)
- risk_keywords: Any words suggesting risk (e.g., "decline", "lawsuit", "warning")
- urgency: "low", "normal", "high", or "critical"

Output ONLY valid JSON, no other text."""
        
        # Get LLM response
        response = self.llm(prompt)
        
        # Parse response
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {}
        except json.JSONDecodeError:
            data = {}
        
        return SanitizedInput(
            id=input_id,
            source_type=source_type,
            source_name=source_name,
            original_timestamp=datetime.now().isoformat(),
            entities=data.get("entities", []),
            sentiment=data.get("sentiment", 0.0),
            topics=data.get("topics", []),
            key_facts=data.get("key_facts", []),
            numerical_data=data.get("numerical_data", {}),
            risk_keywords=data.get("risk_keywords", []),
            urgency=data.get("urgency", "normal"),
            threat_level=ThreatLevel.LOW if injection_detected else ThreatLevel.SAFE,
            injection_detected=injection_detected,
            sanitization_notes=patterns if injection_detected else [],
            reader_model=self.model_name,
            confidence=0.8,
        )


class ThinkerLLM:
    """
    Thinker LLM: Processes sanitized JSON and has tool access.
    
    This LLM:
    - ONLY receives SanitizedInput JSON (never raw text)
    - HAS access to tools for analysis and action
    - Is protected from injection because it never sees raw external content
    """
    
    def __init__(
        self,
        config: SecurityConfig,
        llm_interface: Callable[[str], str] | None = None,
        tools: dict[str, Any] | None = None,
    ):
        self.config = config
        self.llm = llm_interface or self._default_llm
        self.tools = tools or {}
        self.model_name = "thinker-llm-v1"
    
    def _default_llm(self, prompt: str) -> str:
        """Default LLM placeholder."""
        return "Analyzing sanitized input..."
    
    def analyze(self, sanitized_input: SanitizedInput) -> dict:
        """
        Analyze sanitized input and determine actions.
        
        Args:
            sanitized_input: Safe, structured input from Reader LLM
            
        Returns:
            Analysis results and recommended actions
        """
        # Safety check
        if not sanitized_input.is_safe():
            return {
                "status": "blocked",
                "reason": "Input flagged as unsafe",
                "threat_level": sanitized_input.threat_level.name,
            }
        
        # Build analysis prompt with tool information
        tool_descriptions = "\n".join([
            f"- {name}: {tool.__doc__ or 'No description'}"
            for name, tool in self.tools.items()
        ])
        
        prompt = f"""Analyze this market intelligence and determine if action is needed.

Available tools:
{tool_descriptions}

Market Intelligence (sanitized and verified):
{sanitized_input.to_json()}

Based on this information:
1. Assess the risk implications
2. Determine if any portfolio adjustments are warranted
3. Recommend specific tool calls if action is needed

Respond with:
- risk_assessment: Your analysis of the risk
- action_needed: true/false
- recommended_actions: List of tool calls with parameters
- urgency: low/medium/high
- confidence: 0-1"""
        
        response = self.llm(prompt)
        
        return {
            "status": "analyzed",
            "input_id": sanitized_input.id,
            "raw_analysis": response,
            "processed_by": self.model_name,
        }


class DualLLMGateway:
    """
    Main gateway implementing the Dual LLM security pattern.
    
    All external content flows through:
    1. Reader LLM (untrusted content -> sanitized JSON)
    2. Thinker LLM (sanitized JSON -> analysis and actions)
    """
    
    def __init__(
        self,
        config: SecurityConfig | None = None,
        reader_llm: Callable[[str], str] | None = None,
        thinker_llm: Callable[[str], str] | None = None,
        tools: dict[str, Any] | None = None,
    ):
        self.config = config or SecurityConfig()
        self.reader = ReaderLLM(self.config, reader_llm)
        self.thinker = ThinkerLLM(self.config, thinker_llm, tools)
        
        # Audit log
        self.audit_log: list[dict] = []
    
    def process(
        self,
        content: str,
        source_type: str,
        source_name: str,
    ) -> dict:
        """
        Process external content through the dual LLM pipeline.
        
        Args:
            content: Raw untrusted content
            source_type: Type of source
            source_name: Name of the source
            
        Returns:
            Analysis results (or blocked status)
        """
        # Step 1: Reader sanitizes
        sanitized = self.reader.process(content, source_type, source_name)
        
        # Log
        self.audit_log.append({
            "stage": "reader",
            "input_id": sanitized.id,
            "source": source_name,
            "injection_detected": sanitized.injection_detected,
            "threat_level": sanitized.threat_level.name,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Check if blocked
        if not sanitized.is_safe():
            return {
                "status": "blocked",
                "reason": "Security threat detected",
                "input_id": sanitized.id,
                "threat_level": sanitized.threat_level.name,
            }
        
        # Step 2: Thinker analyzes
        analysis = self.thinker.analyze(sanitized)
        
        # Log
        self.audit_log.append({
            "stage": "thinker",
            "input_id": sanitized.id,
            "analysis_status": analysis.get("status"),
            "timestamp": datetime.now().isoformat(),
        })
        
        return analysis
    
    def get_audit_log(self) -> list[dict]:
        """Get the security audit log."""
        return self.audit_log
    
    def get_stats(self) -> dict:
        """Get security statistics."""
        total = len([l for l in self.audit_log if l["stage"] == "reader"])
        blocked = len([l for l in self.audit_log if l.get("injection_detected")])
        
        return {
            "total_processed": total,
            "blocked": blocked,
            "block_rate": blocked / max(total, 1),
            "threat_levels": {
                level.name: len([l for l in self.audit_log if l.get("threat_level") == level.name])
                for level in ThreatLevel
            },
        }
