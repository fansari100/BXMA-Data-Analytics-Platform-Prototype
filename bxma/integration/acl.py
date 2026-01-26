"""
Anti-Corruption Layer (ACL)
===========================

Isolates the clean Titan-X domain model from legacy systems.

The ACL pattern ensures that:
1. Legacy quirks don't pollute core domain logic
2. Format/protocol translation is contained
3. Schema evolution is managed gracefully
4. Core system remains testable in isolation

Key Components:
- LegacyAdapter: Protocol adaptation (SOAP, FIX, etc.)
- MessageConverter: Message format conversion
- SchemaTransformer: Data schema transformation
- ValidationGateway: Input validation and sanitization

References:
- Evans (2003): Domain-Driven Design - Anti-Corruption Layer pattern
- Fowler: Enterprise Integration Patterns

Author: BXMA Quant Team
Date: January 2026
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Protocol, TypeVar
from enum import Enum, auto
import json
import xml.etree.ElementTree as ET
import re


T = TypeVar('T')


class LegacyProtocol(Enum):
    """Supported legacy protocols."""
    REST_JSON = auto()
    SOAP_XML = auto()
    FIX = auto()
    CSV = auto()
    PROPRIETARY = auto()


class MessageFormat(Enum):
    """Message formats."""
    JSON = auto()
    XML = auto()
    PROTOBUF = auto()
    AVRO = auto()
    ARROW = auto()


@dataclass
class LegacySystemConfig:
    """Configuration for a legacy system integration."""
    
    system_id: str
    name: str
    protocol: LegacyProtocol
    
    # Connection
    endpoint: str = ""
    port: int = 0
    
    # Authentication
    auth_type: str = "none"  # none, basic, oauth, certificate
    credentials: dict[str, str] = field(default_factory=dict)
    
    # Timeouts
    connect_timeout_ms: int = 5000
    read_timeout_ms: int = 30000
    
    # Retry
    max_retries: int = 3
    retry_delay_ms: int = 1000
    
    # Schema
    schema_version: str = "1.0"
    schema_path: str = ""


@dataclass
class TransformationRule:
    """Rule for transforming data between schemas."""
    
    source_field: str
    target_field: str
    
    # Transformation
    transform_type: str = "direct"  # direct, rename, convert, compute
    transform_func: Callable[[Any], Any] | None = None
    
    # Type conversion
    source_type: str = ""
    target_type: str = ""
    
    # Validation
    required: bool = False
    default_value: Any = None


@dataclass
class ValidationResult:
    """Result of message validation."""
    
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    # Sanitization applied
    sanitized: bool = False
    sanitization_actions: list[str] = field(default_factory=list)


class LegacyAdapter(Protocol):
    """Protocol for legacy system adapters."""
    
    def connect(self) -> bool:
        """Establish connection to legacy system."""
        ...
    
    def disconnect(self) -> None:
        """Close connection."""
        ...
    
    def send(self, message: bytes) -> bytes:
        """Send message and receive response."""
        ...
    
    def receive(self) -> bytes:
        """Receive incoming message."""
        ...


class SOAPAdapter:
    """
    Adapter for SOAP/XML legacy systems.
    
    Handles:
    - SOAP envelope construction
    - XML parsing and generation
    - Namespace handling
    - WSDL-based message validation
    """
    
    def __init__(self, config: LegacySystemConfig):
        self.config = config
        self._connected = False
        
        # SOAP namespaces
        self.namespaces = {
            "soap": "http://schemas.xmlsoap.org/soap/envelope/",
            "xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsd": "http://www.w3.org/2001/XMLSchema",
        }
    
    def connect(self) -> bool:
        """Establish connection (placeholder for real HTTP client)."""
        self._connected = True
        return True
    
    def disconnect(self) -> None:
        """Close connection."""
        self._connected = False
    
    def build_envelope(self, body: dict) -> str:
        """Build SOAP envelope from body dict."""
        # Create envelope
        envelope = ET.Element(
            "{http://schemas.xmlsoap.org/soap/envelope/}Envelope",
            nsmap=self.namespaces if hasattr(ET, 'nsmap') else {},
        )
        
        # Add body
        soap_body = ET.SubElement(
            envelope,
            "{http://schemas.xmlsoap.org/soap/envelope/}Body",
        )
        
        # Convert dict to XML
        self._dict_to_xml(soap_body, body)
        
        return ET.tostring(envelope, encoding="unicode")
    
    def _dict_to_xml(self, parent: ET.Element, data: dict):
        """Convert dictionary to XML elements."""
        for key, value in data.items():
            child = ET.SubElement(parent, key)
            if isinstance(value, dict):
                self._dict_to_xml(child, value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self._dict_to_xml(child, item)
                    else:
                        child.text = str(item)
            else:
                child.text = str(value)
    
    def parse_response(self, xml_str: str) -> dict:
        """Parse SOAP response to dictionary."""
        root = ET.fromstring(xml_str)
        
        # Find body
        body = root.find(".//{http://schemas.xmlsoap.org/soap/envelope/}Body")
        if body is None:
            body = root.find(".//Body")
        
        if body is None:
            return {}
        
        return self._xml_to_dict(body)
    
    def _xml_to_dict(self, element: ET.Element) -> dict:
        """Convert XML element to dictionary."""
        result = {}
        
        for child in element:
            tag = child.tag.split("}")[-1]  # Remove namespace
            
            if len(child) > 0:
                result[tag] = self._xml_to_dict(child)
            else:
                result[tag] = child.text
        
        return result
    
    def send(self, message: bytes) -> bytes:
        """Send SOAP message (placeholder)."""
        # In production, this would use aiohttp or requests
        return b"<soap:Envelope><soap:Body><Response>OK</Response></soap:Body></soap:Envelope>"


class FIXAdapter:
    """
    Adapter for FIX protocol systems.
    
    Handles:
    - FIX message parsing and generation
    - Session management
    - Sequence number tracking
    - Heartbeat handling
    """
    
    def __init__(self, config: LegacySystemConfig):
        self.config = config
        self._connected = False
        self._seq_num = 1
        
        # FIX field mappings
        self.field_names = {
            8: "BeginString",
            9: "BodyLength",
            35: "MsgType",
            49: "SenderCompID",
            56: "TargetCompID",
            34: "MsgSeqNum",
            52: "SendingTime",
            10: "CheckSum",
            # Trade fields
            11: "ClOrdID",
            55: "Symbol",
            54: "Side",
            38: "OrderQty",
            44: "Price",
            40: "OrdType",
        }
    
    def connect(self) -> bool:
        """Establish FIX session."""
        self._connected = True
        self._seq_num = 1
        return True
    
    def disconnect(self) -> None:
        """Close FIX session."""
        self._connected = False
    
    def parse_message(self, fix_str: str) -> dict:
        """Parse FIX message string to dictionary."""
        fields = {}
        
        # Split by delimiter (SOH = \x01)
        parts = fix_str.replace("\x01", "|").split("|")
        
        for part in parts:
            if "=" in part:
                tag, value = part.split("=", 1)
                try:
                    tag_num = int(tag)
                    field_name = self.field_names.get(tag_num, f"Tag{tag_num}")
                    fields[field_name] = value
                except ValueError:
                    continue
        
        return fields
    
    def build_message(self, msg_type: str, fields: dict) -> str:
        """Build FIX message from dictionary."""
        # Required header fields
        header = {
            8: "FIX.4.4",
            35: msg_type,
            49: self.config.credentials.get("sender_comp_id", "TITAN"),
            56: self.config.credentials.get("target_comp_id", "TARGET"),
            34: self._seq_num,
            52: datetime.now().strftime("%Y%m%d-%H:%M:%S"),
        }
        
        # Build message body
        body_parts = []
        for tag, value in header.items():
            body_parts.append(f"{tag}={value}")
        
        for field_name, value in fields.items():
            # Find tag number
            for tag, name in self.field_names.items():
                if name == field_name:
                    body_parts.append(f"{tag}={value}")
                    break
        
        body = "\x01".join(body_parts) + "\x01"
        
        # Add body length and checksum
        body_len = len(body)
        checksum = sum(ord(c) for c in body) % 256
        
        message = f"8=FIX.4.4\x019={body_len}\x01{body}10={checksum:03d}\x01"
        
        self._seq_num += 1
        return message
    
    def send(self, message: bytes) -> bytes:
        """Send FIX message (placeholder)."""
        return b"8=FIX.4.4|35=8|..."


class MessageConverter:
    """
    Converts messages between formats.
    
    Supports:
    - JSON <-> XML
    - JSON <-> Protobuf
    - Legacy formats <-> Arrow
    """
    
    def __init__(self):
        self._converters: dict[tuple[MessageFormat, MessageFormat], Callable] = {
            (MessageFormat.JSON, MessageFormat.XML): self._json_to_xml,
            (MessageFormat.XML, MessageFormat.JSON): self._xml_to_json,
        }
    
    def convert(
        self,
        data: Any,
        source_format: MessageFormat,
        target_format: MessageFormat,
    ) -> Any:
        """Convert data between formats."""
        if source_format == target_format:
            return data
        
        converter = self._converters.get((source_format, target_format))
        if converter is None:
            raise ValueError(f"No converter for {source_format} -> {target_format}")
        
        return converter(data)
    
    def _json_to_xml(self, data: dict) -> str:
        """Convert JSON dict to XML string."""
        root = ET.Element("root")
        self._dict_to_xml_recursive(root, data)
        return ET.tostring(root, encoding="unicode")
    
    def _dict_to_xml_recursive(self, parent: ET.Element, data: dict):
        """Recursively convert dict to XML."""
        for key, value in data.items():
            child = ET.SubElement(parent, str(key))
            if isinstance(value, dict):
                self._dict_to_xml_recursive(child, value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    item_elem = ET.SubElement(child, "item")
                    if isinstance(item, dict):
                        self._dict_to_xml_recursive(item_elem, item)
                    else:
                        item_elem.text = str(item)
            else:
                child.text = str(value) if value is not None else ""
    
    def _xml_to_json(self, xml_str: str) -> dict:
        """Convert XML string to JSON dict."""
        root = ET.fromstring(xml_str)
        return self._xml_to_dict_recursive(root)
    
    def _xml_to_dict_recursive(self, element: ET.Element) -> dict:
        """Recursively convert XML to dict."""
        result = {}
        for child in element:
            if len(child) > 0:
                result[child.tag] = self._xml_to_dict_recursive(child)
            else:
                result[child.tag] = child.text
        return result


class SchemaTransformer:
    """
    Transforms data between schemas.
    
    Handles:
    - Field mapping and renaming
    - Type conversion
    - Computed fields
    - Default values
    """
    
    def __init__(self, rules: list[TransformationRule]):
        self.rules = rules
        self._rule_map = {rule.source_field: rule for rule in rules}
    
    def transform(self, source_data: dict) -> dict:
        """Transform data according to rules."""
        result = {}
        
        for rule in self.rules:
            value = source_data.get(rule.source_field)
            
            # Handle missing required fields
            if value is None:
                if rule.required:
                    raise ValueError(f"Missing required field: {rule.source_field}")
                value = rule.default_value
            
            # Apply transformation
            if rule.transform_func is not None:
                value = rule.transform_func(value)
            elif rule.transform_type == "convert":
                value = self._convert_type(value, rule.source_type, rule.target_type)
            
            # Set in result
            if value is not None:
                result[rule.target_field] = value
        
        return result
    
    def _convert_type(self, value: Any, source_type: str, target_type: str) -> Any:
        """Convert value between types."""
        if value is None:
            return None
        
        converters = {
            ("string", "float"): float,
            ("string", "int"): int,
            ("string", "bool"): lambda x: x.lower() in ("true", "1", "yes"),
            ("float", "string"): str,
            ("int", "string"): str,
            ("float", "int"): int,
            ("int", "float"): float,
        }
        
        converter = converters.get((source_type, target_type))
        if converter:
            return converter(value)
        
        return value


class ValidationGateway:
    """
    Validates and sanitizes incoming data.
    
    Ensures data integrity before entering the core domain.
    """
    
    def __init__(self):
        self._validators: dict[str, Callable[[Any], bool]] = {}
        self._sanitizers: dict[str, Callable[[Any], Any]] = {}
    
    def add_validator(self, field: str, validator: Callable[[Any], bool]):
        """Add a field validator."""
        self._validators[field] = validator
    
    def add_sanitizer(self, field: str, sanitizer: Callable[[Any], Any]):
        """Add a field sanitizer."""
        self._sanitizers[field] = sanitizer
    
    def validate(self, data: dict) -> ValidationResult:
        """Validate data."""
        errors = []
        warnings = []
        
        for field, validator in self._validators.items():
            if field in data:
                try:
                    if not validator(data[field]):
                        errors.append(f"Validation failed for field: {field}")
                except Exception as e:
                    errors.append(f"Validation error for {field}: {str(e)}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
    
    def sanitize(self, data: dict) -> tuple[dict, list[str]]:
        """Sanitize data."""
        result = data.copy()
        actions = []
        
        for field, sanitizer in self._sanitizers.items():
            if field in result:
                original = result[field]
                result[field] = sanitizer(original)
                if result[field] != original:
                    actions.append(f"Sanitized {field}")
        
        return result, actions


class AntiCorruptionLayer:
    """
    Main Anti-Corruption Layer orchestrator.
    
    Coordinates:
    - Protocol adaptation
    - Message conversion
    - Schema transformation
    - Validation
    
    Ensures clean data enters the Titan-X domain.
    """
    
    def __init__(self):
        self._adapters: dict[str, LegacyAdapter] = {}
        self._transformers: dict[str, SchemaTransformer] = {}
        self._converter = MessageConverter()
        self._gateway = ValidationGateway()
    
    def register_adapter(self, system_id: str, adapter: LegacyAdapter):
        """Register a legacy system adapter."""
        self._adapters[system_id] = adapter
    
    def register_transformer(self, system_id: str, transformer: SchemaTransformer):
        """Register a schema transformer."""
        self._transformers[system_id] = transformer
    
    def ingest(
        self,
        system_id: str,
        raw_data: bytes,
        source_format: MessageFormat = MessageFormat.JSON,
    ) -> dict:
        """
        Ingest data from a legacy system.
        
        1. Parse according to protocol
        2. Convert format if needed
        3. Transform schema
        4. Validate
        5. Return clean domain data
        """
        # Parse
        if source_format == MessageFormat.JSON:
            parsed = json.loads(raw_data.decode())
        elif source_format == MessageFormat.XML:
            parsed = self._converter._xml_to_json(raw_data.decode())
        else:
            parsed = {}
        
        # Transform schema
        transformer = self._transformers.get(system_id)
        if transformer:
            parsed = transformer.transform(parsed)
        
        # Validate
        validation = self._gateway.validate(parsed)
        if not validation.is_valid:
            raise ValueError(f"Validation failed: {validation.errors}")
        
        # Sanitize
        sanitized, _ = self._gateway.sanitize(parsed)
        
        return sanitized
    
    def send(
        self,
        system_id: str,
        domain_data: dict,
        target_format: MessageFormat = MessageFormat.JSON,
    ) -> bytes:
        """
        Send data to a legacy system.
        
        1. Transform schema (reverse)
        2. Convert format
        3. Send via adapter
        """
        # In production, would have reverse transformers
        
        # Convert format
        if target_format == MessageFormat.JSON:
            return json.dumps(domain_data).encode()
        elif target_format == MessageFormat.XML:
            return self._converter._json_to_xml(domain_data).encode()
        
        return b""
