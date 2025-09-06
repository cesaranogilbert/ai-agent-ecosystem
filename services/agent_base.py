"""
Agent Base Interface
Standardized base class for all AI agents with typed contracts and operational methods
"""

import os
import json
import logging
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

class AgentStatus(str, Enum):
    """Agent operational status"""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class TaskPriority(str, Enum):
    """Task execution priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SecurityLevel(str, Enum):
    """Security classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class AgentCapability(BaseModel):
    """Standard agent capability definition"""
    name: str = Field(..., description="Capability name")
    description: str = Field(..., description="Detailed capability description")
    input_types: List[str] = Field(..., description="Supported input data types")
    output_types: List[str] = Field(..., description="Generated output data types")
    processing_time: Optional[str] = Field(None, description="Typical processing time")
    resource_requirements: Dict[str, Any] = Field(default_factory=dict, description="Computational resource requirements")

class AgentRequest(BaseModel):
    """Standardized agent request format"""
    request_id: str = Field(..., description="Unique request identifier")
    agent_id: str = Field(..., description="Target agent identifier")
    capability: str = Field(..., description="Requested capability")
    parameters: Dict[str, Any] = Field(..., description="Request parameters")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Request priority")
    security_level: SecurityLevel = Field(default=SecurityLevel.INTERNAL, description="Security classification")
    deadline: Optional[datetime] = Field(None, description="Request deadline")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class AgentResponse(BaseModel):
    """Standardized agent response format"""
    request_id: str = Field(..., description="Original request identifier")
    agent_id: str = Field(..., description="Responding agent identifier")
    status: str = Field(..., description="Response status")
    result: Dict[str, Any] = Field(..., description="Response data")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    errors: List[str] = Field(default_factory=list, description="Error messages if any")

class SecurityValidation(BaseModel):
    """Security validation and compliance"""
    validation_passed: bool = Field(..., description="Overall validation status")
    security_level: SecurityLevel = Field(..., description="Required security level")
    compliance_checks: Dict[str, bool] = Field(..., description="Individual compliance check results")
    risk_assessment: Dict[str, Any] = Field(..., description="Risk analysis results")
    mitigation_measures: List[str] = Field(default_factory=list, description="Required mitigation measures")

class AgentMetrics(BaseModel):
    """Agent performance and operational metrics"""
    requests_processed: int = Field(default=0, description="Total requests processed")
    average_processing_time: float = Field(default=0.0, description="Average processing time in seconds")
    success_rate: float = Field(default=0.0, description="Success rate percentage")
    error_rate: float = Field(default=0.0, description="Error rate percentage")
    uptime_percentage: float = Field(default=0.0, description="Agent uptime percentage")
    resource_utilization: Dict[str, float] = Field(default_factory=dict, description="Resource utilization metrics")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last metrics update")

class AgentBase(ABC):
    """
    Abstract base class for all AI agents
    Provides standardized interface, security validation, and operational methods
    """
    
    def __init__(self, agent_id: str, version: str = "1.0.0"):
        """Initialize the agent with standard properties"""
        self.agent_id = agent_id
        self.version = version
        self.status = AgentStatus.INITIALIZING
        self.capabilities: List[AgentCapability] = []
        self.metrics = AgentMetrics()
        self.security_config = self._initialize_security_config()
        self.operational_config = self._initialize_operational_config()
        self._initialize_agent()
        self.status = AgentStatus.READY
        
    @abstractmethod
    def _initialize_agent(self) -> None:
        """Initialize agent-specific components"""
        pass
        
    @abstractmethod
    def get_capabilities(self) -> List[AgentCapability]:
        """Return list of agent capabilities"""
        pass
        
    def _initialize_security_config(self) -> Dict[str, Any]:
        """Initialize security configuration"""
        return {
            'max_security_level': SecurityLevel.CONFIDENTIAL,
            'encryption_required': True,
            'audit_logging': True,
            'access_controls': {
                'rate_limiting': True,
                'ip_filtering': False,
                'authentication_required': True
            },
            'data_protection': {
                'pii_detection': True,
                'data_anonymization': True,
                'secure_storage': True
            }
        }
        
    def _initialize_operational_config(self) -> Dict[str, Any]:
        """Initialize operational configuration"""
        return {
            'max_concurrent_requests': 10,
            'request_timeout': 300,  # 5 minutes
            'retry_attempts': 3,
            'health_check_interval': 60,  # 1 minute
            'metrics_collection': True,
            'performance_monitoring': True
        }
        
    async def validate_security(self, request: AgentRequest) -> SecurityValidation:
        """
        Validate request security and compliance requirements
        
        Args:
            request: Incoming agent request
            
        Returns:
            Security validation results
        """
        try:
            compliance_checks = {}
            risk_factors = []
            mitigation_measures = []
            
            # Security level validation
            max_allowed = self.security_config.get('max_security_level', SecurityLevel.INTERNAL)
            security_valid = request.security_level.value <= max_allowed.value
            compliance_checks['security_level'] = security_valid
            
            if not security_valid:
                risk_factors.append(f"Security level {request.security_level} exceeds maximum {max_allowed}")
                mitigation_measures.append("Reduce security level or upgrade agent clearance")
            
            # PII detection
            pii_detected = await self._detect_pii(request.parameters)
            compliance_checks['pii_protection'] = not pii_detected
            
            if pii_detected and request.security_level == SecurityLevel.PUBLIC:
                risk_factors.append("PII detected in public request")
                mitigation_measures.append("Anonymize data or increase security level")
            
            # Rate limiting check
            rate_limit_ok = await self._check_rate_limits(request.request_id)
            compliance_checks['rate_limiting'] = rate_limit_ok
            
            if not rate_limit_ok:
                risk_factors.append("Rate limit exceeded")
                mitigation_measures.append("Implement request throttling")
            
            # Data size validation
            data_size_ok = await self._validate_data_size(request.parameters)
            compliance_checks['data_size'] = data_size_ok
            
            if not data_size_ok:
                risk_factors.append("Request data size exceeds limits")
                mitigation_measures.append("Reduce payload size or use streaming")
            
            validation_passed = all(compliance_checks.values())
            
            return SecurityValidation(
                validation_passed=validation_passed,
                security_level=request.security_level,
                compliance_checks=compliance_checks,
                risk_assessment={
                    'risk_factors': risk_factors,
                    'risk_level': 'high' if not validation_passed else 'low',
                    'assessment_timestamp': datetime.utcnow().isoformat()
                },
                mitigation_measures=mitigation_measures
            )
            
        except Exception as e:
            logger.error(f"Security validation failed: {str(e)}")
            return SecurityValidation(
                validation_passed=False,
                security_level=request.security_level,
                compliance_checks={'validation_error': False},
                risk_assessment={'error': str(e)},
                mitigation_measures=['Manual security review required']
            )
    
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """
        Process incoming request with security validation and error handling
        
        Args:
            request: Standardized agent request
            
        Returns:
            Standardized agent response
        """
        start_time = datetime.utcnow()
        
        try:
            # Update status
            self.status = AgentStatus.PROCESSING
            
            # Security validation
            security_validation = await self.validate_security(request)
            if not security_validation.validation_passed:
                return AgentResponse(
                    request_id=request.request_id,
                    agent_id=self.agent_id,
                    status='security_rejected',
                    result={'security_validation': security_validation.dict()},
                    processing_time=0.0,
                    errors=[f"Security validation failed: {security_validation.risk_assessment}"]
                )
            
            # Capability validation
            if not self._validate_capability(request.capability):
                return AgentResponse(
                    request_id=request.request_id,
                    agent_id=self.agent_id,
                    status='capability_not_found',
                    result={},
                    processing_time=0.0,
                    errors=[f"Capability '{request.capability}' not supported"]
                )
            
            # Process the request
            result = await self._execute_capability(request.capability, request.parameters)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update metrics
            await self._update_metrics(processing_time, success=True)
            
            return AgentResponse(
                request_id=request.request_id,
                agent_id=self.agent_id,
                status='success',
                result=result,
                processing_time=processing_time,
                metadata={'security_level': request.security_level}
            )
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            await self._update_metrics(processing_time, success=False)
            
            logger.error(f"Agent {self.agent_id} request processing failed: {str(e)}")
            
            return AgentResponse(
                request_id=request.request_id,
                agent_id=self.agent_id,
                status='error',
                result={},
                processing_time=processing_time,
                errors=[str(e)]
            )
        finally:
            self.status = AgentStatus.READY
    
    async def _execute_capability(self, capability: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the requested capability with standardized validation and security"""
        if capability not in getattr(self, 'handlers', {}):
            raise ValueError(f"Capability '{capability}' not supported")
            
        # Validate and parse input using Pydantic contracts
        contracts = getattr(self, 'contracts', {})
        if capability in contracts:
            input_model, output_model = contracts[capability]
            validated_input = input_model(**parameters)
            
            # Execute the capability handler
            result = await self.handlers[capability](validated_input)
            
            # Ensure result matches output contract
            if isinstance(result, dict):
                validated_output = output_model(**result)
                return validated_output.dict()
            return result
        else:
            # Fallback for agents without contracts
            return await self.handlers[capability](parameters)
    
    def _validate_capability(self, capability: str) -> bool:
        """Validate if the requested capability is supported"""
        return capability in [cap.name for cap in self.capabilities]
    
    async def _detect_pii(self, data: Dict[str, Any]) -> bool:
        """Detect personally identifiable information in request data"""
        # Simplified PII detection - in production, use specialized libraries
        pii_patterns = ['ssn', 'social_security', 'credit_card', 'phone', 'email', 'address']
        data_str = json.dumps(data).lower()
        return any(pattern in data_str for pattern in pii_patterns)
    
    async def _check_rate_limits(self, request_id: str) -> bool:
        """Check rate limiting constraints"""
        # Simplified rate limiting - in production, use Redis or similar
        return True  # Placeholder implementation
    
    async def _validate_data_size(self, data: Dict[str, Any]) -> bool:
        """Validate request data size"""
        max_size = 10 * 1024 * 1024  # 10MB limit
        data_size = len(json.dumps(data).encode('utf-8'))
        return data_size <= max_size
    
    async def _update_metrics(self, processing_time: float, success: bool) -> None:
        """Update agent performance metrics"""
        self.metrics.requests_processed += 1
        
        # Update average processing time
        total_time = self.metrics.average_processing_time * (self.metrics.requests_processed - 1)
        self.metrics.average_processing_time = (total_time + processing_time) / self.metrics.requests_processed
        
        # Update success/error rates
        if success:
            success_count = self.metrics.success_rate * (self.metrics.requests_processed - 1) / 100 + 1
        else:
            success_count = self.metrics.success_rate * (self.metrics.requests_processed - 1) / 100
            
        self.metrics.success_rate = (success_count / self.metrics.requests_processed) * 100
        self.metrics.error_rate = 100 - self.metrics.success_rate
        self.metrics.last_updated = datetime.utcnow()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform agent health check"""
        return {
            'agent_id': self.agent_id,
            'version': self.version,
            'status': self.status.value,
            'uptime': self._calculate_uptime(),
            'metrics': self.metrics.dict(),
            'capabilities_count': len(self.capabilities),
            'last_health_check': datetime.utcnow().isoformat()
        }
    
    def _calculate_uptime(self) -> float:
        """Calculate agent uptime percentage"""
        # Simplified uptime calculation
        return 99.9  # Placeholder
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        return {
            'agent_summary': {
                'id': self.agent_id,
                'version': self.version,
                'status': self.status.value,
                'capabilities': len(self.capabilities)
            },
            'performance_metrics': self.metrics.dict(),
            'security_configuration': {
                'max_security_level': self.security_config.get('max_security_level'),
                'security_features_enabled': len([k for k, v in self.security_config.items() if v is True])
            },
            'operational_status': {
                'concurrent_capacity': self.operational_config.get('max_concurrent_requests'),
                'timeout_settings': self.operational_config.get('request_timeout'),
                'monitoring_enabled': self.operational_config.get('performance_monitoring')
            },
            'report_generated': datetime.utcnow().isoformat()
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get comprehensive agent information"""
        return {
            'agent_id': self.agent_id,
            'version': self.version,
            'status': self.status.value,
            'capabilities': [cap.dict() for cap in self.capabilities],
            'security_level': self.security_config.get('max_security_level'),
            'operational_config': self.operational_config,
            'metrics_summary': {
                'requests_processed': self.metrics.requests_processed,
                'success_rate': self.metrics.success_rate,
                'average_processing_time': self.metrics.average_processing_time
            }
        }