"""Task definitions and configurations for prompt optimization."""

from .multiclass_email3 import EmailClassificationTask3
from .multiclass_email10 import EmailClassificationTask10
from .multiclass_health3 import HealthClassificationTask3
from .multiclass_health10 import HealthClassificationTask10
from .tooluse_finance import FinanceToolUseTask
from .tooluse_ecommerce import EcommerceToolUseTask
from .extract_code import CodeExtractionTask
from .extract_legal import LegalExtractionTask

__all__ = [
    "EmailClassificationTask3",
    "EmailClassificationTask10",
    "HealthClassificationTask3",
    "HealthClassificationTask10",
    "FinanceToolUseTask",
    "EcommerceToolUseTask",
    "CodeExtractionTask",
    "LegalExtractionTask",
]
