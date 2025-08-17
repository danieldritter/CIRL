# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class BaseFactorParser(ABC):
    """Base class for factor parsers."""

    @abstractmethod
    def parse(self, text: str) -> Tuple[List[str], float]:
        """
        Parse factors from text.

        Returns:
            Tuple of (factors, format_score) where format_score is 0.0-1.0
        """
        pass


class ListFactorParser(BaseFactorParser):
    """Parse factors from list format."""

    def __init__(self, max_factors: int = 10):
        self.max_factors = max_factors

    def parse(self, text: str) -> Tuple[List[str], float]:
        """
        Parse factors from list format.
        Expected format: ["factor1", "factor2", "factor3"]
        """
        # Try to find list array in the text
        list_match = re.search(r"\[.*?\]", text, re.DOTALL)
        if list_match:
            list_str = list_match.group()
            parsed = json.loads(list_str)

            if isinstance(parsed, list):
                factors = [str(factor).strip() for factor in parsed if factor]
                factors = [self._clean_factor(f) for f in factors if f.strip()]

                # Calculate format score based on list validity and structure
                format_score = 1.0  # Perfect list format
                if len(factors) == 0:
                    format_score = 0.5  # Valid list but empty
                elif len(factors) > self.max_factors:
                    format_score = 0.8  # Valid but too many factors

                return factors[: self.max_factors], format_score

        # Check if text looks like it was attempting list format
        if "[" in text and "]" in text:
            return [], 0.3  # Attempted list but failed

        return [], 0.0

    def _clean_factor(self, factor: str) -> str:
        """Clean factor text."""
        factor = factor.strip().strip("\"'")
        factor = re.sub(r"^\d+\.\s*", "", factor)  # Remove numbering
        return factor


class TextFactorParser:
    """
    Factor parser that uses a single specified format.
    Provides format scoring to encourage proper formatting.
    """

    def __init__(self, parser_type: str = "structured_json", max_factors: int = 15, validation_enabled: bool = True):
        """
        Initialize factor parser.

        Args:
            parser_type: Parser type ("json", "list", "structured")
            max_factors: Maximum number of factors to extract
            validation_enabled: Whether to validate extracted factors
        """
        self.parser_type = parser_type
        self.max_factors = max_factors
        self.validation_enabled = validation_enabled

        # Initialize the specified parser
        self.parsers = {
            "list": ListFactorParser(max_factors),
        }

        if parser_type not in self.parsers:
            raise ValueError(f"Unknown parser type: {parser_type}. Available: {list(self.parsers.keys())}")

        self.parser = self.parsers[parser_type]

    def parse(self, text: str) -> List[str]:
        """
        Parse factors from text using configured parser.

        Args:
            text: Text containing factor information

        Returns:
            List of extracted factor strings
        """
        if not text or not text.strip():
            return []

        factors, format_score = self.parser.parse(text)

        # Store format score for potential use in reward calculation
        self.last_format_score = format_score

        return factors

    def parse_with_format_score(self, text: str) -> Tuple[List[str], float]:
        """
        Parse factors and return format score.

        Args:
            text: Text containing factor information

        Returns:
            Tuple of (factors, format_score)
        """
        if not text or not text.strip():
            return [], 0.0

        factors, format_score = self.parser.parse(text)
        return factors, format_score

    def get_format_score(self) -> float:
        """Get the format score from the last parse operation."""
        return getattr(self, "last_format_score", 0.0)

    def get_parser_stats(self) -> Dict:
        """Get statistics about parser configuration."""
        return {
            "parser_type": self.parser_type,
            "max_factors": self.max_factors,
            "validation_enabled": self.validation_enabled,
            "available_parsers": list(self.parsers.keys()),
        }
