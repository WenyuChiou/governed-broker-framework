"""
Test Validator Factory

Tests for dynamic validator loading from configuration.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from validators.factory import (
    VALIDATOR_REGISTRY,
    get_validator_class,
    create_validator,
    create_validators_from_config,
    get_default_validators,
    list_available_validators
)
from validators.skill_validators import SkillValidator


class TestValidatorRegistry:
    """Test validator registry."""
    
    def test_registry_has_core_validators(self):
        """Ensure all core validators are registered."""
        core_validators = [
            "SkillAdmissibilityValidator",
            "ContextFeasibilityValidator",
            "InstitutionalConstraintValidator",
            "EffectSafetyValidator",
            "PMTConsistencyValidator",
            "UncertaintyValidator"
        ]
        for name in core_validators:
            assert name in VALIDATOR_REGISTRY, f"Missing: {name}"
    
    def test_registry_has_aliases(self):
        """Test short name aliases."""
        assert "Admissibility" in VALIDATOR_REGISTRY
        assert "PMT" in VALIDATOR_REGISTRY
        assert "Feasibility" in VALIDATOR_REGISTRY


class TestValidatorFactory:
    """Test validator factory functions."""
    
    def test_get_validator_class(self):
        """Test getting validator class by name."""
        cls = get_validator_class("PMTConsistencyValidator")
        assert issubclass(cls, SkillValidator)
    
    def test_get_validator_class_alias(self):
        """Test getting validator class by alias."""
        cls = get_validator_class("PMT")
        assert issubclass(cls, SkillValidator)
    
    def test_get_validator_class_not_found(self):
        """Test error for unknown validator."""
        with pytest.raises(KeyError):
            get_validator_class("NonexistentValidator")
    
    def test_create_validator(self):
        """Test validator instantiation."""
        validator = create_validator("SkillAdmissibilityValidator")
        assert isinstance(validator, SkillValidator)
        assert validator.name == "SkillAdmissibilityValidator"
    
    def test_create_validators_from_config(self):
        """Test creating multiple validators from list."""
        names = ["Admissibility", "PMT", "EffectSafety"]
        validators = create_validators_from_config(names)
        assert len(validators) == 3
        assert all(isinstance(v, SkillValidator) for v in validators)
    
    def test_create_validators_skip_missing(self):
        """Test skipping unknown validators."""
        names = ["Admissibility", "UnknownValidator", "PMT"]
        validators = create_validators_from_config(names, skip_missing=True)
        assert len(validators) == 2
    
    def test_create_validators_raise_on_missing(self):
        """Test error when skip_missing=False."""
        names = ["Admissibility", "UnknownValidator"]
        with pytest.raises(KeyError):
            create_validators_from_config(names, skip_missing=False)


class TestDefaultValidators:
    """Test default validator set."""
    
    def test_get_default_validators(self):
        """Test getting default validators."""
        validators = get_default_validators()
        assert len(validators) == 6
        assert all(isinstance(v, SkillValidator) for v in validators)
    
    def test_list_available_validators(self):
        """Test listing available validators."""
        available = list_available_validators()
        assert len(available) > 10  # Core + aliases + legacy
        assert "PMTConsistencyValidator" in available


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
