"""
Instrument system for diverse instrument support.

This module provides comprehensive instrument handling including:
- Modern synths (808, FM, analog, digital)
- Ethnic percussion instruments
- Wind instruments (brass, woodwind, reed)
- Layered and hybrid instrument configurations
- Preset management system
- Instrument-specific parameter handling
"""

from .instrument_categories import (
    InstrumentCategory,
    InstrumentSubcategory,
    InstrumentPreset,
    InstrumentLayer,
    InstrumentRegistry,
    instrument_registry,
    get_instrument_categories,
    get_preset_config
)

__all__ = [
    'InstrumentCategory',
    'InstrumentSubcategory',
    'InstrumentPreset',
    'InstrumentLayer',
    'InstrumentRegistry',
    'instrument_registry',
    'get_instrument_categories',
    'get_preset_config'
]