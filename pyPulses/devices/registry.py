"""
Hardware Registry - Tracks physical instruments by logical identifier.

Instruments are keyed by a stable logical ID (e.g., "dcbox_main").
Each device class manages its own connection parameters internally.

Supports both registered classes (participate in serialization) and
unregistered classes (tracked but not serialized).
"""

from __future__ import annotations
from typing import Dict, Type, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from .abstract_device import abstractDevice

T = TypeVar('T')

# Global mapping: class_tag -> class type
# Populated by @register_hardware_class decorator
_HARDWARE_CLASS_REGISTRY: Dict[str, Type] = {}
_ID_COUNTERS: Dict[str, int] = {}  # For auto-generating IDs

def register_hardware_class(tag: str):
    """
    Decorator that registers a hardware class with a string tag for serialization.

    Usage:
        @register_hardware_class("ad5781")
        class ad5791(pyvisaDevice):
            ...

    This enables reconstruction from JSON.
    """

    def decorator(cls: Type[T]) -> Type[T]:
        if tag in _HARDWARE_CLASS_REGISTRY:
            raise ValueError(
                f"Hardware class tag '{tag}' already registered to {_HARDWARE_CLASS_REGISTRY[tag]}"
            )
        _HARDWARE_CLASS_REGISTRY[tag] = cls
        cls._registry_class_tag_ = tag
        return cls
    
    return decorator

def get_hardware_class(tag: str) -> Type | None:
    """Look up a hardware class by its registration tag."""
    return _HARDWARE_CLASS_REGISTRY.get(tag)

def list_hardware_classes() -> Dict[str, Type]:
    """Return a copy of all registered hardware classes."""
    return _HARDWARE_CLASS_REGISTRY.copy()

class HardwareRegistry():
    """
    Singleton registry of physical instruments.

    Keyed by stable logical identifiers (e.g., "dcbox_main", "lockin_1").
    
    - Registered classes (via @register_hardware_class) participate in 
      serialization/deserialization.
    - Unregistered classes are still tracked to prevent duplicate connections,
      but are skipped during serialization.
    """

    _instance: HardwareRegistry | None = None
    _devices: Dict[str, abstractDevice] = {}

    def __new__(cls) -> HardwareRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._devices = {}
        return cls._instance
    
    @classmethod
    def _generate_id(cls, device: abstractDevice) -> str:
        """
        Generate a unique ID for a device.
        Uses the class's registry tag if available, otherwise falls back to the
        class name. Increments counter until finding an unused ID.
        """

        device_class = type(device)
        base = getattr(device_class, '_registry_class_tag_', None) or device_class.__name__
        
        count = _ID_COUNTERS.get(base, 0)
        while True:
            candidate = f"{base}_{count}"
            if candidate not in cls._devices:
                _ID_COUNTERS[base] = count + 1
                return candidate
            count += 1

    @classmethod
    def _validate_id(cls, registry_id: str, device: abstractDevice) -> str:
        """
        Validate a requested ID, generating a new one if it's taken.
        
        If the requested ID is already in use, appends the class base name
        and a counter to create a unique variant.
        """
        if registry_id not in cls._devices:
            return registry_id
        
        # Requested ID is taken - generate a unique variant
        device_class = type(device)
        base = getattr(device_class, '_registry_class_tag_', None) or device_class.__name__
        
        # Try appending class info to distinguish
        count = 0
        while True:
            candidate = f"{registry_id}_{base}_{count}"
            if candidate not in cls._devices:
                return candidate
            count += 1

    @classmethod
    def get(cls, registry_id: str) -> abstractDevice | None:
        """
        Get a hardware device by resource ID.

        Parameters
        ----------
        registry_id : str
            Unique identifier for the physical instrument.
        
        Returns
        -------
        device : abstractDevice or None
        """

        return cls._devices.get(registry_id)
    
    @classmethod
    def register(
        cls, 
        device: abstractDevice, 
        registry_id: str | None = None,
    ) -> str:
        """
        Register a hardware device.

        Parameters
        ----------
        device : abstractDevice
            The device instance.
        registry_id : str, optional
            Logical identifier. If None, auto-generates from class tag/name.

        Returns
        -------
        registry_id : str
            The ID under which the device was registered (may differ from
            requested ID if allow_duplicate_id=True).
        """

        if registry_id is None:
            registry_id = cls._generate_id(device)
        elif registry_id in cls._devices:
            existing = cls._devices[registry_id]
            raise ValueError(
                f"Registry ID '{registry_id}' already registered to "
                f"{type(existing).__name__}."
            )

        device._hardware_registry_id_ = registry_id
        cls._devices[registry_id] = device
        return registry_id

    @classmethod
    def unregister(cls, registry_id: str) -> None:
        """
        Unregister a hardware device.

        Parameters
        ----------
        registry_id : str
            Unique identifier for the physical instrument.
        """

        if registry_id in cls._devices:
            device = cls._devices[registry_id]
            if hasattr(device, '_hardware_registry_id_'):
                device._hardware_registry_id_ = None
            del cls._devices[registry_id]

    @classmethod
    def all(cls) -> Dict[str, abstractDevice]:
        """Return a copy of all registered hardware."""
        return cls._devices.copy()

    @classmethod
    def clear(cls) -> None:
        """Unregister all devices."""
        for registry_id in list(cls._devices.keys()):
            cls.unregister(registry_id)
        cls._devices.clear()
        _ID_COUNTERS.clear()

    @classmethod
    def serialize_all(cls) -> Dict[str, dict]:
        """
        Serialize all registered hardware.
        
        Only includes devices whose class was registered with
        @register_hardware_class. Unregistered classes are skipped.
        """
        result = {}

        for registry_id, device in cls._devices.items():
            class_tag = getattr(device, '_registry_class_tag_', None)
            if class_tag is None:
                # Unregistered class - skip serialization
                continue

            try:
                config = device._serialize_state()
            except Exception:
                config = {}

            result[registry_id] = {
                'class': class_tag,
                'config': config,
            }

        return result

    @classmethod
    def deserialize_all(cls, data: Dict[str, dict]) -> Dict[str, abstractDevice]:
        """
        Reconstruct hardware from serialized data.
        
        For each entry:
        - If device already exists at that registry_id, calls _deserialize_state
        to restore settings.
        - If device doesn't exist, creates it via from_config.
        
        Parameters
        ----------
        data : dict
            Output from serialize_all().
        
        Returns
        -------
        dict
            Mapping of registry_id -> device instance.
        """
        devices = {}

        for registry_id, entry in data.items():
            class_tag = entry.get('class')
            config = entry.get('config', {}).copy()

            existing = cls.get(registry_id)
            
            if existing is not None:
                # Device exists - just restore state
                if hasattr(existing, '_deserialize_state'):
                    try:
                        existing._deserialize_state(config)
                    except Exception as e:
                        # TODO Log warning but continue
                        pass
                devices[registry_id] = existing
                continue

            # Device doesn't exist - create it
            device_class = get_hardware_class(class_tag)
            if device_class is None:
                raise ValueError(f"Unknown hardware class '{class_tag}'")

            config['registry_id'] = registry_id
            device = device_class.from_config(config)
            devices[registry_id] = device

        return devices

"""
Device Registry - Tracks logical/abstract devices by identifier.

Logical devices are abstractions built on top of hardware:
- CalibratedChannel wrapping a DAC channel
- PulsePair combining two signal generators

Devices reference each other by registry ID, resolved at access time.
This avoids dependency ordering issues during deserialization.
"""

from __future__ import annotations
from typing import Dict, Type, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from .abstract_device import abstractDevice

T = TypeVar('T')

_DEVICE_CLASS_REGISTRY: Dict[str, Type] = {}
_ID_COUNTERS: Dict[str, int] = {}


def register_device_class(tag: str):
    """Decorator that registers a device class with a string tag."""
    def decorator(cls: Type[T]) -> Type[T]:
        if tag in _DEVICE_CLASS_REGISTRY:
            raise ValueError(
                f"Device class tag '{tag}' already registered to "
                f"{_DEVICE_CLASS_REGISTRY[tag]}"
            )
        _DEVICE_CLASS_REGISTRY[tag] = cls
        cls._registry_class_tag_ = tag
        return cls
    return decorator


def get_device_class(tag: str) -> Type | None:
    """Look up a device class by its registration tag."""
    return _DEVICE_CLASS_REGISTRY.get(tag)


def list_device_classes() -> Dict[str, Type]:
    """Return a copy of all registered device classes."""
    return _DEVICE_CLASS_REGISTRY.copy()


class DeviceRegistry:
    """
    Singleton registry of logical devices.

    Keyed by stable logical identifiers. Devices reference each other
    by registry ID (not direct pointers), resolved at access time.
    """

    _instance: DeviceRegistry | None = None
    _devices: Dict[str, abstractDevice] = {}

    def __new__(cls) -> DeviceRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._devices = {}
        return cls._instance

    @classmethod
    def _generate_id(cls, device: abstractDevice) -> str:
        """Generate a unique ID for a device."""
        device_class = type(device)
        base = getattr(device_class, '_registry_class_tag_', None) or device_class.__name__
        
        count = _ID_COUNTERS.get(base, 0)
        while True:
            candidate = f"{base}_{count}"
            if candidate not in cls._devices:
                _ID_COUNTERS[base] = count + 1
                return candidate
            count += 1

    @classmethod
    def get(cls, registry_id: str) -> abstractDevice | None:
        """Get a device by its registry ID."""
        return cls._devices.get(registry_id)

    @classmethod
    def register(cls, device: abstractDevice, registry_id: str | None = None) -> str:
        """
        Register a logical device.

        Parameters
        ----------
        device : abstractDevice
            The device instance.
        registry_id : str, optional
            Logical identifier. If None, auto-generates.

        Returns
        -------
        registry_id : str
            The ID under which the device was registered.
        """
        if registry_id is None:
            registry_id = cls._generate_id(device)
        elif registry_id in cls._devices:
            existing = cls._devices[registry_id]
            raise ValueError(
                f"Registry ID '{registry_id}' already registered to "
                f"{type(existing).__name__}."
            )

        device._device_registry_id_ = registry_id
        cls._devices[registry_id] = device
        return registry_id

    @classmethod
    def unregister(cls, registry_id: str) -> None:
        """Unregister a device."""
        if registry_id in cls._devices:
            device = cls._devices[registry_id]
            if hasattr(device, '_device_registry_id_'):
                device._device_registry_id_ = None
            del cls._devices[registry_id]

    @classmethod
    def all(cls) -> Dict[str, abstractDevice]:
        """Return a copy of all registered devices."""
        return cls._devices.copy()

    @classmethod
    def clear(cls) -> None:
        """Unregister all devices."""
        for registry_id in list(cls._devices.keys()):
            cls.unregister(registry_id)
        cls._devices.clear()
        _ID_COUNTERS.clear()

    @classmethod
    def serialize_all(cls) -> Dict[str, dict]:
        """Serialize all registered devices."""
        result = {}

        for registry_id, device in cls._devices.items():
            class_tag = getattr(device, '_registry_class_tag_', None)
            if class_tag is None:
                continue

            try:
                config = device._serialize_state()
            except Exception:
                config = {}

            result[registry_id] = {
                'class': class_tag,
                'config': config,
            }

        return result

    @classmethod
    def deserialize_all(cls, data: Dict[str, dict]) -> Dict[str, abstractDevice]:
        """
        Reconstruct devices from serialized data.
        
        Uses two passes:
        1. Create all devices (or retrieve existing)
        2. Call _deserialize_state on all of them
        
        This ensures all devices exist before any state restoration that
        might reference other devices.
        """
        devices = {}
        configs = {}

        # Pass 1: Create or retrieve all devices
        for registry_id, entry in data.items():
            class_tag = entry.get('class')
            config = entry.get('config', {}).copy()
            configs[registry_id] = config  # Save for pass 2

            existing = cls.get(registry_id)
            if existing is not None:
                devices[registry_id] = existing
                continue

            device_class = get_device_class(class_tag)
            if device_class is None:
                raise ValueError(f"Unknown device class '{class_tag}'")

            config_for_creation = config.copy()
            config_for_creation['registry_id'] = registry_id
            device = device_class.from_config(config_for_creation)
            devices[registry_id] = device

        # Pass 2: Restore state on all devices
        for registry_id, device in devices.items():
            config = configs[registry_id]
            if hasattr(device, '_deserialize_state'):
                try:
                    device._deserialize_state(config)
                except Exception:
                    # TODO: log warning
                    pass

        return devices
    
"""
Rack State - Save and load the entire system state.

Coordinates serialization of both HardwareRegistry and DeviceRegistry,
ensuring proper ordering (hardware must exist before devices that reference it).
"""

import json
from pathlib import Path
from typing import Dict, Any

class RackState:
    """
    Manages serialization and deserialization of the complete rack state.
    
    Example
    -------
    # Save current state
    RackState.save("rack_config.json")
    
    # Later, restore state
    RackState.load("rack_config.json")
    
    # Or work with dicts directly
    state = RackState.serialize()
    RackState.deserialize(state)
    """
    
    @classmethod
    def serialize(cls) -> Dict[str, Any]:
        """
        Serialize entire rack state to a dictionary.
        
        Returns
        -------
        dict
            Complete rack state with 'hardware' and 'devices' sections.
        """
        return {
            'version': 1,
            'hardware': HardwareRegistry.serialize_all(),
            'devices': DeviceRegistry.serialize_all(),
        }
    
    @classmethod
    def deserialize(cls, state: Dict[str, Any]) -> None:
        """
        Restore rack state from a dictionary.
        
        Hardware is deserialized first, then devices (which may reference hardware).
        For existing entries, _deserialize_state is called to restore settings.
        
        Parameters
        ----------
        state : dict
            Output from serialize().
        """
        version = state.get('version', 1)
        if version != 1:
            raise ValueError(f"Unknown rack state version: {version}")
        
        # Hardware first (devices depend on hardware)
        hardware_state = state.get('hardware', {})
        HardwareRegistry.deserialize_all(hardware_state)
        
        # Then devices
        device_state = state.get('devices', {})
        DeviceRegistry.deserialize_all(device_state)
    
    @classmethod
    def save(cls, path: str | Path, indent: int = 2) -> None:
        """
        Save rack state to a JSON file.
        
        Parameters
        ----------
        path : str or Path
            Output file path.
        indent : int, default=2
            JSON indentation level.
        """
        state = cls.serialize()
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=indent)
    
    @classmethod
    def load(cls, path: str | Path) -> None:
        """
        Load rack state from a JSON file.
        
        Parameters
        ----------
        path : str or Path
            Input file path.
        """
        with open(path, 'r') as f:
            state = json.load(f)
        
        cls.deserialize(state)

"""
Handling References to Registered Objects
"""

def resolve_reference(ref: str):
    """
    Resolve a reference string to a device or sub-object (e.g., channel).
    
    Format: "registry_id" or "registry_id->accessor"
    
    The accessor is passed to the device's `resolve(accessor)` method if it exists
    
    Checks HardwareRegistry first, then DeviceRegistry.
    
    Examples:
        "ad5791_0->ch3"         -> ad5791_0.resolve("ch3")
        "dtg5274_0->1A1"        -> dtg5274_0.resolve("1A1")
        "sr865a_0"              -> lockin_1 (device itself)
    
    Parameters
    ----------
    ref : str
        Reference string.
    
    Returns
    -------
    object or None
        The resolved device or sub-object or None if the reference could not be 
        resolved.
    
    """

    if '->' in ref:
        registry_id, accessor = ref.split('->', 1)
        
        device = HardwareRegistry.get(registry_id) or DeviceRegistry.get(registry_id)
        if device is None:
            return None
        
        # Try device's own resolve method first
        if hasattr(device, 'resolve'):
            return device.resolve(accessor)
        
        # Fall back to getChannel for string accessors
        if hasattr(device, 'getChannel'):
            return device.getChannel(accessor)
        
        # Fall back to channel(int) for "chN" format
        if accessor.startswith('ch') and hasattr(device, 'channel'):
            try:
                channel_index = int(accessor[2:])
                return device.channel(channel_index)
            except ValueError:
                pass
        
        return None
    
    else:
        device = HardwareRegistry.get(ref) or DeviceRegistry.get(ref)
        return device

def format_reference(obj) -> str | None:
    """
    Format an object back to a reference string.
    
    The object must either:
    - Be registered directly (has _hardware_registry_id_ or _device_registry_id_)
    - Have a `format_ref()` method that returns (parent, accessor_string)
    
    Parameters
    ----------
    obj : object
        The object to format.
    
    Returns
    -------
    str
        Reference string like "ad5791_0->ch3" or "dtg5274_0->1A1".
    """

    if obj is None:
        return None

    # Check if object knows how to format itself
    if hasattr(obj, 'format_ref'):
        parent, accessor = obj.format_ref()
        parent_id = _get_registry_id(parent)
        return f"{parent_id}->{accessor}"
    
    # Must be a registered device itself
    device_id = _get_registry_id(obj)
    return device_id


def _get_registry_id(obj) -> str:
    """Get the registry ID of an object, or raise ValueError."""
    device_id = (
        getattr(obj, '_hardware_registry_id_', None) or
        getattr(obj, '_device_registry_id_', None)
    )
    if device_id is None:
        raise ValueError(f"Object {type(obj).__name__} is not registered")
    return device_id

class DeferredReference:
    """
    Placeholder for a reference that must be explicitly resolved.
    
    Any attribute access before resolution raises an error, making it
    obvious when you've forgotten to unwrap.
    """
    
    def __init__(self, ref: str):
        object.__setattr__(self, '_ref', ref)
        object.__setattr__(self, '_resolved', None)
    
    def unwrap(self):
        """Resolve and return the referenced object."""
        if self._resolved is None:
            object.__setattr__(self, '_resolved', resolve_reference(self._ref))
        return self._resolved
    
    @property
    def ref(self) -> str:
        """The reference string."""
        return self._ref
    
    def __repr__(self):
        return f"DeferredReference({self._ref!r})"
    
    def __getattr__(self, name):
        raise RuntimeError(
            f"Attempted to access '{name}' on unresolved DeferredReference('{self._ref}'). "
            f"Did you forget to call unwrap() in _deserialize_state()?"
        )
    
    def __setattr__(self, name, value):
        raise RuntimeError(
            f"Attempted to set '{name}' on unresolved DeferredReference('{self._ref}'). "
            f"Did you forget to call unwrap() in _deserialize_state()?"
        )