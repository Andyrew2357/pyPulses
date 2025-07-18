import inspect
import importlib
from pathlib import Path

PACKAGE = "pyPulses"
SUBMODULES = ["devices", "plotting", "routines", "utils"]
OUTPUT_ROOT = Path("docs-site/docs/reference")

MKDOCSTRINGS_OPTIONS = {
    "show_signature": True,
    "show_source": True,
    "group_by_category": True
}
FILTERS = ["!^_.*"]  # Hide private methods/functions/classes

# Base options for class documentation
CLASS_MKDOCSTRINGS_OPTIONS_BASE = {
    "show_signature": True,
    "show_source": True,
    "group_by_category": True,
    "members_order": "source",  # Keep source order for better control
    "show_if_no_docstring": False,
    "heading_level": 2,
    "inherited_members": True,
    "merge_init_into_class": True
}

CLASS_FILTERS = ["!^_(?!_init__).*"]  # Hide private methods except __init__

# Base classes to exclude from inheritance documentation
INHERITANCE_BLACKLIST = {
    "pyvisaDevice",
    "zhinstDevice",
    "abstractDevice", 
    "object",
    "IntEnum",
    "Enum",
    "ABC"
}

def get_allowed_members(cls_obj):
    """
    Get list of members that should be documented, excluding those from 
    blacklisted base classes
    """
    # Get the method resolution order (MRO) to find all base classes
    mro = inspect.getmro(cls_obj)
    
    # Find methods from blacklisted base classes
    blacklisted_methods = set()
    for base_class in mro[1:]:  # Skip the class itself (index 0)
        if base_class.__name__ in INHERITANCE_BLACKLIST:
            for name, member in inspect.getmembers(base_class):
                if (callable(member) and 
                    not name.startswith('_') and 
                    name != '__init__'):  # Keep __init__ always
                    blacklisted_methods.add(name)
    
    # Get all members from the class (including inherited)
    all_members = []
    for name, member in inspect.getmembers(cls_obj):
        if not name.startswith('_') or name == '__init__':
            # Check if this member should be excluded
            if name in blacklisted_methods:
                # Check if it's overridden in a non-blacklisted class
                is_overridden = False
                for base_class in mro:
                    if (base_class.__name__ not in INHERITANCE_BLACKLIST and 
                        hasattr(base_class, name) and 
                        name in base_class.__dict__):
                        is_overridden = True
                        break
                
                if not is_overridden:
                    continue  # Skip this member
            
            all_members.append(name)
    
    return all_members


def get_class_options_with_members(cls_obj):
    """Get class options with explicit members list"""
    options = CLASS_MKDOCSTRINGS_OPTIONS_BASE.copy()
    
    # Get allowed members
    allowed_members = get_allowed_members(cls_obj)
    
    # Add explicit members list
    options["members"] = allowed_members
    
    return options

def format_options_block(use_class_options=False, cls_obj=None):
    lines = []
    lines.append("    options:")
    
    if use_class_options and cls_obj:
        # Use class-specific options with explicit members list
        options = get_class_options_with_members(cls_obj)
        for k, v in options.items():
            if k == "members":
                # Handle members list specially
                lines.append(f"      {k}:")
                for member in v:
                    lines.append(f"        - {member}")
            else:
                lines.append(f"      {k}: {str(v).lower()}")
    else:
        # Use regular options
        options = CLASS_MKDOCSTRINGS_OPTIONS_BASE if use_class_options \
                                                    else MKDOCSTRINGS_OPTIONS
        for k, v in options.items():
            lines.append(f"      {k}: {str(v).lower()}")
        
        # Add filters for non-class options or when no explicit members
        lines.append("    filters:")
        filters = CLASS_FILTERS if use_class_options else FILTERS
        for f in filters:
            lines.append(f'      - "{f}"')
    
    return "\n".join(lines)


def write_class_doc(title, import_path, output_path, cls_obj):
    """
    Write documentation for a class with special handling for methods and 
    properties
    """
    with open(output_path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(f"::: {import_path}\n")
        f.write(format_options_block(use_class_options=True, cls_obj=cls_obj))
        
        # Check if class has properties with docstrings - only show documented
        documented_properties = []
        for name, obj in inspect.getmembers(cls_obj):
            if (isinstance(obj, property) and 
                not name.startswith('_') and 
                obj.__doc__ is not None and 
                obj.__doc__.strip()):  # properties with non-empty docstrings
                documented_properties.append(name)
        
        if documented_properties:
            f.write("\n\n## Properties\n\n")
            for prop_name in documented_properties:
                f.write(f"::: {import_path}.{prop_name}\n")
                f.write("    options:\n")
                f.write("      show_signature: false\n")
                f.write("      show_source: false\n")
                f.write("      heading_level: 4\n")
                f.write("\n")


def write_functions_doc(title, import_path, output_path, functions_dict):
    """Write documentation specifically for standalone functions"""
    with open(output_path, "w") as f:
        f.write(f"# {title}\n\n")
        
        # Document each function individually for better control
        for func_name, func_obj in functions_dict.items():
            f.write(f"## {func_name}\n\n")
            f.write(f"::: {func_obj.__module__}.{func_name}\n")
            f.write("    options:\n")
            f.write("      show_signature: true\n")
            f.write("      show_source: true\n")
            f.write("      heading_level: 3\n")
            f.write("    filters:\n")
            f.write('      - "!^_.*"\n')
            f.write("\n")


def write_doc(title, import_path, output_path):
    """Generic doc writer for backward compatibility"""
    with open(output_path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(f"::: {import_path}\n")
        f.write(format_options_block())


def main():
    full_nav = []

    for submodule in SUBMODULES:
        output_dir = OUTPUT_ROOT / submodule
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            mod = importlib.import_module(f"{PACKAGE}.{submodule}")
        except ImportError as e:
            print(f"❌ Could not import {PACKAGE}.{submodule}: {e}")
            continue

        classes = {
            name: obj
            for name, obj in inspect.getmembers(mod, inspect.isclass)
            if obj.__module__.startswith(f"{PACKAGE}.{submodule}")
            and not name.startswith("_")
        }

        functions = {
            name: obj
            for name, obj in inspect.getmembers(mod, inspect.isfunction)
            if obj.__module__.startswith(f"{PACKAGE}.{submodule}")
            and not name.startswith("_")
        }

        print(f"\n📚 {submodule.title()}:")
        sub_nav = []

        # Process classes with enhanced documentation
        for name, cls in classes.items():
            import_path = f"{cls.__module__}.{name}"
            md_path = output_dir / f"{name}.md"
            write_class_doc(name, import_path, md_path, cls)
            
            # Show inheritance info in console output
            mro = inspect.getmro(cls)
            if len(mro) > 1:  # Has base classes
                base_names = [base.__name__ for base in mro[1:] \
                              if base.__name__ != 'object']
                if base_names:
                    blacklisted_bases = [b for b in base_names if b in \
                                         INHERITANCE_BLACKLIST]
                    included_bases = [b for b in base_names if b not in \
                                      INHERITANCE_BLACKLIST]
                    
                    inheritance_info = []
                    if included_bases:
                        inheritance_info.append(f"inherits from: { \
                            ', '.join(included_bases)}")
                    if blacklisted_bases:
                        inheritance_info.append(f"excludes: { \
                            ', '.join(blacklisted_bases)}")
                    
                    if inheritance_info:
                        print(
                            f"  - Class: {name} ({' | '.join(inheritance_info)})"
                        )
                    else:
                        print(f"  - Class: {name}")
                        
                    # Show which methods are being excluded
                    allowed_members = get_allowed_members(cls)
                    all_methods = [n for n, obj in inspect.getmembers(cls) 
                                 if callable(obj) and not n.startswith('_')]
                    excluded_methods = set(all_methods) - set(allowed_members)
                    if excluded_methods:
                        print(
                            f"    └─ Excluding methods: { \
                                ', '.join(sorted(excluded_methods))}"
                        )
                else:
                    print(f"  - Class: {name}")
            else:
                print(f"  - Class: {name}")
            
            # Check for documented properties to mention in output
            documented_properties = [n for n, obj in inspect.getmembers(cls) 
                        if isinstance(obj, property) and not n.startswith('_')
                        and obj.__doc__ is not None and obj.__doc__.strip()]
            if documented_properties:
                print(f"    └─ Properties: {', '.join(documented_properties)}")
            
            sub_nav.append(f"      - {name}: reference/{submodule}/{name}.md")

        # Process functions
        if functions:
            md_path = output_dir / "functions.md"
            write_functions_doc("Functions", f"{PACKAGE}.{submodule}", \
                                md_path, functions)
            print(f"  - Functions: {', '.join(functions.keys())}")
            sub_nav.append(
                f"      - Functions: reference/{submodule}/functions.md"
            )

        if sub_nav:
            full_nav.append(f"    - {submodule.title()}:")
            full_nav.extend(sub_nav)

    # Print nav block
    print("\n🧾 Suggested `mkdocs.yml` nav block:\n")
    print("  - Reference:")
    for line in full_nav:
        print(line)

if __name__ == "__main__":
    main()
