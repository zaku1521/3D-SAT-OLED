import sys
from importlib.metadata import version, PackageNotFoundError

# List all imported packages in your project (organized based on your code)
packages = [
    "ast",       # Built-in module, no version
    "os",        # Built-in module, no version
    "joblib",
    "json",      # Built-in module, no version
    "numpy",
    "pandas",
    "collections",  # Built-in module, no version
    "addict",
    "tqdm",
    "logging",   # Built-in module, no version
    "sklearn",   # Alias for scikit-learn
    "torch",
    "torch.nn",  # Submodule of torch, version same as torch
    "argparse",   # Built-in module, no version
    "pyyaml",
    "scipy",
    "copy",
    "matplotlib"
]

print("Project Dependent Packages Version Information:")
print("=" * 40)
for pkg in packages:
    # Skip built-in modules (no version number)
    if pkg in ["ast", "os", "json", "collections", "logging", "argparse", "torch.nn", "copy"]:
        continue
    # Handle aliases (e.g., sklearn's actual package name is scikit-learn)
    actual_pkg = "scikit-learn" if pkg == "sklearn" else pkg
    try:
        pkg_version = version(actual_pkg)
        print(f"{pkg:<10} Version: {pkg_version}")
    except PackageNotFoundError:
        print(f"{pkg:<10} Not installed or incorrect name")
print("=" * 40)
print("Note: Built-in Python modules (such as os, json) have no independent versions; their versions depend on Python itself")