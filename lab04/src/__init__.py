"""

Lab 04: Data Engineering, Validation, Leakage-Free Preprocessing,

Feature Engineering, and Model Auditing for UCI Adult Dataset.

"""



import sys



__version__ = "1.0.0"



# Register module aliases if loaded as lab04.src so unpickling resolves seamlessly

if "lab04.src" in sys.modules and "src" not in sys.modules:

    sys.modules["src"] = sys.modules["lab04.src"]
