import os
import sys

# Make the repository root importable so tests can import the tracker modules
# whether pytest is invoked from the root or from inside tests/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
