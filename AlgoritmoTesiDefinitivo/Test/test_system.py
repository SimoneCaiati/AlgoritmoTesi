import sys
import subprocess

def test_main_execution():
    result = subprocess.run([sys.executable, "main.py"], capture_output=True, text=True)
    assert result.returncode == 0