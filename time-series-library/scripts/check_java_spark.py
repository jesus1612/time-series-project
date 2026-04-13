#!/usr/bin/env python3
"""
Verify Java 17+ for PySpark. Used by Makefile check-java and lib_java_spark.sh.
Parses `java -version` reliably (handles 1.8.x vs 17+ style version strings).
"""
from __future__ import annotations

import re
import subprocess
import sys


def main() -> int:
    try:
        proc = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        print("✗ Java not found. PySpark 4.0.1 requires Java 17+")
        print("  Install Java 17+ using: make install-java-macos (or your OS package manager)")
        return 1
    except subprocess.TimeoutExpired:
        print("✗ `java -version` timed out (unexpected). Check JAVA_HOME / PATH.")
        return 1

    combined = (proc.stderr or "") + (proc.stdout or "")
    m = re.search(r'version "(\d+)\.(\d+)', combined)
    if not m:
        print("✗ Could not parse Java version from `java -version`. Output (first lines):")
        for line in combined.splitlines()[:5]:
            print(f"    {line}")
        print("  Install a JDK 17+ and ensure `java` points to it (see README / make install-java-macos).")
        return 1

    a, b = int(m.group(1)), int(m.group(2))
    major = b if a == 1 else a

    if major >= 17:
        print(f"✓ Java {major} is compatible with PySpark 4.0.1")
        return 0

    print(f"✗ Java (logical major {major}) is not compatible. PySpark 4.0.1 requires Java 17+")
    print("  Install Java 17+ using: make install-java-macos")
    return 1


if __name__ == "__main__":
    sys.exit(main())
