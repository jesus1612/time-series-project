#!/usr/bin/env bash
# Run pytest with JAVA_HOME/PATH set when a JDK 17+ is discoverable (same logic as make_java.sh).
# If no JDK is found, still runs pytest so unit tests pass and Spark cases skip cleanly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x "$ROOT/venv/bin/pytest" ]]; then
	echo "✗ No se encontró venv/bin/pytest. Ejecuta: make prepare-test   (o make test)"
	exit 1
fi
# shellcheck source=lib_java_spark.sh
source "$ROOT/scripts/lib_java_spark.sh"

set +e
resolve_java_spark
_java_rc=$?
set -e

if [ "$_java_rc" -ne 0 ]; then
	echo "⚠ Sin Java 17+ usable: los tests que arrancan Spark se omitirán. Usa make install-java-macos, instala openjdk@17, o exporta JAVA_HOME."
fi

exec "$ROOT/venv/bin/pytest" "$@"
