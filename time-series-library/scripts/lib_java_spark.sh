#!/usr/bin/env bash
# Shared Java 17+ resolution for Spark (sourced by make_java.sh and run_pytest_with_java.sh).
# Tries PATH first, then typical Homebrew (macOS) and Linux distro paths.

resolve_java_spark() {
	local root
	root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
	cd "$root" || return 1

	local py="python3"
	if [[ -x "$root/venv/bin/python" ]]; then
		py="$root/venv/bin/python"
	fi

	if "$py" "$root/scripts/check_java_spark.py" 2>/dev/null; then
		return 0
	fi

	case "$(uname -s)" in
	Darwin)
		for j in /opt/homebrew/opt/openjdk@17 /opt/homebrew/opt/openjdk@21 \
			/usr/local/opt/openjdk@17 /usr/local/opt/openjdk@21; do
			if [[ -x "$j/bin/java" ]]; then
				export PATH="$j/bin:$PATH"
				export JAVA_HOME="$j"
				echo "→ Usando JDK en $j (no estaba en el PATH de esta sesión)."
				echo "  Para que quede fijo en nuevas terminales, añade a ~/.zshrc o ~/.bashrc:"
				echo "    export PATH=\"$j/bin:\$PATH\""
				echo "    export JAVA_HOME=\"$j\""
				if "$py" "$root/scripts/check_java_spark.py"; then
					return 0
				fi
			fi
		done
		;;
	Linux)
		for j in /usr/lib/jvm/java-21-openjdk-amd64 /usr/lib/jvm/java-17-openjdk-amd64 \
			/usr/lib/jvm/java-21-openjdk /usr/lib/jvm/java-17-openjdk; do
			if [[ -x "$j/bin/java" ]]; then
				export PATH="$j/bin:$PATH"
				export JAVA_HOME="$j"
				echo "→ Usando JDK en $j"
				if "$py" "$root/scripts/check_java_spark.py"; then
					return 0
				fi
			fi
		done
		;;
	esac

	echo ""
	echo "No se encontró Java 17+ usable."
	echo "  macOS + Homebrew:  make install-java-macos"
	echo "  Luego:             make install-spark"
	echo "  Linux (ejemplo):   sudo apt install openjdk-17-jdk"
	echo ""
	return 1
}
