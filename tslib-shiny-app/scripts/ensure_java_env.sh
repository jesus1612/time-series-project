#!/bin/sh
# Resolve JAVA_HOME / PATH for PySpark across macOS (Homebrew), Windows (Git Bash),
# and typical Linux OpenJDK layouts. POSIX sh for /bin/sh on macOS.

_set_java_from_home() {
	jh="$1"
	# trim trailing slash
	case "$jh" in
		*/) jh="${jh%/}" ;;
	esac
	if [ -x "$jh/bin/java" ]; then
		JAVA_HOME="$jh"
		export JAVA_HOME
		PATH="$jh/bin:$PATH"
		export PATH
		return 0
	fi
	return 1
}

_java_major_version() {
	java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1
}

_java_version_ok() {
	command -v java >/dev/null 2>&1 || return 1
	ver="$(_java_major_version)"
	case "$ver" in
		'' | *[!0-9]*) return 1 ;;
	esac
	[ "$ver" -ge 17 ] || return 1
	return 0
}

# Export JAVA_HOME when we find JDK 17+ at a known location, or keep PATH java if already 17+.
ensure_java_env() {
	if _java_version_ok; then
		return 0
	fi

	if [ -n "${JAVA_HOME:-}" ] && _set_java_from_home "$JAVA_HOME" && _java_version_ok; then
		return 0
	fi

	# macOS Homebrew
	for jh in /opt/homebrew/opt/openjdk@17 /usr/local/opt/openjdk@17; do
		if [ -d "$jh" ] && _set_java_from_home "$jh" && _java_version_ok; then
			return 0
		fi
	done

	uname_s="$(uname -s 2>/dev/null || echo unknown)"

	case "$uname_s" in
	Linux)
		for jh in \
			/usr/lib/jvm/java-21-openjdk-amd64 \
			/usr/lib/jvm/java-17-openjdk-amd64 \
			/usr/lib/jvm/java-21-openjdk \
			/usr/lib/jvm/java-17-openjdk; do
			if [ -d "$jh" ] && _set_java_from_home "$jh" && _java_version_ok; then
				return 0
			fi
		done
		;;
	MINGW* | MSYS* | CYGWIN*)
		for base in \
			"/c/Program Files/Eclipse Adoptium" \
			"/c/Program Files/Java" \
			"/c/Program Files/Microsoft" \
			"/c/Program Files/OpenJDK"; do
			if [ -d "$base" ]; then
				# Prefer 17, then 21 (newer LTS)
				for jh in "$base"/jdk-17* "$base"/jdk-21*; do
					if [ -d "$jh" ] && _set_java_from_home "$jh" && _java_version_ok; then
						return 0
					fi
				done
			fi
		done
		;;
	esac

	return 1
}
