.PHONY: install run clean clean-all format help setup-venv check-venv check-java setup-java-env install-java install-java-macos install-java-windows

# Default target
help:
	@echo "Available commands:"
	@echo "  install         - Install dependencies (creates venv if needed)"
	@echo "  run             - Run the Shiny application (requires venv, checks Java for ARIMA parallel)"
	@echo "  check-java      - Check if Java 17+ is installed and configured"
	@echo "  install-java    - Install Java 17+ (macOS: Homebrew | Windows: winget + hints)"
	@echo "  setup-java-env  - Configure Java environment variables"
	@echo "  clean           - Clean cache and temporary files"
	@echo "  clean-all       - Clean everything including virtual environment"
	@echo "  format          - Format Python code with black"
	@echo "  setup-venv      - Create virtual environment manually"
	@echo "  help            - Show this help message"

# Check if virtual environment exists
check-venv:
	@if [ ! -d "venv" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv venv; \
		echo "Virtual environment created."; \
	fi

# Check Java version compatibility for Spark (macOS Homebrew, Windows Adoptium paths, Linux OpenJDK, or PATH)
check-java:
	@echo "Checking Java installation..."
	@. scripts/ensure_java_env.sh; \
	if ensure_java_env; then \
		echo "✓ Java 17+ OK$${JAVA_HOME:+ (JAVA_HOME=$$JAVA_HOME)}"; \
		java -version 2>&1 | head -n 1; \
	else \
		echo "✗ Java 17+ not found. PySpark requires a JDK 17 or newer."; \
		echo "💡 Run 'make install-java' (macOS/Windows) or install openjdk-17-jdk (Linux)"; \
		exit 1; \
	fi

# Setup Java environment variables
setup-java-env:
	@. scripts/ensure_java_env.sh; \
	if ensure_java_env; then \
		echo "✓ JAVA_HOME=$$JAVA_HOME"; \
		java -version 2>&1 | head -n 1; \
	else \
		echo "✗ Java 17+ not found — set JAVA_HOME or run 'make install-java'"; \
	fi

# Install Java 17+ (OS-specific: macOS Homebrew, Windows winget + manual hints)
install-java:
	@uname_s=$$(uname -s 2>/dev/null || echo unknown); \
	case "$$uname_s" in \
		MINGW*|MSYS*|CYGWIN*) $(MAKE) install-java-windows ;; \
		Darwin*) $(MAKE) install-java-macos ;; \
		*) echo "Auto-install only on macOS (Homebrew) or Windows (winget). Sistema: $$uname_s"; \
		   echo "  Linux: sudo apt install openjdk-17-jdk  (o equivalente)"; \
		   echo "  Manual: https://adoptium.net/temurin/releases/?version=17" ;; \
	esac

# Install Java 17+ on macOS
install-java-macos:
	@echo "Installing Java 17+ on macOS..."
	@if command -v brew >/dev/null 2>&1; then \
		if [ -d "/opt/homebrew/opt/openjdk@17" ] || [ -d "/usr/local/opt/openjdk@17" ]; then \
			echo "✓ Java 17 already installed"; \
		else \
			brew install openjdk@17 || true; \
		fi; \
		if [ -d "/opt/homebrew/opt/openjdk@17" ]; then \
			echo "✓ Java 17 installed at /opt/homebrew/opt/openjdk@17"; \
		elif [ -d "/usr/local/opt/openjdk@17" ]; then \
			echo "✓ Java 17 installed at /usr/local/opt/openjdk@17"; \
		else \
			echo "✗ Java 17 installation failed. Please install manually:"; \
			echo "  https://adoptium.net/temurin/releases/?version=17"; \
		fi; \
	else \
		echo "Homebrew not found. Please install Java 17+ manually:"; \
		echo "  https://adoptium.net/temurin/releases/?version=17"; \
	fi

# Install Java 17+ on Windows (Git Bash / MSYS — Temurin via winget when available)
install-java-windows:
	@echo "Installing Java 17+ on Windows..."
	@if command -v winget >/dev/null 2>&1; then \
		echo "Trying winget (Eclipse Temurin JDK 17)..."; \
		winget install -e --id EclipseAdoptium.Temurin.17.JDK --accept-package-agreements --accept-source-agreements || true; \
	else \
		echo "winget not in PATH. Install JDK 17+ manually or from Microsoft Store."; \
	fi
	@echo ""
	@echo "After install, set user environment variables if needed:"
	@echo "  JAVA_HOME -> JDK folder (e.g. C:\\Program Files\\Eclipse Adoptium\\jdk-17.x.x-hotspot)"
	@echo "  Add %JAVA_HOME%\\bin to PATH"
	@echo "Download: https://adoptium.net/temurin/releases/?version=17"

# Create virtual environment manually
setup-venv:
	python3 -m venv venv
	@echo "Virtual environment created. Run 'source venv/bin/activate' to activate it."

# Install dependencies (auto-creates venv if needed)
install: check-venv
	@echo "Installing dependencies..."
	@if [ -f "venv/bin/activate" ]; then \
		. venv/bin/activate && pip install -r requirements.txt; \
	else \
		python3 -m pip install -r requirements.txt; \
	fi
	@echo "Dependencies installed successfully!"

# Run the application
run: check-venv
	@if [ ! -f "venv/bin/activate" ]; then \
		echo "❌ Error: El entorno virtual está corrupto."; \
		echo "💡 Solución: Ejecuta 'make clean && make install' para recrear el entorno."; \
		exit 1; \
	fi
	@echo "Verificando Java para modelos paralelos (PySpark)..."
	@echo "✅ Ejecutando aplicación con entorno virtual (hot reload activado)..."
	@. scripts/ensure_java_env.sh; ensure_java_env || true; \
	. venv/bin/activate && shiny run --reload --host 0.0.0.0 --port 8000 app:app

# Clean cache and temporary files
clean:
	@echo "🧹 Limpiando archivos temporales..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "✅ Limpieza completada."

# Clean everything including virtual environment
clean-all: clean
	@echo "🗑️ Eliminando entorno virtual..."
	rm -rf venv
	@echo "✅ Entorno virtual eliminado."

# Format code
format:
	black . --line-length 88
