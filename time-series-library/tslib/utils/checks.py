"""
Utility functions for dependency checks and system validation

Provides functions to check for optional dependencies like PySpark
and validate system requirements.
"""

import importlib
import sys
from typing import Dict, Any, Optional


def check_spark_availability() -> bool:
    """
    Check if PySpark is available in the system
    
    Returns:
    --------
    is_available : bool
        True if PySpark is available, False otherwise
    """
    try:
        import pyspark
        return True
    except ImportError:
        return False


def check_dependency(dependency_name: str, min_version: Optional[str] = None) -> Dict[str, Any]:
    """
    Check if a dependency is available and optionally verify version
    
    Parameters:
    -----------
    dependency_name : str
        Name of the dependency to check
    min_version : str, optional
        Minimum required version
        
    Returns:
    --------
    check_result : dict
        Dictionary containing availability and version information
    """
    result = {
        'available': False,
        'version': None,
        'error': None
    }
    
    try:
        module = importlib.import_module(dependency_name)
        result['available'] = True
        
        if hasattr(module, '__version__'):
            result['version'] = module.__version__
        elif hasattr(module, 'version'):
            result['version'] = module.version
        
        # Check minimum version if specified
        if min_version and result['version']:
            if _compare_versions(result['version'], min_version) < 0:
                result['error'] = f"Version {result['version']} is below minimum required {min_version}"
        
    except ImportError as e:
        result['error'] = str(e)
    except Exception as e:
        result['error'] = f"Unexpected error: {str(e)}"
    
    return result


def _compare_versions(version1: str, version2: str) -> int:
    """
    Compare two version strings
    
    Parameters:
    -----------
    version1 : str
        First version string
    version2 : str
        Second version string
        
    Returns:
    --------
    comparison : int
        -1 if version1 < version2, 0 if equal, 1 if version1 > version2
    """
    def version_tuple(v):
        return tuple(map(int, (v.split("."))))
    
    try:
        v1_tuple = version_tuple(version1)
        v2_tuple = version_tuple(version2)
        
        if v1_tuple < v2_tuple:
            return -1
        elif v1_tuple > v2_tuple:
            return 1
        else:
            return 0
    except ValueError:
        # If version parsing fails, do string comparison
        if version1 < version2:
            return -1
        elif version1 > version2:
            return 1
        else:
            return 0


def check_system_requirements() -> Dict[str, Any]:
    """
    Check system requirements for the time series library
    
    Returns:
    --------
    requirements : dict
        System requirements check results
    """
    requirements = {
        'python_version': {
            'required': '3.8+',
            'current': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'satisfied': sys.version_info >= (3, 8)
        },
        'dependencies': {}
    }
    
    # Check core dependencies
    core_deps = {
        'numpy': '1.21.0',
        'scipy': '1.7.0',
        'pandas': '1.3.0',
        'matplotlib': '3.4.0'
    }
    
    for dep, min_version in core_deps.items():
        requirements['dependencies'][dep] = check_dependency(dep, min_version)
    
    # Check optional dependencies
    optional_deps = {
        'pyspark': '3.2.0',
        'sklearn': None  # No minimum version requirement
    }
    
    for dep, min_version in optional_deps.items():
        requirements['dependencies'][dep] = check_dependency(dep, min_version)
    
    return requirements


def validate_environment() -> bool:
    """
    Validate that the environment meets minimum requirements
    
    Returns:
    --------
    is_valid : bool
        True if environment is valid, False otherwise
    """
    requirements = check_system_requirements()
    
    # Check Python version
    if not requirements['python_version']['satisfied']:
        return False
    
    # Check core dependencies
    core_deps = ['numpy', 'scipy', 'pandas', 'matplotlib']
    for dep in core_deps:
        if not requirements['dependencies'][dep]['available']:
            return False
    
    return True


def get_environment_info() -> str:
    """
    Get comprehensive environment information
    
    Returns:
    --------
    info : str
        Formatted environment information
    """
    requirements = check_system_requirements()
    
    info = "Environment Information\n"
    info += "=" * 50 + "\n\n"
    
    # Python version
    python_info = requirements['python_version']
    status = "✅" if python_info['satisfied'] else "❌"
    info += f"Python Version: {python_info['current']} {status}\n"
    info += f"Required: {python_info['required']}\n\n"
    
    # Dependencies
    info += "Dependencies:\n"
    for dep_name, dep_info in requirements['dependencies'].items():
        status = "✅" if dep_info['available'] else "❌"
        version_info = f" (v{dep_info['version']})" if dep_info['version'] else ""
        info += f"  {dep_name}: {status}{version_info}\n"
        
        if dep_info['error']:
            info += f"    Error: {dep_info['error']}\n"
    
    return info


def check_memory_usage() -> Dict[str, Any]:
    """
    Check current memory usage (if psutil is available)
    
    Returns:
    --------
    memory_info : dict
        Memory usage information
    """
    try:
        import psutil
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'available': True,
            'rss': memory_info.rss,  # Resident Set Size
            'vms': memory_info.vms,  # Virtual Memory Size
            'percent': process.memory_percent(),
            'system_memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent
            }
        }
    except ImportError:
        return {
            'available': False,
            'error': 'psutil not available'
        }


def check_cpu_info() -> Dict[str, Any]:
    """
    Check CPU information (if psutil is available)
    
    Returns:
    --------
    cpu_info : dict
        CPU information
    """
    try:
        import psutil
        
        return {
            'available': True,
            'count': psutil.cpu_count(),
            'count_logical': psutil.cpu_count(logical=True),
            'count_physical': psutil.cpu_count(logical=False),
            'freq': psutil.cpu_freq(),
            'percent': psutil.cpu_percent(interval=1)
        }
    except ImportError:
        return {
            'available': False,
            'error': 'psutil not available'
        }




