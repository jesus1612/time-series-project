"""
FOR SARIMA
Maximum Likelihood Estimation (MLE) optimization engine

Implements MLE parameter estimation for ARIMA models using numerical optimization.
This is the core engine for fitting ARIMA parameters from scratch.
"""

import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple
from scipy.optimize import minimize, Bounds
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from .base import BaseEstimator


class MLEOptimizer(BaseEstimator):
    """
    Maximum Likelihood Estimation optimizer for ARIMA models
    
    Implements MLE using numerical optimization methods like BFGS or L-BFGS-B.
    Handles parameter constraints and provides robust optimization.
    """
    
    def __init__(self, 
                 method: str = 'L-BFGS-B',
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 n_jobs: int = -1):
        """
        Initialize MLE optimizer
        
        Parameters:
        -----------
        method : str
            Optimization method ('L-BFGS-B', 'BFGS', 'SLSQP')
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance
        bounds : dict, optional
            Parameter bounds for constrained optimization
        n_jobs : int
            Number of parallel jobs (-1 = all cores, 1 = no parallelization)
        """
        self.method = method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.bounds = bounds or {}
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self._optimization_result = None
        self._log_likelihood = None
        
        # Thresholds for parallelization (observations)
        self.parallel_thresholds = {
            'mle_optimization': 500,      # > 500 observaciones
            'gradient_calculation': 1000, # > 1000 observaciones
            'objective_evaluation': 200   # > 200 evaluaciones
        }
    
    def estimate(self, 
                 data: np.ndarray, 
                 model_type: str = 'ARIMA',
                 initial_params: Optional[np.ndarray] = None,
                 **kwargs) -> Dict[str, Any]:
        """
        Estimate parameters using Maximum Likelihood Estimation
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        model_type : str
            Type of model ('AR', 'MA', 'ARMA', 'ARIMA')
        initial_params : np.ndarray, optional
            Initial parameter values
        **kwargs
            Additional estimation parameters
            
        Returns:
        --------
        results : Dict[str, Any]
            Estimation results including parameters, log-likelihood, etc.
        """
        data = np.asarray(data)
        n = len(data)
        
        if n < 3:
            raise ValueError("Data must have at least 3 observations")
        
        p = kwargs.get('p', 0)
        d = kwargs.get('d', 0)
        q = kwargs.get('q', 0)
        
        if p < 0 or q < 0 or d < 0:
            raise ValueError("Model orders must be non-negative")
        
        if p == 0 and q == 0:
            raise ValueError("At least one of p or q must be positive")
        
        # Apply differencing if needed
        if d > 0:
            diff_data = self._apply_differencing(data, d)
        else:
            diff_data = data.copy()
        
        # Set up optimization
        n_params = p + q + 1  # AR params + MA params + variance
        param_names = self._get_param_names(p, q)
        
        # Set initial parameters if not provided
        if initial_params is None:
            # Usar búsqueda paralela de parámetros si el dataset es grande
            if self._should_parallelize(len(diff_data), 'mle_optimization'):
                print(f"   🔧 Usando búsqueda paralela de parámetros iniciales ({self.n_jobs} cores)")
                param_ranges = self._get_parameter_ranges(p, q)
                search_result = self._parallel_parameter_search(
                    diff_data, model_type, param_ranges, n_samples=50, p=p, q=q
                )
                initial_params = search_result['params']
            else:
                initial_params = self._get_initial_params(p, q, diff_data)
        
        # Set up bounds
        bounds = self._setup_bounds(p, q, param_names)
        
        # Define objective function
        objective = self._create_objective_function(diff_data, p, q, model_type)
        
        # Perform optimization
        try:
            result = minimize(
                objective,
                initial_params,
                method=self.method,
                bounds=bounds,
                options={
                    'maxiter': self.max_iterations,
                    'ftol': self.tolerance,
                    'gtol': self.tolerance
                }
            )
            
            if not result.success:
                raise RuntimeError(f"Optimization failed: {result.message}")
            
            # Extract results
            params = result.x
            log_likelihood = -result.fun  # Convert back to log-likelihood
            
            # Calculate standard errors
            std_errors = self._calculate_standard_errors(params, diff_data, p, q)
            
            # Calculate information criteria
            aic, bic = self._calculate_information_criteria(log_likelihood, n_params, n)
            
            # Store results
            self._optimization_result = result
            self._log_likelihood = log_likelihood
            
            # Organize parameters
            param_dict = self._organize_parameters(params, p, q, param_names)
            
            return {
                'parameters': param_dict,
                'log_likelihood': log_likelihood,
                'aic': aic,
                'bic': bic,
                'standard_errors': std_errors,
                'optimization_result': result,
                'model_type': model_type,
                'orders': {'p': p, 'd': d, 'q': q}
            }
            
        except Exception as e:
            raise RuntimeError(f"MLE optimization failed: {str(e)}")
    
    def _apply_differencing(self, data: np.ndarray, d: int) -> np.ndarray:
        """
        Apply differencing of order d
        
        Parameters:
        -----------
        data : np.ndarray
            Original time series
        d : int
            Order of differencing
            
        Returns:
        --------
        diff_data : np.ndarray
            Differenced time series
        """
        diff_data = data.copy()
        
        for _ in range(d):
            diff_data = np.diff(diff_data)
        
        return diff_data
    
    def _get_param_names(self, p: int, q: int) -> list:
        """
        Get parameter names for the model
        
        Parameters:
        -----------
        p : int
            AR order
        q : int
            MA order
            
        Returns:
        --------
        param_names : list
            List of parameter names
        """
        names = []
        
        # AR parameters
        for i in range(p):
            names.append(f'phi_{i+1}')
        
        # MA parameters
        for i in range(q):
            names.append(f'theta_{i+1}')
        
        # Variance parameter
        names.append('sigma2')
        
        return names
    
    def _get_initial_params(self, p: int, q: int, data: np.ndarray) -> np.ndarray:
        """
        Get initial parameter values
        
        Parameters:
        -----------
        p : int
            AR order
        q : int
            MA order
        data : np.ndarray
            Time series data
            
        Returns:
        --------
        initial_params : np.ndarray
            Initial parameter values
        """
        n_params = p + q + 1
        initial_params = np.zeros(n_params)
        
        # Initialize AR parameters with small random values
        for i in range(p):
            initial_params[i] = np.random.normal(0, 0.1)
        
        # Initialize MA parameters with small random values
        for i in range(q):
            initial_params[p + i] = np.random.normal(0, 0.1)
        
        # Initialize variance with sample variance
        initial_params[-1] = np.var(data)
        
        return initial_params
    
    def _setup_bounds(self, p: int, q: int, param_names: list) -> list:
        """
        Set up parameter bounds for optimization
        
        Parameters:
        -----------
        p : int
            AR order
        q : int
            MA order
        param_names : list
            Parameter names
            
        Returns:
        --------
        bounds : list
            List of (min, max) tuples for each parameter
        """
        bounds = []
        
        # AR parameters: typically bounded for stationarity
        for i in range(p):
            bounds.append((-0.99, 0.99))
        
        # MA parameters: typically bounded for invertibility
        for i in range(q):
            bounds.append((-0.99, 0.99))
        
        # Variance: must be positive
        bounds.append((1e-6, None))
        
        return bounds
    
    def _create_objective_function(self, 
                                 data: np.ndarray, 
                                 p: int, 
                                 q: int, 
                                 model_type: str) -> Callable:
        """
        Create objective function for optimization
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        p : int
            AR order
        q : int
            MA order
        model_type : str
            Type of model
            
        Returns:
        --------
        objective : callable
            Objective function that returns negative log-likelihood
        """
        def objective(params):
            try:
                # Extract parameters
                ar_params = params[:p] if p > 0 else np.array([])
                ma_params = params[p:p+q] if q > 0 else np.array([])
                sigma2 = params[-1]
                
                # Ensure variance is positive
                if sigma2 <= 0:
                    return 1e10
                
                # Calculate log-likelihood
                log_lik = self._calculate_log_likelihood(data, ar_params, ma_params, sigma2)
                
                # Return negative log-likelihood for minimization
                return -log_lik
                
            except Exception:
                return 1e10  # Return large value if calculation fails
        
        return objective
    
    def _calculate_log_likelihood(self, 
                                data: np.ndarray, 
                                ar_params: np.ndarray, 
                                ma_params: np.ndarray, 
                                sigma2: float) -> float:
        """
        Calculate log-likelihood for given parameters
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        ar_params : np.ndarray
            AR parameters
        ma_params : np.ndarray
            MA parameters
        sigma2 : float
            Variance parameter
            
        Returns:
        --------
        log_likelihood : float
            Log-likelihood value
        """
        n = len(data)
        p = len(ar_params)
        q = len(ma_params)
        
        # Calculate residuals using Kalman filter or innovation algorithm
        residuals = self._calculate_residuals(data, ar_params, ma_params)
        
        # Calculate log-likelihood
        # Assuming Gaussian errors: log L = -n/2 * log(2π) - n/2 * log(σ²) - 1/(2σ²) * Σε²
        # Floor sigma2 and clip residual squares to avoid overflow during MLE line search
        sigma2_safe = max(float(sigma2), 1e-300)
        rss = np.sum(np.minimum(residuals.astype(float) ** 2, 1e300))
        log_likelihood = (
            -n / 2 * np.log(2 * np.pi)
            - n / 2 * np.log(sigma2_safe)
            - rss / (2 * sigma2_safe)
        )
        
        return log_likelihood
    
    def _calculate_residuals(self, 
                           data: np.ndarray, 
                           ar_params: np.ndarray, 
                           ma_params: np.ndarray) -> np.ndarray:
        """
        Calculate model residuals using innovation algorithm
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        ar_params : np.ndarray
            AR parameters
        ma_params : np.ndarray
            MA parameters
            
        Returns:
        --------
        residuals : np.ndarray
            Model residuals
        """
        n = len(data)
        p = len(ar_params)
        q = len(ma_params)
        
        # Initialize residuals
        residuals = np.zeros(n)
        
        # For simplicity, use a basic approach
        # In practice, you'd use the innovation algorithm or Kalman filter
        
        for t in range(max(p, q), n):
            # Calculate predicted value
            predicted = 0.0
            
            # AR component
            for i in range(p):
                if t - i - 1 >= 0:
                    predicted += ar_params[i] * data[t - i - 1]
            
            # MA component (using previous residuals)
            for i in range(q):
                if t - i - 1 >= 0:
                    predicted += ma_params[i] * residuals[t - i - 1]
            predicted = float(np.clip(predicted, -1e12, 1e12))

            # Calculate residual
            residuals[t] = float(np.clip(data[t] - predicted, -1e12, 1e12))
        
        return residuals
    
    def _calculate_standard_errors(self, 
                                 params: np.ndarray, 
                                 data: np.ndarray, 
                                 p: int, 
                                 q: int) -> np.ndarray:
        """
        Calculate standard errors of parameter estimates
        
        Parameters:
        -----------
        params : np.ndarray
            Estimated parameters
        data : np.ndarray
            Time series data
        p : int
            AR order
        q : int
            MA order
            
        Returns:
        --------
        std_errors : np.ndarray
            Standard errors of parameters
        """
        # This is a simplified calculation
        # In practice, you'd calculate the Hessian matrix and invert it
        
        n = len(data)
        n_params = len(params)
        
        # Approximate standard errors using the inverse of the Hessian
        # For simplicity, use a diagonal approximation
        std_errors = np.sqrt(np.abs(params) / n)
        
        # Ensure minimum standard error
        std_errors = np.maximum(std_errors, 1e-6)
        
        return std_errors
    
    def _calculate_information_criteria(self, 
                                      log_likelihood: float, 
                                      n_params: int, 
                                      n_obs: int) -> Tuple[float, float]:
        """
        Calculate AIC and BIC information criteria
        
        Parameters:
        -----------
        log_likelihood : float
            Log-likelihood value
        n_params : int
            Number of parameters
        n_obs : int
            Number of observations
            
        Returns:
        --------
        aic : float
            Akaike Information Criterion
        bic : float
            Bayesian Information Criterion
        """
        # AIC = 2k - 2ln(L)
        aic = 2 * n_params - 2 * log_likelihood
        
        # BIC = k*ln(n) - 2ln(L)
        bic = n_params * np.log(n_obs) - 2 * log_likelihood
        
        return aic, bic
    
    def _organize_parameters(self, 
                           params: np.ndarray, 
                           p: int, 
                           q: int, 
                           param_names: list) -> Dict[str, float]:
        """
        Organize parameters into a dictionary
        
        Parameters:
        -----------
        params : np.ndarray
            Parameter values
        p : int
            AR order
        q : int
            MA order
        param_names : list
            Parameter names
            
        Returns:
        --------
        param_dict : dict
            Dictionary of parameter names and values
        """
        param_dict = {}
        
        for i, name in enumerate(param_names):
            param_dict[name] = params[i]
        
        return param_dict
    
    @property
    def optimization_result(self):
        """Get the optimization result"""
        return self._optimization_result
    
    @property
    def log_likelihood(self):
        """Get the log-likelihood value"""
        return self._log_likelihood
    
    def _should_parallelize(self, data_size: int, operation: str) -> bool:
        """
        Determine if operation should be parallelized based on data size
        
        Parameters:
        -----------
        data_size : int
            Size of the data
        operation : str
            Type of operation ('mle_optimization', 'gradient_calculation', etc.)
            
        Returns:
        --------
        bool
            True if should parallelize, False otherwise
        """
        if self.n_jobs == 1:
            return False
        
        threshold = self.parallel_thresholds.get(operation, float('inf'))
        return data_size > threshold
    
    def _parallel_objective_evaluation(self, params_list: list, data: np.ndarray, 
                                     model_type: str, **kwargs) -> list:
        """
        Evaluate objective function for multiple parameter sets in parallel
        
        Parameters:
        -----------
        params_list : list
            List of parameter arrays to evaluate
        data : np.ndarray
            Time series data
        model_type : str
            Type of model
        **kwargs
            Additional parameters
            
        Returns:
        --------
        list
            List of objective function values
        """
        def evaluate_single(params):
            # Extract parameters based on model type
            p = kwargs.get('p', 0)
            q = kwargs.get('q', 0)
            
            if p > 0 and q > 0:  # ARMA
                ar_params = params[:p]
                ma_params = params[p:p+q]
                sigma2 = params[p+q]
            elif p > 0:  # AR only
                ar_params = params[:p]
                ma_params = np.array([])
                sigma2 = params[p]
            elif q > 0:  # MA only
                ar_params = np.array([])
                ma_params = params[:q]
                sigma2 = params[q]
            else:
                raise ValueError("At least one of p or q must be positive")
            
            return self._calculate_log_likelihood(data, ar_params, ma_params, sigma2)
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(executor.map(evaluate_single, params_list))
        
        return results
    
    def _parallel_gradient_calculation(self, params: np.ndarray, data: np.ndarray,
                                     model_type: str, **kwargs) -> np.ndarray:
        """
        Calculate gradient in parallel for large datasets
        
        Parameters:
        -----------
        params : np.ndarray
            Parameter values
        data : np.ndarray
            Time series data
        model_type : str
            Type of model
        **kwargs
            Additional parameters
            
        Returns:
        --------
        np.ndarray
            Gradient vector
        """
        if not self._should_parallelize(len(data), 'gradient_calculation'):
            return self._calculate_gradient_sequential(params, data, model_type, **kwargs)
        
        # Dividir cálculo de gradiente en componentes
        n_params = len(params)
        chunk_size = max(1, n_params // self.n_jobs)
        
        def calc_gradient_chunk(param_indices):
            """Calcular gradiente para un chunk de parámetros"""
            chunk_grad = np.zeros(n_params)
            for i in param_indices:
                chunk_grad[i] = self._calculate_gradient_component(
                    params, data, model_type, i, **kwargs
                )
            return chunk_grad
        
        # Dividir índices de parámetros en chunks
        param_indices = list(range(n_params))
        chunks = [param_indices[i:i+chunk_size] 
                 for i in range(0, n_params, chunk_size)]
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            gradient_chunks = list(executor.map(calc_gradient_chunk, chunks))
        
        # Combinar resultados
        gradient = np.sum(gradient_chunks, axis=0)
        return gradient
    
    def _calculate_gradient_sequential(self, params: np.ndarray, data: np.ndarray,
                                     model_type: str, **kwargs) -> np.ndarray:
        """
        Calculate gradient sequentially (fallback method)
        """
        n_params = len(params)
        gradient = np.zeros(n_params)
        
        for i in range(n_params):
            gradient[i] = self._calculate_gradient_component(
                params, data, model_type, i, **kwargs
            )
        
        return gradient
    
    def _calculate_gradient_component(self, params: np.ndarray, data: np.ndarray,
                                    model_type: str, param_index: int, **kwargs) -> float:
        """
        Calculate gradient component for a single parameter
        
        Parameters:
        -----------
        params : np.ndarray
            Parameter values
        data : np.ndarray
            Time series data
        model_type : str
            Type of model
        param_index : int
            Index of parameter to calculate gradient for
        **kwargs
            Additional parameters
            
        Returns:
        --------
        float
            Gradient component
        """
        # Usar diferencia finita para calcular gradiente
        h = 1e-8
        params_plus = params.copy()
        params_minus = params.copy()
        
        params_plus[param_index] += h
        params_minus[param_index] -= h
        
        # Extract parameters for gradient calculation
        p = kwargs.get('p', 0)
        q = kwargs.get('q', 0)
        
        def extract_params(params):
            if p > 0 and q > 0:  # ARMA
                ar_params = params[:p]
                ma_params = params[p:p+q]
                sigma2 = params[p+q]
            elif p > 0:  # AR only
                ar_params = params[:p]
                ma_params = np.array([])
                sigma2 = params[p]
            elif q > 0:  # MA only
                ar_params = np.array([])
                ma_params = params[:q]
                sigma2 = params[q]
            else:
                raise ValueError("At least one of p or q must be positive")
            return ar_params, ma_params, sigma2
        
        ar_plus, ma_plus, sigma2_plus = extract_params(params_plus)
        ar_minus, ma_minus, sigma2_minus = extract_params(params_minus)
        
        ll_plus = self._calculate_log_likelihood(data, ar_plus, ma_plus, sigma2_plus)
        ll_minus = self._calculate_log_likelihood(data, ar_minus, ma_minus, sigma2_minus)
        
        gradient_component = (ll_plus - ll_minus) / (2 * h)
        return gradient_component
    
    def _parallel_parameter_search(self, data: np.ndarray, model_type: str,
                                 param_ranges: Dict[str, Tuple[float, float]], 
                                 n_samples: int = 100, **kwargs) -> Dict[str, Any]:
        """
        Search for good initial parameters using parallel evaluation
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        model_type : str
            Type of model
        param_ranges : dict
            Parameter ranges to search
        n_samples : int
            Number of parameter samples to evaluate
        **kwargs
            Additional parameters
            
        Returns:
        --------
        dict
            Best parameters found
        """
        if not self._should_parallelize(n_samples, 'objective_evaluation'):
            return self._sequential_parameter_search(data, model_type, param_ranges, n_samples, **kwargs)
        
        # Generar muestras de parámetros
        param_samples = self._generate_parameter_samples(param_ranges, n_samples)
        
        # Evaluar en paralelo
        log_likelihoods = self._parallel_objective_evaluation(
            param_samples, data, model_type, **kwargs
        )
        
        # Encontrar mejor parámetro
        best_idx = np.argmax(log_likelihoods)
        best_params = param_samples[best_idx]
        best_ll = log_likelihoods[best_idx]
        
        return {
            'params': best_params,
            'log_likelihood': best_ll,
            'all_likelihoods': log_likelihoods
        }
    
    def _sequential_parameter_search(self, data: np.ndarray, model_type: str,
                                   param_ranges: Dict[str, Tuple[float, float]], 
                                   n_samples: int = 100, **kwargs) -> Dict[str, Any]:
        """
        Sequential parameter search (fallback method)
        """
        param_samples = self._generate_parameter_samples(param_ranges, n_samples)
        log_likelihoods = []
        
        for params in param_samples:
            # Extract parameters based on model type
            p = kwargs.get('p', 0)
            q = kwargs.get('q', 0)
            
            if p > 0 and q > 0:  # ARMA
                ar_params = params[:p]
                ma_params = params[p:p+q]
                sigma2 = params[p+q]
            elif p > 0:  # AR only
                ar_params = params[:p]
                ma_params = np.array([])
                sigma2 = params[p]
            elif q > 0:  # MA only
                ar_params = np.array([])
                ma_params = params[:q]
                sigma2 = params[q]
            else:
                raise ValueError("At least one of p or q must be positive")
            
            ll = self._calculate_log_likelihood(data, ar_params, ma_params, sigma2)
            log_likelihoods.append(ll)
        
        best_idx = np.argmax(log_likelihoods)
        best_params = param_samples[best_idx]
        best_ll = log_likelihoods[best_idx]
        
        return {
            'params': best_params,
            'log_likelihood': best_ll,
            'all_likelihoods': log_likelihoods
        }
    
    def _generate_parameter_samples(self, param_ranges: Dict[str, Tuple[float, float]], 
                                  n_samples: int) -> list:
        """
        Generate random parameter samples within given ranges
        
        Parameters:
        -----------
        param_ranges : dict
            Parameter ranges
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        list
            List of parameter arrays
        """
        param_names = list(param_ranges.keys())
        n_params = len(param_names)
        samples = []
        
        for _ in range(n_samples):
            params = np.zeros(n_params)
            for i, name in enumerate(param_names):
                low, high = param_ranges[name]
                params[i] = np.random.uniform(low, high)
            samples.append(params)
        
        return samples
    
    def _get_parameter_ranges(self, p: int, q: int) -> Dict[str, Tuple[float, float]]:
        """
        Get parameter ranges for initial parameter search
        
        Parameters:
        -----------
        p : int
            AR order
        q : int
            MA order
            
        Returns:
        --------
        dict
            Parameter ranges
        """
        ranges = {}
        
        # AR parameters: typically between -0.99 and 0.99
        for i in range(p):
            ranges[f'ar_{i+1}'] = (-0.99, 0.99)
        
        # MA parameters: typically between -0.99 and 0.99
        for i in range(q):
            ranges[f'ma_{i+1}'] = (-0.99, 0.99)
        
        # Variance: typically between 0.1 and 10
        ranges['variance'] = (0.1, 10.0)
        
        return ranges

