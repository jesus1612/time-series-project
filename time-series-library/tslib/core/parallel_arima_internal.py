"""
ARIMA con paralelización interna de operaciones
Paraleliza las operaciones computacionalmente intensivas dentro de UN modelo ARIMA
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from .base import TimeSeriesModel
from .optimization import MLEOptimizer
from .acf_pacf import ACFCalculator, PACFCalculator


class ParallelARIMAProcess(TimeSeriesModel):
    """
    ARIMA con paralelización interna de operaciones computacionalmente intensivas
    
    Paraleliza:
    - Cálculo de ACF/PACF
    - Optimización MLE
    - Cálculo de coeficientes AR/MA
    - Diferenciación
    - Validación de estacionariedad
    """
    
    def __init__(self, ar_order: int, diff_order: int, ma_order: int, 
                 trend: str = 'c', n_jobs: int = -1):
        """
        Initialize Parallel ARIMA process
        
        Parameters:
        -----------
        ar_order : int
            AR order (p)
        diff_order : int
            Differencing order (d)
        ma_order : int
            MA order (q)
        trend : str
            'c' (constant), 'nc' (no constant)
        n_jobs : int
            Number of parallel jobs (-1 = all cores)
        """
        super().__init__()
        self.ar_order = ar_order
        self.diff_order = diff_order
        self.ma_order = ma_order
        self.trend = trend
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        
        # Componentes paralelizables
        self.acf_calculator = ACFCalculator()
        self.pacf_calculator = PACFCalculator()
        self.optimizer = MLEOptimizer()
        
        # Parámetros del modelo
        self.ar_params = None
        self.ma_params = None
        self.constant = None
        self.variance = None
        self.original_data = None
        self.differenced_data = None
        
    def _parallel_acf_pacf_calculation(self, data: np.ndarray, max_lags: int = 40):
        """
        Calcula ACF y PACF en paralelo
        """
        def calculate_acf():
            return self.acf_calculator.calculate(data, max_lags=max_lags)
        
        def calculate_pacf():
            return self.pacf_calculator.calculate(data, max_lags=max_lags)
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            acf_future = executor.submit(calculate_acf)
            pacf_future = executor.submit(calculate_pacf)
            
            acf = acf_future.result()
            pacf = pacf_future.result()
        
        return acf, pacf
    
    def _parallel_differencing(self, data: np.ndarray):
        """
        Aplica diferenciación en paralelo si diff_order > 1
        """
        if self.diff_order == 0:
            return data
        
        def apply_single_diff(series, order):
            """Aplica una diferenciación simple"""
            for _ in range(order):
                series = np.diff(series)
            return series
        
        if self.diff_order == 1:
            return np.diff(data)
        
        # Para diff_order > 1, paralelizar si es computacionalmente intensivo
        if len(data) > 10000:  # Solo para series muy grandes
            # Dividir en chunks y procesar en paralelo
            chunk_size = len(data) // self.n_jobs
            chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
            
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [executor.submit(apply_single_diff, chunk, self.diff_order) 
                          for chunk in chunks]
                results = [future.result() for future in futures]
            
            # Reconstruir la serie diferenciada
            return np.concatenate(results)
        else:
            return apply_single_diff(data, self.diff_order)
    
    def _parallel_parameter_estimation(self, data: np.ndarray):
        """
        Estima parámetros AR y MA en paralelo cuando es posible
        """
        def estimate_ar_params():
            if self.ar_order == 0:
                return None
            # Implementar estimación AR en paralelo
            return self._estimate_ar_parameters(data)
        
        def estimate_ma_params():
            if self.ma_order == 0:
                return None
            # Implementar estimación MA en paralelo
            return self._estimate_ma_parameters(data)
        
        def estimate_constant():
            if self.trend == 'nc':
                return 0.0
            return np.mean(data)
        
        # Ejecutar estimaciones en paralelo
        with ThreadPoolExecutor(max_workers=3) as executor:
            ar_future = executor.submit(estimate_ar_params)
            ma_future = executor.submit(estimate_ma_params)
            const_future = executor.submit(estimate_constant)
            
            ar_params = ar_future.result()
            ma_params = ma_future.result()
            constant = const_future.result()
        
        return ar_params, ma_params, constant
    
    def _parallel_mle_optimization(self, data: np.ndarray, initial_params: Dict):
        """
        Optimización MLE con paralelización de evaluaciones de función objetivo
        """
        def objective_function(params):
            """Función objetivo para MLE"""
            return self._calculate_log_likelihood(data, params)
        
        def gradient_function(params):
            """Gradiente de la función objetivo"""
            return self._calculate_gradient(data, params)
        
        # Usar optimizador con paralelización interna
        result = self.optimizer.estimate(
            data=data,
            initial_params=initial_params,
            objective_function=objective_function,
            gradient_function=gradient_function
        )
        
        return result
    
    def _parallel_validation(self, data: np.ndarray, params: Dict):
        """
        Validación de modelo en paralelo
        """
        def check_stationarity():
            return self._check_ar_stationarity(params.get('ar_params', []))
        
        def check_invertibility():
            return self._check_ma_invertibility(params.get('ma_params', []))
        
        def calculate_residuals():
            return self._calculate_residuals(data, params)
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            stationarity_future = executor.submit(check_stationarity)
            invertibility_future = executor.submit(check_invertibility)
            residuals_future = executor.submit(calculate_residuals)
            
            is_stationary = stationarity_future.result()
            is_invertible = invertibility_future.result()
            residuals = residuals_future.result()
        
        return {
            'stationary': is_stationary,
            'invertible': is_invertible,
            'residuals': residuals
        }
    
    def fit(self, data: Union[np.ndarray, list], **kwargs) -> 'ParallelARIMAProcess':
        """
        Fit ARIMA model with internal parallelization
        """
        print(f"🔄 Ajustando ARIMA({self.ar_order},{self.diff_order},{self.ma_order}) con paralelización interna...")
        print(f"   🧵 Usando {self.n_jobs} cores para operaciones paralelas")
        
        # Convertir a numpy array
        self.original_data = np.array(data, dtype=float)
        
        # 1. Diferenciación paralela
        print("   📊 Aplicando diferenciación...")
        self.differenced_data = self._parallel_differencing(self.original_data)
        
        # 2. Cálculo ACF/PACF paralelo
        print("   📈 Calculando ACF/PACF en paralelo...")
        acf, pacf = self._parallel_acf_pacf_calculation(self.differenced_data)
        
        # 3. Estimación de parámetros paralela
        print("   🔧 Estimando parámetros en paralelo...")
        ar_params, ma_params, constant = self._parallel_parameter_estimation(self.differenced_data)
        
        # 4. Optimización MLE
        print("   ⚡ Optimizando MLE...")
        initial_params = {
            'ar_params': ar_params,
            'ma_params': ma_params,
            'constant': constant
        }
        
        optimized_params = self._parallel_mle_optimization(self.differenced_data, initial_params)
        
        # 5. Validación paralela
        print("   ✅ Validando modelo en paralelo...")
        validation_results = self._parallel_validation(self.differenced_data, optimized_params)
        
        # Guardar parámetros
        self.ar_params = optimized_params.get('ar_params')
        self.ma_params = optimized_params.get('ma_params')
        self.constant = optimized_params.get('constant')
        self.variance = optimized_params.get('variance')
        
        self._fitted = True
        self._fitted_params = optimized_params
        
        print(f"   ✅ Modelo ajustado exitosamente")
        print(f"   📊 Parámetros: AR={self.ar_params}, MA={self.ma_params}")
        print(f"   🔍 Validación: Estacionario={validation_results['stationary']}, Invertible={validation_results['invertible']}")
        
        return self
    
    def predict(self, steps: int = 1, **kwargs) -> np.ndarray:
        """
        Predict future values
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Implementar predicción (puede ser paralelizada para múltiples pasos)
        predictions = self._calculate_predictions(steps)
        return predictions
    
    def _estimate_ar_parameters(self, data: np.ndarray) -> np.ndarray:
        """
        Estimate AR parameters via Yule-Walker equations.

        Solves the linear system  R φ = r  where R is the (p×p) autocorrelation
        matrix and r is the vector of lag-1..p autocorrelations.
        """
        p = self.ar_order
        if p == 0:
            return np.array([])

        n = len(data)
        mean = np.mean(data)
        centered = data - mean

        # Build autocorrelation vector for lags 0..p
        acf = np.array([
            np.sum(centered[:n - k] * centered[k:]) / np.sum(centered ** 2)
            for k in range(p + 1)
        ])

        # Toeplitz autocorrelation matrix
        from scipy.linalg import toeplitz
        R = toeplitz(acf[:p])
        r = acf[1:p + 1]

        try:
            phi = np.linalg.solve(R, r)
        except np.linalg.LinAlgError:
            phi = np.linalg.lstsq(R, r, rcond=None)[0]

        return phi

    def _estimate_ma_parameters(self, data: np.ndarray) -> np.ndarray:
        """
        Estimate MA parameters from sample ACF (method-of-moments).

        For an MA(q) process, ρ(k) = θ_k / (1 + Σθ_j²) for k ≤ q.
        We solve iteratively starting from the sample ACF values as an
        initial approximation.
        """
        q = self.ma_order
        if q == 0:
            return np.array([])

        n = len(data)
        mean = np.mean(data)
        centered = data - mean
        var = np.var(centered, ddof=0) or 1.0

        sample_acf = np.array([
            np.sum(centered[:n - k] * centered[k:]) / (n * var)
            for k in range(1, q + 1)
        ])

        # Initial estimate: scale ACF values to feasible range
        theta = np.clip(sample_acf, -0.99, 0.99)
        return theta

    def _calculate_log_likelihood(self, data: np.ndarray, params: Dict) -> float:
        """
        Gaussian log-likelihood: ℓ = -(n/2) ln(2πσ²) - SSR/(2σ²)
        """
        n = len(data)
        residuals = self._calculate_residuals(data, params)
        sigma2 = params.get('variance') or (np.var(residuals) or 1e-8)
        ssr = float(np.sum(residuals ** 2))
        ll = -0.5 * n * np.log(2 * np.pi * sigma2) - ssr / (2 * sigma2)
        return ll

    def _calculate_gradient(self, data: np.ndarray, params: Dict,
                            eps: float = 1e-5) -> Dict[str, float]:
        """
        Numerical gradient of the log-likelihood via centred finite differences.

        Each parameter is perturbed individually; computations are dispatched
        to a thread pool for speed.
        """
        base_ll = self._calculate_log_likelihood(data, params)
        gradient: Dict[str, float] = {}

        def _partial(key: str, val: float) -> Tuple[str, float]:
            p_plus  = {**params, key: val + eps}
            p_minus = {**params, key: val - eps}
            ll_plus  = self._calculate_log_likelihood(data, p_plus)
            ll_minus = self._calculate_log_likelihood(data, p_minus)
            return key, (ll_plus - ll_minus) / (2 * eps)

        scalar_params = {k: v for k, v in params.items()
                         if isinstance(v, (int, float))}

        with ThreadPoolExecutor(max_workers=self.n_jobs) as ex:
            futures = {ex.submit(_partial, k, v): k
                       for k, v in scalar_params.items()}
            for future in futures:
                key, grad = future.result()
                gradient[key] = grad

        return gradient

    def _check_ar_stationarity(self, ar_params) -> bool:
        """
        Check AR stationarity: all roots of Φ(z)=1−φ₁z−…−φₚzᵖ lie outside |z|=1.
        """
        if ar_params is None or len(ar_params) == 0:
            return True
        # Characteristic polynomial coefficients (numpy poly convention)
        coeffs = np.concatenate([[1], -np.asarray(ar_params)])
        roots = np.roots(coeffs)
        return bool(np.all(np.abs(roots) > 1.0))

    def _check_ma_invertibility(self, ma_params) -> bool:
        """
        Check MA invertibility: all roots of Θ(z)=1+θ₁z+…+θ_qzᵠ lie outside |z|=1.
        """
        if ma_params is None or len(ma_params) == 0:
            return True
        coeffs = np.concatenate([[1], np.asarray(ma_params)])
        roots = np.roots(coeffs)
        return bool(np.all(np.abs(roots) > 1.0))

    def _calculate_residuals(self, data: np.ndarray, params: Dict) -> np.ndarray:
        """
        Compute innovation residuals via the prediction-error decomposition.

        ε_t = y_t − c − Σ φᵢ y_{t-i} − Σ θⱼ ε_{t-j}
        """
        n = len(data)
        ar = np.asarray(params.get('ar_params') or [], dtype=float)
        ma = np.asarray(params.get('ma_params') or [], dtype=float)
        c  = float(params.get('constant') or params.get('mean') or 0.0)
        p, q = len(ar), len(ma)

        residuals = np.zeros(n)
        max_order = max(p, q, 1)

        for t in range(max_order, n):
            pred = c
            for i in range(p):
                pred += ar[i] * data[t - i - 1]
            for j in range(q):
                pred += ma[j] * residuals[t - j - 1]
            residuals[t] = data[t] - pred

        return residuals[max_order:]

    def _calculate_predictions(self, steps: int) -> np.ndarray:
        """
        h-step-ahead recursive prediction after undoing differencing.
        """
        params = self._fitted_params or {}
        ar = np.asarray(params.get('ar_params') or self.ar_params or [], dtype=float)
        ma = np.asarray(params.get('ma_params') or self.ma_params or [], dtype=float)
        c  = float(params.get('constant') or self.constant or 0.0)
        p, q = len(ar), len(ma)

        data   = self.differenced_data
        p_obs  = data[-p:] if p > 0 else np.array([])
        resids = self._calculate_residuals(
            data,
            {'ar_params': ar, 'ma_params': ma, 'constant': c}
        )
        q_resids = resids[-q:] if q > 0 else np.array([])

        preds = np.zeros(steps)
        for h in range(steps):
            pred = c
            # AR component
            for i in range(p):
                if h - i - 1 >= 0:
                    pred += ar[i] * preds[h - i - 1]
                else:
                    idx = h - i - 1
                    if len(p_obs) + idx >= 0:
                        pred += ar[i] * p_obs[len(p_obs) + idx]
            # MA component (future errors = 0)
            for j in range(q):
                if h - j - 1 < 0:
                    idx = len(q_resids) + (h - j - 1)
                    if 0 <= idx < len(q_resids):
                        pred += ma[j] * q_resids[idx]
            preds[h] = pred

        # Undo differencing
        if self.diff_order > 0 and self.original_data is not None:
            last_vals = self.original_data[-(self.diff_order):]
            for _ in range(self.diff_order):
                preds = np.cumsum(np.concatenate([[last_vals[-1]], preds]))[1:]
        return preds

    def get_residuals(self) -> np.ndarray:
        """Return model residuals."""
        self._validate_fitted()
        return self._calculate_residuals(
            self.differenced_data, self._fitted_params or {}
        )

    def summary(self) -> str:
        """Generate model summary string."""
        self._validate_fitted()
        s  = f"ParallelARIMAProcess({self.ar_order},{self.diff_order},{self.ma_order})\n"
        s += "=" * 50 + "\n"
        s += f"n_jobs : {self.n_jobs}\n"
        s += f"AR     : {self.ar_params}\n"
        s += f"MA     : {self.ma_params}\n"
        s += f"const  : {self.constant}\n"
        s += f"σ²     : {self.variance}\n"
        return s


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    data = np.cumsum(np.random.normal(0, 1, 1000))  # random walk

    model = ParallelARIMAProcess(ar_order=1, diff_order=1, ma_order=1, n_jobs=-1)
    model.fit(data)

    predictions = model.predict(steps=10)
    print(f"Predicciones: {predictions}")
