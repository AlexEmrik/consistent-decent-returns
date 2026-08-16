"""Factor-based portfolio optimizer."""

from typing import Literal

import cvxpy as cp
import numpy as np


def construct_portfolio(
    alphas: np.ndarray,
    curr_weights: np.ndarray,
    F: np.ndarray,
    D_half: np.ndarray,
    Sigma: np.ndarray,
    universe: np.ndarray,
    max_weights: np.array | None = None,
    ffr: float = 0.0,
    bid_ask_spread: float = 5e-4,
    t_lim: float = 1.0,
    asset_t_lim: float = 1.0,
    target_vol: float = 0.01,
    min_weights: float = 0.0,
    kappa_risk: float = 0.02,
    gamma_risk: float = 1e1,
    first_solve: bool = False,
    asset_dd_limit: float = None,  
    method: Literal["markowitz70", "fixed_weight", "fixed_weight_vol_control", "risk_parity"] = "markowitz70",
) -> np.ndarray:
    """Construct a portfolio.

    Args:
        alphas (np.ndarray): Expected returns for each asset.
        curr_weights (np.ndarray): Current weights of the portfolio.
        F (np.ndarray): Factor loadings matrix (n_assets, n_factors).
        D_half (np.ndarray): Diagonal half of the idiosyncratic covariance (n_assets,).
        universe (np.ndarray): Boolean mask of investable assets.
        ffr (float, optional): Fed Funds Rate. Defaults to 0.0.
        bid_ask_spread (float, optional): Bid-ask spread. Defaults to 5e-4.
        t_lim (float, optional): Transaction limit. Defaults to 1.0.
        target_vol (float, optional): Target volatility. Defaults to 0.01.
        max_leverage (float, optional): Maximum leverage. Defaults to 1.5.
        min_weights (float, optional): Minimum weights. Defaults to 0.0.
        max_weights (float, optional): Maximum weights. Defaults to 1.0.
        kappa_risk (float, optional): Risk aversion parameter. Defaults to 0.02.
        gamma_risk (float, optional): Risk aversion parameter. Defaults to 1e1.
        first_solve (bool, optional): Whether the portfolio has not been initialized.
            Defaults to False.
        method (Literal["markowitz70", "fixed_weight", "fixed_weight_vol_control"], optional):
            Method to use for optimization. Defaults to "markowitz70".

    Raises:
        ValueError: If the method is invalid.

    Returns:
        np.ndarray: Weights for each asset. The sum of the weights and cash should be 1.
    """
    if max_weights is None:
        max_weights = np.ones(len(alphas), dtype=float)

    match method:
        case "markowitz70":
            return markowitz70_portfolio(
                alphas=alphas,
                curr_weights=curr_weights,
                F=F,
                D_half=D_half,
                Sigma=Sigma,
                universe=universe,
                ffr=ffr,
                bid_ask_spread=bid_ask_spread,
                t_lim=t_lim,
                asset_t_lim = asset_t_lim,
                target_vol=target_vol,
                min_weights=min_weights,
                max_weights=max_weights,
                gamma_risk=gamma_risk,
                first_solve=first_solve,
            )
        case "fixed_weight":
            return fixed_weight_portfolio(alphas=alphas, universe=universe)
        case "fixed_weight_vol_control":
            return fixed_weight_vol_control_portfolio(
                alphas=alphas,
                F=F,
                D_half=D_half,
                Sigma=Sigma,
                universe=universe,
                target_vol=target_vol,
            )
        case "markowitz_hard":
            return markowitz(
                alphas=alphas,
                F=F,
                D_half=D_half,
                Sigma=Sigma,
                universe=universe,
                max_weights=max_weights,
                target_vol=target_vol,
                hard_constraint=True,
            )
        case "markowitz_soft":
            return markowitz(
                alphas=alphas,
                F=F,
                D_half=D_half,
                Sigma=Sigma,
                universe=universe,
                max_weights=max_weights,
                gamma_risk=gamma_risk,
                hard_constraint=False,
            )
        case "inv_vol":
            return inv_vol(
                F=F,
                D_half=D_half,
                Sigma=Sigma,
                universe=universe,
                target_vol=target_vol
            )
        case "risk_parity":
            return risk_parity(
                F=F,
                D_half=D_half,
                Sigma=Sigma,
                universe=universe,
                target_vol=target_vol,
                kappa=kappa_risk
            )
        case "min_vol":
            return min_vol(
                F=F,
                D_half=D_half,
                Sigma=Sigma,
                universe=universe,
                target_vol=target_vol,
            )
        case "max_diversification":
            return max_diversification(
                F=F,
                D_half=D_half,
                Sigma=Sigma,
                universe=universe,
                target_vol=target_vol,
            )
        
        case _:
            raise ValueError(f"Invalid method: {method}")


def fixed_weight_portfolio(alphas: np.ndarray, universe: np.ndarray) -> np.ndarray:
    U = len(universe)
    N = np.sum(universe)

    if N == 0: 
        return np.zeros(U)

    return (alphas * universe) / alphas[universe].sum()


def fixed_weight_vol_control_portfolio(
    alphas: np.ndarray,
    F: np.ndarray,
    D_half: np.ndarray,
    Sigma: np.ndarray,
    universe: np.ndarray,
    target_vol: float,
) -> np.ndarray:
    U = len(universe)
    N = np.sum(universe)

    if N == 0: 
        return np.zeros(U)
    
    weights = (alphas * universe) / alphas[universe].sum()
    weights_scaled = vol_scale(weights=weights, F=F, D_half=D_half, Sigma=Sigma, target_vol=target_vol)
    return weights_scaled


def markowitz(
    alphas: np.ndarray,
    F: np.ndarray,
    D_half: np.ndarray,
    Sigma: np.ndarray,
    universe: np.ndarray,
    max_weights = None,
    target_vol: float = None, # hard
    gamma_risk: float = None, # soft
    hard_constraint: bool = True,
) -> np.ndarray:
    U = len(universe)
    N = sum(universe)

    if N == 0:
        return np.zeros(len(universe))
    
    weights = cp.Variable(U)
    mask = (~universe).astype(float)
    
    constraints = [
        cp.sum(weights) <= 1.0,
        weights >= 0.0,
        cp.multiply(weights, mask) == 0.0,
        weights <= (max_weights if max_weights is not None else np.ones(U))
    ]

    if F is not None or D_half is not None:
        sigma_sqrt = np.vstack([F.T, np.diag(D_half)])
        risk_term = cp.norm2(sigma_sqrt @ weights)
    elif Sigma is not None:
        L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(Sigma.shape[0]))
        risk_term = cp.norm2(L.T @ weights)
    else:
        raise ValueError()

    if hard_constraint == True:
        objective = cp.Maximize(cp.scalar_product(alphas, weights))
        constraints.append(risk_term <= target_vol)
    else:
        objective = cp.Maximize(cp.scalar_product(alphas, weights) - gamma_risk * risk_term**2)

    problem = cp.Problem(objective=objective, constraints=constraints)
    problem.solve(solver=cp.CLARABEL, verbose=False)
    return weights.value


def markowitz70_portfolio(
    alphas: np.ndarray,
    curr_weights: np.ndarray,
    F: np.ndarray,
    D_half: np.ndarray,
    Sigma: np.ndarray,
    universe: np.ndarray,
    max_weights: np.ndarray,
    target_vol: float,
    ffr: float = 0.0,
    bid_ask_spread: float = 5e-4,
    t_lim: float = 1.0,
    asset_t_lim: float = 1.0,
    min_weights: float = 0.0,
    gamma_risk: float = 1e1,
    first_solve: bool = False,
) -> np.ndarray:
    n_assets = len(alphas)
    curr_cash = float(1 - curr_weights.sum())

    mask = (~universe).astype(float)

    weights = cp.Variable(n_assets)
    cash = cp.Variable(nonneg=True)
    risk_slack = cp.Variable(nonneg=True)
    turnover_slack = cp.Variable(nonneg=True)

    expected_returns = cp.scalar_product(alphas, weights) + (cash * ffr)
    t_cost = bid_ask_spread * cp.norm1(weights - curr_weights)

    if F is not None or D_half is not None:
        sigma_sqrt = np.vstack([F.T, np.diag(D_half)])
        # sigma_diag = np.sum(F**2, axis=1) + D_half**2
        risk_term = cp.norm2(sigma_sqrt @ weights)
    elif Sigma is not None:
        L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(Sigma.shape[0]))
        risk_term = cp.norm2(L.T @ weights)
        # sigma_diag = ...  # FIXME if added back in code
    else:
        raise ValueError()
    
    factor_risk = cp.norm2(risk_term)
    double_turnover = cp.norm1(weights - curr_weights) + cp.abs(cash - curr_cash)
    constraints = [
        cp.sum(weights) + cash == 1.0,
        weights >= min_weights,
        weights <= max_weights,
        factor_risk <= target_vol + risk_slack,
        cp.multiply(weights, mask) == 0.0,
    ]

    objective = expected_returns - (gamma_risk * risk_slack)
    if not first_solve:
        objective -= t_cost
        constraints += [
            double_turnover <= 2 * t_lim,
            cp.abs(weights - curr_weights) <= asset_t_lim]

    problem = cp.Problem(cp.Maximize(objective), constraints)

    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)
    except (cp.SolverError, Exception):
        return np.zeros(n_assets)

    return weights.value


def risk_parity(
        F: np.darray,
        D_half: np.ndarray,
        Sigma: np.ndarray,
        universe: np.ndarray,
        target_vol: float,
        kappa: float
) -> np.ndarray:
    U = len(universe)
    N = sum(universe)
    weights = np.zeros(U)

    if N == 0:
        return weights
    
    w = cp.Variable(N, pos=True)
    
    if F is not None or D_half is not None:
        F_n = F[universe]
        D_half_n = D_half[universe]
        sigma_sqrt = np.vstack([F_n.T, np.diag(D_half_n)])
        risk_term = cp.sum_squares(sigma_sqrt @ w)
    elif Sigma is not None:
        Sigma_n = Sigma[np.ix_(universe, universe)]
        risk_term = cp.quad_form(w, Sigma_n)
    else:
        raise ValueError()

    # kappa = 1/N
    prob = cp.Problem(
        objective=cp.Minimize(0.5 * risk_term - kappa * cp.sum(cp.log(w))),
        constraints=[
            cp.sum(w) == 1.0,
            
        ]
    )
    prob.solve(solver=cp.CLARABEL, verbose=False)

    weights[universe] = w.value
    weights_scaled = vol_scale(weights=weights, F=F, D_half=D_half, Sigma=Sigma, target_vol=target_vol)
    return weights_scaled


def min_vol(F: np.ndarray, D_half: np.ndarray, Sigma: np.ndarray, universe: np.ndarray, target_vol: float) -> np.ndarray:
    U = len(universe)
    weights = np.zeros(U)
    N = np.sum(universe)
    if N == 0:
        return weights
    w = cp.Variable(N, nonneg=True)
    
    if F is not None or D_half is not None:
        F_n = F[universe] 
        D_half_n = D_half[universe]
        Sigma_n = F_n @ F_n.T + np.diag(D_half_n**2)
    elif Sigma is not None:
        Sigma_n = Sigma[np.ix_(universe, universe)]
    else:
        raise ValueError()
    
    risk_term = cp.quad_form(w, cp.psd_wrap(Sigma_n))
    prob = cp.Problem(
        objective = cp.Minimize(risk_term),
        constraints=[cp.sum(w) == 1]
    )
    prob.solve(solver=cp.CLARABEL, verbose=False)

    weights[universe] = w.value
    weights_scaled = vol_scale(weights=weights, F=F, D_half=D_half, Sigma=Sigma, target_vol=target_vol)
    return weights_scaled


def max_diversification(F: np.ndarray, D_half: np.ndarray, Sigma: np.ndarray, universe: np.ndarray, target_vol: float):
    U = len(universe)
    weights = np.zeros(U)
    N = np.sum(universe)
    if N == 0:
        return weights

    if F is not None or D_half is not None:
        F_n = F[universe] 
        D_half_n = D_half[universe]
        Sigma_n = F_n @ F_n.T + np.diag(D_half_n**2)
    elif Sigma is not None:
        Sigma_n = Sigma[np.ix_(universe, universe)]
    else:
        raise ValueError()
    
    sigma_vec = np.clip(np.sqrt(np.diag(Sigma_n)), 1e-8, 3)          
    C = Sigma_n / np.outer(sigma_vec, sigma_vec)
    
    y = cp.Variable(N, nonneg=True)
    prob = cp.Problem(
        objective = cp.Minimize(cp.quad_form(y, cp.psd_wrap(C))),
        constraints=[
            cp.sum(y) == 1, 
        ]
    )
    prob.solve(solver=cp.CLARABEL, verbose=False)
    w = y.value / sigma_vec
    w /= np.sum(w)
    
    weights[universe] = w
    weights_scaled = vol_scale(weights=weights, F=F, D_half=D_half, Sigma=Sigma, target_vol=target_vol)
    return weights_scaled


def inv_vol(F: np.ndarray, D_half: np.ndarray, Sigma: np.ndarray, universe: np.ndarray, target_vol: float):
    U = len(universe)
    weights = np.zeros(U)
    N = np.sum(universe)
    if N == 0:
        return weights
    
    if F is not None or D_half is not None:
        F_n = F[universe] 
        D_half_n = D_half[universe]
        Sigma_n = F_n @ F_n.T + np.diag(D_half_n**2)
    elif Sigma is not None:
        Sigma_n = Sigma[np.ix_(universe, universe)]
    else:
        raise ValueError()

    sigma_vec = np.clip(np.sqrt(np.diag(Sigma_n)), 1e-8, 3)        
    sigma_vec_inv = 1/sigma_vec

    w = sigma_vec_inv / sigma_vec_inv.sum()
    
    weights[universe] = w

    weights_scaled = vol_scale(weights=weights, F=F, D_half=D_half, Sigma=Sigma, target_vol=target_vol)
    return weights_scaled

def vol_scale(
        weights: np.ndarray,
        F: np.ndarray,
        D_half: np.ndarray,
        Sigma: np.ndarray,
        target_vol: float,
) -> np.ndarray:
    if F is not None or D_half is not None:
        f = np.concatenate([F.T @ weights, D_half * weights])
        total_vol = np.linalg.norm(f)
    elif Sigma is not None:
        total_vol = np.sqrt(weights.T @ Sigma @ weights).item()
    else:
        raise ValueError()
    scaling = np.clip(target_vol / (total_vol + 1e-8), 0.0, 1.0)
    return weights * scaling