import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

# Dados extraídos da imagem
mean_returns = np.array([0.270, 0.144, -0.030, 0.099, 0.020, 0.122, 0.016, 0.070, 0.090, 0.076])
cov_matrix = np.array([
    [0.001802, 0.000421, 0.000218, 0.000442, 0.000180, 0.000242, 0.000224, 0.000225, 0.000247, 0.000147],
    [0.000421, 0.001018, 0.000301, 0.000441, 0.000229, 0.000304, 0.000245, 0.000265, 0.000286, 0.000151],
    [0.000218, 0.000301, 0.000227, 0.000327, 0.000142, 0.000196, 0.000187, 0.000174, 0.000187, 0.000098],
    [0.000442, 0.000441, 0.000327, 0.000535, 0.000228, 0.000308, 0.000317, 0.000300, 0.000324, 0.000182],
    [0.000180, 0.000229, 0.000142, 0.000228, 0.000133, 0.000170, 0.000156, 0.000156, 0.000165, 0.000089],
    [0.000242, 0.000304, 0.000196, 0.000308, 0.000170, 0.000274, 0.000219, 0.000212, 0.000235, 0.000115],
    [0.000224, 0.000245, 0.000187, 0.000317, 0.000156, 0.000219, 0.000240, 0.000197, 0.000225, 0.000109],
    [0.000225, 0.000265, 0.000174, 0.000300, 0.000156, 0.000212, 0.000197, 0.000250, 0.000229, 0.000121],
    [0.000247, 0.000286, 0.000187, 0.000324, 0.000165, 0.000235, 0.000225, 0.000229, 0.000309, 0.000124],
    [0.000147, 0.000151, 0.000098, 0.000182, 0.000089, 0.000115, 0.000109, 0.000121, 0.000124, 0.000094]
])
risk_free_rate = 0.0425

# Número de ativos
num_assets = len(mean_returns)

# Pesos do portfólio ingênuo
weights_ingenious = np.ones(num_assets) / num_assets

# Função para calcular o retorno e a variância de um portfólio
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std

# Função para calcular o portfólio de mínima variância
def min_variance_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (cov_matrix,)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(lambda weights, *args: portfolio_performance(weights, mean_returns, cov_matrix)[1], 
                      num_assets * [1. / num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Portfólio tangente (máxima razão de Sharpe)
def tangency_portfolio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
        p_returns, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
        return -(p_returns - risk_free_rate) / p_std
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(neg_sharpe_ratio, num_assets * [1. / num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Função para calcular a Fronteira Eficiente
def efficient_frontier(mean_returns, cov_matrix, num_points):
    results = np.zeros((3, num_points))
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(mean_returns)))
    
    target_returns = np.linspace(0, 0.3, num_points)
    for i, ret in enumerate(target_returns):
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x, ret=ret: portfolio_performance(x, mean_returns, cov_matrix)[0] - ret}
        )
        result = minimize(lambda weights, *args: portfolio_performance(weights, mean_returns, cov_matrix)[1], 
                          len(mean_returns) * [1. / len(mean_returns),], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        results[0, i] = ret
        results[1, i] = result.fun
    return results

# Função para calcular o Índice de Sharpe
def sharpe_ratio(returns, std, risk_free_rate):
    return (returns - risk_free_rate) / std

# Função para calcular o Value at Risk (VaR)
def value_at_risk(returns, std, confidence_level=0.05):
    return norm.ppf(1 - confidence_level, returns, std)

# Função para calcular o beta
def beta(weights, cov_matrix, market_cov):
    portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
    return portfolio_var / market_cov

# Função para calcular o Índice de Treynor
def treynor_ratio(returns, risk_free_rate, beta):
    return (returns - risk_free_rate) / beta

# Função para calcular o risco diversificável
def diversifiable_risk(weights, cov_matrix):
    portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
    market_var = np.var(np.sum(weights * mean_returns))
    return portfolio_var - market_var

if __name__ == "__main__":
    # Calculando os pesos dos portfólios
    weights_min_variance = min_variance_portfolio(mean_returns, cov_matrix)
    weights_tangency = tangency_portfolio(mean_returns, cov_matrix, risk_free_rate)

    # Calculando o desempenho dos portfólios
    performance_ingenious = portfolio_performance(weights_ingenious, mean_returns, cov_matrix)
    performance_min_variance = portfolio_performance(weights_min_variance, mean_returns, cov_matrix)
    performance_tangency = portfolio_performance(weights_tangency, mean_returns, cov_matrix)

    # Calculando métricas adicionais
    metrics = {}
    for name, weights, performance in zip(['Ingênuo', 'Mínima Variância', 'Tangente'], 
                                           [weights_ingenious, weights_min_variance, weights_tangency], 
                                           [performance_ingenious, performance_min_variance, performance_tangency]):
        returns, std = performance
        var = std ** 2
        sharpe = sharpe_ratio(returns, std, risk_free_rate)
        var_95 = value_at_risk(returns, std, confidence_level=0.05)
        portfolio_beta = beta(weights, cov_matrix, np.var(mean_returns))
        treynor = treynor_ratio(returns, risk_free_rate, portfolio_beta)
        diversifiable = diversifiable_risk(weights, cov_matrix)
        
        metrics[name] = {
            'Retorno Médio': returns,
            'Mediana': np.median(mean_returns),
            'Desvio Padrão': std,
            'Variância': var,
            'Índice de Sharpe': sharpe,
            'VaR (95%)': var_95,
            'Beta': portfolio_beta,
            'Índice de Treynor': treynor,
            'Risco Diversificável': diversifiable
        }

    # Exibindo os resultados
    for name, metrics_values in metrics.items():
        print(f"\n{name}:\n")
        for metric_name, value in metrics_values.items():
            if isinstance(value, float):
                print(f"{metric_name}: {value:.3f}")
            else:
                print(f"{metric_name}: {value}")

    # Calculando a Fronteira Eficiente
    num_points = 100
    results = efficient_frontier(mean_returns, cov_matrix, num_points)
    returns, std_devs = results[0, :], results[1, :]

    # Plotando a Fronteira Eficiente com detalhes
    plt.figure(figsize=(14, 8))
    plt.plot(std_devs, returns, label='Fronteira Eficiente', color='b')
    
    # Marcando portfólios específicos
    plt.scatter(performance_ingenious[1], performance_ingenious[0], color='r', marker='o', s=100, label='Portfólio Ingênuo')
    plt.scatter(performance_min_variance[1], performance_min_variance[0], color='g', marker='x', s=100, label='Portfólio de Mínima Variância')
    plt.scatter(performance_tangency[1], performance_tangency[0], color='m', marker='*', s=100, label='Portfólio Tangente')
    
    # Marcando os pontos dos ativos individuais
    for i, (ret, std) in enumerate(zip(mean_returns, np.sqrt(np.diag(cov_matrix)))):
        plt.scatter(std, ret, marker='D', s=100)
        plt.text(std, ret, f"Ativo {i+1}", fontsize=12, ha='right')

    # Adicionando detalhes dos dados
    plt.title('Fronteira Eficiente com Detalhes dos Portfólios e Ativos')
    plt.xlabel('Desvio Padrão (Risco)')
    plt.ylabel('Retorno Esperado')
    plt.legend(loc='upper left')
    plt.grid(True)
    
    # Adicionando anotação para cada portfólio
    plt.annotate('Portfólio Ingênuo', xy=(performance_ingenious[1], performance_ingenious[0]), 
                 xytext=(performance_ingenious[1]+0.005, performance_ingenious[0]-0.02),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
    
    plt.annotate('Portfólio de Mínima Variância', xy=(performance_min_variance[1], performance_min_variance[0]), 
                 xytext=(performance_min_variance[1]+0.005, performance_min_variance[0]-0.02),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
    
    plt.annotate('Portfólio Tangente', xy=(performance_tangency[1], performance_tangency[0]), 
                 xytext=(performance_tangency[1]+0.005, performance_tangency[0]-0.02),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
    
    plt.show()
