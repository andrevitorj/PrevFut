"""Configurações do PrevFut.

Este módulo contém todas as configurações e constantes usadas no projeto.
"""

from typing import Dict

# Configurações da API
BASE_URL = "https://v3.football.api-sports.io"
REQUEST_LIMIT_PER_MINUTE = 100
REQUEST_INTERVAL = 60 / REQUEST_LIMIT_PER_MINUTE

# Configurações de Cache
CACHE_FILE = "api_cache.json"
CACHE_TTL = 3600  # 1 hora em segundos

# Mapeamento de estatísticas
ESTATISTICAS_MODELO: Dict[str, str] = {
    "shots_on_goal": "Chutes no Gol",
    "total_shots": "Finalizações Totais",
    "ball_possession": "Posse de Bola (%)",
    "corners": "Escanteios",
    "fouls": "Faltas",
    "yellow_cards": "Cartões Amarelos",
    "red_cards": "Cartões Vermelhos",
    "gols": "Gols",
    "offsides": "Impedimentos"
}

# Pesos por competição
COMPETICAO_PESOS: Dict[str, float] = {
    "UEFA Champions League": 1.0,
    "La Liga": 0.85,
    "Premier League": 0.85,
    "Serie A": 0.8,
    "Bundesliga": 0.8,
    "CONMEBOL Libertadores": 0.75,
    "Ligue 1": 0.75,
    "Série A": 0.7,
    "Série B": 0.55,
    "Série C": 0.4,
    "Copa do Brasil": 0.6,
    "Paranaense - 1": 0.5,
    "Copa del Rey": 0.7,
    "Estadual": 0.5,
    "FA Cup": 0.7,
    "Coupe de France": 0.5,
    "Desconhecida": 0.5
}

# Thresholds para análise de apostas
THRESHOLD_VALOR = 0.1  # Diferença mínima para considerar uma oportunidade
MIN_PROBABILIDADE = 0.15  # Probabilidade mínima para considerar uma aposta
MAX_ODD = 10.0  # Odd máxima para considerar
