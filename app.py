import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
from scipy.stats import poisson, norm
import openpyxl
from datetime import datetime
import os
from dotenv import load_dotenv
import logging

# Configurar logging para depuração (não será exibido na interface)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente
load_dotenv()
API_KEY = os.getenv("API_FOOTBALL_KEY")

# Configuração da API
API_BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {
    "x-apisports-key": API_KEY
}

# Verificar chave da API
if not API_KEY:
    st.error("Chave da API não encontrada. Configure 'API_FOOTBALL_KEY' nos secrets do Streamlit.")
    st.stop()

# Função para testar a chave da API
def test_api_key():
    url = f"{API_BASE_URL}/status"
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            return True
        else:
            st.error(f"Erro ao testar chave: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        st.error(f"Erro ao testar chave: {str(e)}")
        return False

# Carregar pesos das competições com base nos IDs
def load_weights():
    try:
        with open("pesos.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "CONMEBOL Libertadores": 0.75,  # ID 13
            "UEFA Champions League": 1.0,   # ID 2
            "UEFA Europa Conference League": 0.65,  # ID 848
            "UEFA Europa League": 0.85,     # ID 3
            "Serie A (Brazil)": 0.7,        # ID 71
            "Serie B (Brazil)": 0.6,        # ID 72
            "Serie C (Brazil)": 0.5,        # ID 75
            "Liga Profesional Argentina": 0.65,  # ID 128
            "Copa do Brasil": 0.65,         # ID 73
            "Premier League": 0.85,         # ID 39
            "La Liga": 0.8,                 # ID 140
            "Bundesliga": 0.8,              # ID 78
            "Serie A (Italy)": 0.8,         # ID 135
            "Ligue 1": 0.75,                # ID 61
            "Primeira Liga": 0.7,           # ID 94
            "FIFA Club World Cup": 0.8,     # ID 15
            "Outras ligas não mapeadas": 0.5
        }

# Mapeamento de IDs de ligas para nomes usados nos pesos
LEAGUE_MAPPING = {
    13: "CONMEBOL Libertadores",
    2: "UEFA Champions League",
    848: "UEFA Europa Conference League",
    3: "UEFA Europa League",
    71: "Serie A (Brazil)",
    72: "Serie B (Brazil)",
    75: "Serie C (Brazil)",
    128: "Liga Profesional Argentina",
    73: "Copa do Brasil",
    39: "Premier League",
    140: "La Liga",
    78: "Bundesliga",
    135: "Serie A (Italy)",
    61: "Ligue 1",
    94: "Primeira Liga",
    15: "FIFA Club World Cup"
}

# Função para buscar times
def search_team(team_name):
    if not API_KEY:
        st.error("Chave da API não configurada.")
        return []
    url = f"{API_BASE_URL}/teams"
    params = {"search": team_name}
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code == 200:
            return response.json().get("response", [])
        else:
            st.error(f"Erro na API: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"Erro ao buscar times: {str(e)}")
        return []

# Função para buscar jogos passados
def get_team_games(team_id, season, home=True, limit=20):
    if not API_KEY:
        st.error("Chave da API não configurada.")
        return []
    url = f"{API_BASE_URL}/fixtures"
    params = {
        "team": team_id,
        "season": season,
        "last": limit,
        "status": "FT"  # Apenas jogos finalizados
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code == 200:
            data = response.json()
            games = data.get("response", [])
            # Filtrar jogos como mandante ou visitante
            filtered_games = [
                game for game in games
                if (home and game["teams"]["home"]["id"] == team_id) or
                   (not home and game["teams"]["away"]["id"] == team_id)
            ]
            # Fallback para temporada anterior se não houver jogos
            if not filtered_games and season == 2025:
                st.warning(f"Nenhum jogo finalizado encontrado para temporada {season}. Tentando temporada {season-1}...")
                params["season"] = season - 1
                response = requests.get(url, headers=HEADERS, params=params)
                if response.status_code == 200:
                    data = response.json()
                    games = data.get("response", [])
                    filtered_games = [
                        game for game in games
                        if (home and game["teams"]["home"]["id"] == team_id) or
                           (not home and game["teams"]["away"]["id"] == team_id)
                    ]
            return filtered_games
        st.error(f"Erro ao buscar jogos: {response.status_code} - {response.text}")
        return []
    except Exception as e:
        st.error(f"Erro ao buscar jogos: {str(e)}")
        return []

# Função para buscar o próximo jogo entre dois times
def find_next_fixture(team_a_id, team_b_id, season):
    if not API_KEY:
        st.error("Chave da API não configurada.")
        return None
    url = f"{API_BASE_URL}/fixtures"
    params = {
        "team": team_a_id,
        "season": season,
        "next": 20  # Aumentar para buscar mais jogos futuros
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        logger.debug(f"Resposta da API para buscar fixture: {response.status_code} - {response.text}")
        if response.status_code == 200:
            data = response.json()
            games = data.get("response", [])
            for game in games:
                if (game["teams"]["home"]["id"] == team_a_id and game["teams"]["away"]["id"] == team_b_id) or \
                   (game["teams"]["home"]["id"] == team_b_id and game["teams"]["away"]["id"] == team_a_id):
                    fixture_id = game["fixture"]["id"]
                    logger.debug(f"Fixture encontrado: {fixture_id} para {team_a_id} vs {team_b_id}")
                    return fixture_id
            logger.debug(f"Nenhum jogo futuro encontrado entre {team_a_id} e {team_b_id} na temporada {season}")
            st.warning("Nenhum jogo futuro encontrado entre os times selecionados na temporada especificada.")
            return None
        st.error(f"Erro ao buscar próximo jogo: {response.status_code} - {response.text}")
        return None
    except Exception as e:
        st.error(f"Erro ao buscar próximo jogo: {str(e)}")
        return None

# Função para buscar estatísticas de um jogo
def get_game_stats(fixture_id):
    if not API_KEY:
        st.error("Chave da API não configurada.")
        return []
    url = f"{API_BASE_URL}/fixtures/statistics"
    params = {"fixture": fixture_id}
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code == 200:
            return response.json().get("response", [])
        st.error(f"Erro ao buscar estatísticas: {response.status_code} - {response.text}")
        return []
    except Exception as e:
        st.error(f"Erro ao buscar estatísticas: {str(e)}")
        return []

# Função para buscar competições de um time
def get_team_leagues(team_id, season):
    if not API_KEY:
        st.error("Chave da API não configurada.")
        return []
    url = f"{API_BASE_URL}/leagues"
    params = {"team": team_id, "season": season}
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code == 200:
            return response.json().get("response", [])
        st.error(f"Erro ao buscar ligas: {response.status_code} - {response.text}")
        return []
    except Exception as e:
        st.error(f"Erro ao buscar ligas: {str(e)}")
        return []

# Função para calcular médias (feitas e sofridas) com número de partidas
def calculate_averages(games, team_id, weights):
    stats = {
        "goals_scored": [], "goals_conceded": [],
        "shots": [], "shots_conceded": [],
        "shots_on_target": [], "shots_on_target_conceded": [],
        "corners": [], "corners_conceded": [],
        "possession": [], "possession_conceded": [],
        "offsides": [], "offsides_conceded": [],
        "fouls_committed": [], "fouls_suffered": [],
        "yellow_cards": [], "yellow_cards_conceded": [],
        "red_cards": [], "red_cards_conceded": [],
        "passes_accurate": [], "passes_accurate_conceded": [],
        "passes_missed": [], "passes_missed_conceded": [],
        "xg": [], "xga": [],
        "free_kicks": [], "free_kicks_conceded": []
    }
    weighted_values = {k: [] for k in stats.keys()}
    game_weights = {k: [] for k in stats.keys()}
    game_counts = {k: 0 for k in stats.keys()}

    for game in games:
        game_stats = get_game_stats(game["fixture"]["id"])
        team_data = {k: 0 for k in stats.keys()}
        is_home = game["teams"]["home"]["id"] == team_id
        team_data["goals_scored"] = game["goals"]["home" if is_home else "away"] or 0
        team_data["goals_conceded"] = game["goals"]["away" if is_home else "home"] or 0

        if game_stats:
            team_stats = next((s for s in game_stats if s["team"]["id"] == team_id), None)
            opponent_stats = next((s for s in game_stats if s["team"]["id"] != team_id), None)
            if team_stats and opponent_stats:
                for stat in team_stats["statistics"]:
                    stat_type = stat["type"].lower()
                    value = stat["value"]
                    if value is None:
                        value = 0
                    elif stat_type == "ball possession" and isinstance(value, str):
                        value = value.replace("%", "").strip()
                        value = float(value) if value else 0.0
                    else:
                        try:
                            value = float(value) if value else 0.0
                        except (ValueError, TypeError):
                            value = 0.0
                    if stat_type == "total shots":
                        team_data["shots"] = value
                    elif stat_type == "shots on goal":
                        team_data["shots_on_target"] = value
                    elif stat_type == "corner kicks":
                        team_data["corners"] = value
                    elif stat_type == "ball possession":
                        team_data["possession"] = value
                    elif stat_type == "offsides":
                        team_data["offsides"] = value
                    elif stat_type == "fouls":
                        team_data["fouls_committed"] = value
                    elif stat_type == "yellow cards":
                        team_data["yellow_cards"] = value
                    elif stat_type == "red cards":
                        team_data["red_cards"] = value
                    elif stat_type == "passes accurate":
                        team_data["passes_accurate"] = value
                    elif stat_type == "passes":
                        team_data["passes_missed"] = (value or 0) - (team_data["passes_accurate"] or 0)
                    elif stat_type == "expected goals":
                        team_data["xg"] = value
                    elif stat_type == "free kicks":
                        team_data["free_kicks"] = value

                for stat in opponent_stats["statistics"]:
                    stat_type = stat["type"].lower()
                    value = stat["value"]
                    if value is None:
                        value = 0
                    elif stat_type == "ball possession" and isinstance(value, str):
                        value = value.replace("%", "").strip()
                        value = float(value) if value else 0.0
                    else:
                        try:
                            value = float(value) if value else 0.0
                        except (ValueError, TypeError):
                            value = 0.0
                    if stat_type == "total shots":
                        team_data["shots_conceded"] = value
                    elif stat_type == "shots on goal":
                        team_data["shots_on_target_conceded"] = value
                    elif stat_type == "corner kicks":
                        team_data["corners_conceded"] = value
                    elif stat_type == "ball possession":
                        team_data["possession_conceded"] = value
                    elif stat_type == "offsides":
                        team_data["offsides_conceded"] = value
                    elif stat_type == "fouls":
                        team_data["fouls_suffered"] = value
                    elif stat_type == "yellow cards":
                        team_data["yellow_cards_conceded"] = value
                    elif stat_type == "red cards":
                        team_data["red_cards_conceded"] = value
                    elif stat_type == "passes accurate":
                        team_data["passes_accurate_conceded"] = value
                    elif stat_type == "passes":
                        team_data["passes_missed_conceded"] = (value or 0) - (team_data["passes_accurate_conceded"] or 0)
                    elif stat_type == "expected goals":
                        team_data["xga"] = value
                    elif stat_type == "expected goals against":
                        team_data["xga"] = value
                    elif stat_type == "free kicks":
                        team_data["free_kicks_conceded"] = value

        league_id = game["league"]["id"]
        mapped_name = LEAGUE_MAPPING.get(league_id, "Outras ligas não mapeadas")
        weight = weights.get(mapped_name, 0.5)

        for key in stats:
            stats[key].append(team_data[key])
            if team_data[key] != 0 or (game_stats and team_stats and opponent_stats):
                game_counts[key] += 1

        for key in weighted_values:
            if "conceded" in key or "suffered" in key:
                weighted_values[key].append(team_data[key] / max(weight, 0.1))
                game_weights[key].append(1 / max(weight, 0.1))
            else:
                weighted_values[key].append(team_data[key] * weight)
                game_weights[key].append(weight)

    simple_averages = {k: np.mean(v) if v else 0 for k, v in stats.items()}
    weighted_averages = {}
    for key in weighted_values:
        weighted_sum = sum(weighted_values[key])
        total_weight = sum(game_weights[key]) if game_weights[key] else 1
        weighted_averages[key] = weighted_sum / total_weight if total_weight > 0 else 0
    return simple_averages, weighted_averages, game_counts

# Função para prever estatísticas (usando feitas e sofridas)
def predict_stats(team_a_simple, team_a_weighted, team_b_simple, team_b_weighted, team_a_counts, team_b_counts):
    if team_a_weighted["goals_scored"] == 0 or team_b_weighted["goals_scored"] == 0:
        st.warning("Não há dados suficientes de gols para previsões estatísticas confiáveis.")
        return {}, {}, {}, {}
    predicted_stats = {}
    confidence_intervals = {}
    predicted_counts = {}
    for stat in team_a_weighted.keys():
        opposite_stat = stat.replace("conceded", "shots").replace("suffered", "committed") if "conceded" in stat or "suffered" in stat else stat.replace("scored", "conceded").replace("committed", "suffered")
        if opposite_stat not in team_b_weighted:
            opposite_stat = stat
        a_pred = (team_a_weighted[stat] + team_b_weighted.get(opposite_stat, team_b_weighted[stat])) * 1.2 / 2 if "conceded" in stat or "suffered" in stat else (team_a_weighted[stat] + team_b_simple.get(stat, team_b_weighted[stat])) / 2
        b_pred = (team_b_weighted[stat] + team_a_weighted.get(opposite_stat, team_a_weighted[stat])) / 1.2 / 2 if "conceded" in stat or "suffered" in stat else (team_b_weighted[stat] + team_a_simple.get(stat, team_a_weighted[stat])) / 2
        predicted_stats[stat] = {"team_a": a_pred, "team_b": b_pred}
        a_values = [team_a_weighted[stat], team_b_weighted.get(opposite_stat, team_b_weighted[stat])]
        b_values = [team_b_weighted[stat], team_a_weighted.get(opposite_stat, team_a_weighted[stat])]
        a_std = np.std(a_values) if len(a_values) > 1 and None not in a_values else 0
        b_std = np.std(b_values) if len(b_values) > 1 and None not in b_values else 0
        z = norm.ppf(0.925)
        confidence_intervals[stat] = {
            "team_a": (a_pred - z * a_std / np.sqrt(2), a_pred + z * a_std / np.sqrt(2)),
            "team_b": (b_pred - z * b_std / np.sqrt(2), b_pred + z * b_std / np.sqrt(2))
        }
        predicted_counts[stat] = (team_a_counts[stat] + team_b_counts.get(opposite_stat, team_b_counts[stat])) // 2 if "conceded" in stat or "suffered" in stat else (team_a_counts[stat] + team_b_counts.get(stat, team_a_counts[stat])) // 2
    return predicted_stats, confidence_intervals, predicted_counts, predicted_counts

# Função para prever placar
def predict_score(team_a_weighted, team_b_weighted):
    if team_a_weighted["goals_scored"] == 0 or team_b_weighted["goals_scored"] == 0:
        st.warning("Não há dados suficientes de gols para prever o placar.")
        return {"score": "N/A", "probs": {"win": 0, "draw": 0, "loss": 0}, "ci": {"team_a": (0, 0), "team_b": (0, 0)}, "total_goals_prob": 0}
    lambda_a = (team_a_weighted["goals_scored"] + team_b_weighted["goals_conceded"]) * 1.2 / 2
    lambda_b = (team_b_weighted["goals_scored"] + team_a_weighted["goals_conceded"]) / 1.2 / 2
    max_goals = 10
    prob_matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            prob_matrix[i, j] = poisson.pmf(i, lambda_a) * poisson.pmf(j, lambda_b)
    most_likely_score = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
    win_prob = np.sum(prob_matrix[np.where(np.arange(max_goals)[:, None] > np.arange(max_goals)[None, :])])
    draw_prob = np.sum(np.diag(prob_matrix))
    loss_prob = np.sum(prob_matrix[np.where(np.arange(max_goals)[:, None] < np.arange(max_goals)[None, :])])
    ci_a = (poisson.ppf(0.075, lambda_a), poisson.ppf(0.925, lambda_a))
    ci_b = (poisson.ppf(0.075, lambda_b), poisson.ppf(0.925, lambda_b))
    total_goals = lambda_a + lambda_b
    over_2_5_prob = 1 - poisson.cdf(2, total_goals)
    return {
        "score": f"{most_likely_score[0]} x {most_likely_score[1]}",
        "probs": {"win": win_prob, "draw": draw_prob, "loss": loss_prob},
        "ci": {"team_a": ci_a, "team_b": ci_b},
        "total_goals_prob": over_2_5_prob
    }

# Função para buscar odds reais
def get_odds(fixture_id):
    if not API_KEY:
        st.error("Chave da API não configurada.")
        return []
    if not fixture_id:
        st.warning("Não foi possível encontrar um jogo futuro entre os times selecionados para buscar odds.")
        return []
    url = f"{API_BASE_URL}/odds"
    params = {
        "fixture": fixture_id,
        "bet": "1"  # Mercado 1x2 (Vitória/Empate/Derrota)
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        logger.debug(f"Resposta da API para odds (fixture {fixture_id}): {response.status_code} - {response.text}")
        if response.status_code == 200:
            data = response.json()
            odds = data.get("response", [])
            if not odds:
                st.warning("Nenhuma odd disponível para o jogo selecionado.")
            return odds
        st.error(f"Erro ao buscar odds: {response.status_code} - {response.text}")
        return []
    except Exception as e:
        st.error(f"Erro ao buscar odds: {str(e)}")
        return []

# Função para comparar odds e previsões
def compare_odds(predicted_stats, score_pred, odds):
    if not predicted_stats or score_pred["score"] == "N/A":
        st.warning("Não há previsões válidas para comparar com odds.")
        return []
    if not odds:
        st.warning("Nenhuma odd disponível para o jogo selecionado.")
        return []

    comparison_data = []
    btts_prob = poisson.pmf(1, predicted_stats["goals_scored"]["team_a"]) * poisson.pmf(1, predicted_stats["goals_scored"]["team_b"])
    over_2_5_prob = score_pred["total_goals_prob"]
    under_2_5_prob = 1 - over_2_5_prob

    for market in odds:
        market_name = market.get("name", "Desconhecido")
        bets = market.get("bets", [])
        if not bets:
            continue
        for bet in bets:
            bet_name = bet.get("name", "Desconhecido")
            values = bet.get("values", [])
            if not values:
                continue
            if bet_name in ["Match Winner", "Both Teams To Score", "Goals Over/Under"]:
                for value in values:
                    odd = float(value.get("odd", 0))
                    if odd <= 0:
                        continue
                    implied_prob = 1 / odd
                    predicted_prob = 0
                    if bet_name == "Match Winner":
                        if value.get("value") == "Home":
                            predicted_prob = score_pred["probs"]["win"]
                        elif value.get("value") == "Draw":
                            predicted_prob = score_pred["probs"]["draw"]
                        elif value.get("value") == "Away":
                            predicted_prob = score_pred["probs"]["loss"]
                    elif bet_name == "Both Teams To Score":
                        if value.get("value") == "Yes":
                            predicted_prob = btts_prob
                        elif value.get("value") == "No":
                            predicted_prob = 1 - btts_prob
                    elif bet_name == "Goals Over/Under" and value.get("value") in ["Over 2.5", "Under 2.5"]:
                        if value.get("value") == "Over 2.5":
                            predicted_prob = over_2_5_prob
                        elif value.get("value") == "Under 2.5":
                            predicted_prob = under_2_5_prob

                    comparison_data.append({
                        "market": market_name,
                        "bet": bet_name,
                        "value": value.get("value", "Desconhecido"),
                        "odd": odd,
                        "implied_prob": implied_prob * 100,
                        "predicted_prob": predicted_prob * 100
                    })

    return comparison_data

# Função para exportar resultados
def export_results(team_a, team_b, simple_a, weighted_a, simple_b, weighted_b, predicted_stats, confidence_intervals, score_pred, comparison_data, team_a_counts, team_b_counts, predicted_counts):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Médias"
    ws.append(["Time", "Estatística", "Média Simples", "Média Ponderada", "Nº Partidas"])
    for stat in simple_a.keys():
        ws.append([team_a["team"]["name"], stat, round(simple_a[stat], 1), round(weighted_a[stat], 1), team_a_counts[stat]])
        ws.append([team_b["team"]["name"], stat, round(simple_b[stat], 1), round(weighted_b[stat], 1), team_b_counts[stat]])
    ws = wb.create_sheet("Previsões")
    ws.append(["Estatística", f"{team_a['team']['name']} (Prev)", f"{team_a['team']['name']} (IC 85%)", f"{team_b['team']['name']} (Prev)", f"{team_b['team']['name']} (IC 85%)", "Nº Partidas"])
    for stat in predicted_stats.keys():
        ws.append([
            stat,
            round(predicted_stats[stat]["team_a"], 1),
            f"[{round(confidence_intervals[stat]['team_a'][0], 1)}, {round(confidence_intervals[stat]['team_a'][1], 1)}]",
            round(predicted_stats[stat]["team_b"], 1),
            f"[{round(confidence_intervals[stat]['team_b'][0], 1)}, {round(confidence_intervals[stat]['team_b'][1], 1)}]",
            predicted_counts[stat]
        ])
    ws = wb.create_sheet("Placar Provável")
    ws.append(["Placar", score_pred["score"]])
    ws.append(["Prob. Vitória", round(score_pred["probs"]["win"], 1)])
    ws.append(["Prob. Empate", round(score_pred["probs"]["draw"], 1)])
    ws.append(["Prob. Derrota", round(score_pred["probs"]["loss"], 1)])
    ws.append([f"IC Gols {team_a['team']['name']}", f"[{round(score_pred['ci']['team_a'][0], 1)}, {round(score_pred['ci']['team_a'][1], 1)}]"])
    ws.append([f"IC Gols {team_b['team']['name']}", f"[{round(score_pred['ci']['team_b'][0], 1)}, {round(score_pred['ci']['team_b'][1], 1)}]"])
    ws = wb.create_sheet("Odds e Probabilidades")
    ws.append(["Mercado", "Aposta", "Valor", "Odd", "Prob. Implícita (%)", "Prob. Prevista (%)"])
    for data in comparison_data:
        ws.append([
            data["market"],
            data["bet"],
            data["value"],
            round(data["odd"], 1),
            round(data["implied_prob"], 1),
            round(data["predicted_prob"], 1)
        ])
    filename = f"previsao_{team_a['team']['name']}_vs_{team_b['team']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    wb.save(filename)
    return filename

# Função principal
def main():
    st.set_page_config(page_title="Previsão de Partidas de Futebol", layout="wide")
    st.title("Previsão Estatística de Partidas de Futebol")

    st.subheader("Verificação da Chave da API")
    if st.button("Testar Chave"):
        if test_api_key():
            st.success("Chave da API está funcionando!")
        else:
            st.error("Chave da API inválida ou houve um erro.")

    weights = load_weights()
    tabs = st.tabs([
        "Seleção de Times",
        "Jogos Analisados",
        "Médias",
        "Estatísticas Previstas",
        "Placar Provável",
        "Odds x Previsão",
        "Exportar"
    ])

    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            team_a_name = st.text_input("Time A (Mandante)", placeholder="Digite o nome do time")
            season_a = st.selectbox("Temporada Time A", list(range(2020, 2026)), index=5)
        with col2:
            team_b_name = st.text_input("Time B (Visitante)", placeholder="Digite o nome do time")
            season_b = st.selectbox("Temporada Time B", list(range(2020, 2026)), index=5)
        if st.button("Buscar Times"):
            if len(team_a_name) < 3 or len(team_b_name) < 3:
                st.error("O nome do time deve ter pelo menos 3 caracteres.")
            else:
                teams_a = search_team(team_a_name