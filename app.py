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

# Carregar pesos das competições
def load_weights():
    try:
        with open("pesos.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "Champions League": 0.95,
            "Europa League": 0.85,
            "Libertadores": 0.80,
            "Sul-Americana": 0.72,
            "Premier League": 0.90,
            "La Liga": 0.87,
            "Brasileirão A": 0.78,
            "Brasileirão B": 0.65,
            "Copa do Brasil": 0.65,
            "Estaduais Fortes": 0.58,
            "Estaduais Fracos": 0.55,
            "MLS": 0.70,
            "Liga MX": 0.72,
            "Saudi Pro League": 0.74
        }

# Mapeamento de nomes de ligas da API para pesos.json
LEAGUE_MAPPING = {
    "Serie B": "Brasileirão B",
    "Paranaense - 1": "Estaduais Fracos",
    "Campeonato Paranaense": "Estaduais Fracos",
    "Brasileiro Serie A": "Brasileirão A",
    "Copa Libertadores": "Libertadores",
    "Copa Sudamericana": "Sul-Americana",
    "Serie B Brasil": "Brasileirão B",
    "Paranaense 1ª Divisão": "Estaduais Fracos",
    "UEFA Champions League": "Champions League",
    "UEFA Europa League": "Europa League"
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
        "next": 1  # Próximo jogo
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code == 200:
            data = response.json()
            games = data.get("response", [])
            # Verificar se o próximo jogo é contra o Time B
            for game in games:
                if (game["teams"]["home"]["id"] == team_a_id and game["teams"]["away"]["id"] == team_b_id) or \
                   (game["teams"]["home"]["id"] == team_b_id and game["teams"]["away"]["id"] == team_a_id):
                    return game["fixture"]["id"]
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

# Função para calcular médias (feitas e sofridas)
def calculate_averages(games, team_id, weights):
    # Inicializar todas as estatísticas, incluindo as sofridas
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

    for game in games:
        game_stats = get_game_stats(game["fixture"]["id"])
        team_data = {k: 0 for k in stats.keys()}
        is_home = game["teams"]["home"]["id"] == team_id
        team_data["goals_scored"] = game["goals"]["home" if is_home else "away"] or 0
        team_data["goals_conceded"] = game["goals"]["away" if is_home else "home"] or 0

        if game_stats:
            team_stats = next((s for s in game_stats if s["team"]["id"] == team_id), None)
            opponent_stats = next((s for s in game_stats if s["team"]["id"] != team_id), None)
            if not team_stats or not opponent_stats:
                st.warning(f"Sem estatísticas completas para o jogo entre {game['teams']['home']['name']} vs {game['teams']['away']['name']} em {datetime.strptime(game['fixture']['date'], '%Y-%m-%dT%H:%M:%S+00:00').strftime('%d/%m/%Y %H:%M')}. Usando apenas gols de /fixtures.")
            else:
                # Estatísticas do time
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

                # Estatísticas sofridas (do adversário)
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
        else:
            st.warning(f"Sem estatísticas para o jogo entre {game['teams']['home']['name']} vs {game['teams']['away']['name']} em {datetime.strptime(game['fixture']['date'], '%Y-%m-%dT%H:%M:%S+00:00').strftime('%d/%m/%Y %H:%M')}. Usando apenas gols de /fixtures.")

        opponent_id = game["teams"]["away"]["id"] if is_home else game["teams"]["home"]["id"]
        leagues = get_team_leagues(opponent_id, game["league"]["season"])
        weight = 0.50
        league_name = game["league"]["name"]
        mapped_name = LEAGUE_MAPPING.get(league_name, league_name)
        if mapped_name in weights:
            weight = weights[mapped_name]
        
        # Adicionar os dados ao dicionário stats
        for key in stats:
            stats[key].append(team_data[key])
        
        # Calcular valores ponderados
        for key in weighted_values:
            if "conceded" in key or "suffered" in key:
                weighted_values[key].append(team_data[key] / max(weight, 0.1))
                game_weights[key].append(1 / max(weight, 0.1))
            else:
                weighted_values[key].append(team_data[key] * weight)
                game_weights[key].append(weight)

    # Calcular médias simples e ponderadas
    simple_averages = {k: np.mean(v) if v else 0 for k, v in stats.items()}
    weighted_averages = {}
    for key in weighted_values:
        weighted_sum = sum(weighted_values[key])
        total_weight = sum(game_weights[key]) if game_weights[key] else 1
        weighted_averages[key] = weighted_sum / total_weight if total_weight > 0 else 0
    return simple_averages, weighted_averages

# Função para prever estatísticas (usando feitas e sofridas)
def predict_stats(team_a_simple, team_a_weighted, team_b_simple, team_b_weighted):
    if team_a_weighted["goals_scored"] == 0 or team_b_weighted["goals_scored"] == 0:
        st.warning("Não há dados suficientes de gols para previsões estatísticas confiáveis.")
        return {}, {}
    predicted_stats = {}
    confidence_intervals = {}
    for stat in team_a_weighted.keys():
        # Usar tanto estatísticas feitas quanto sofridas para prever
        if "scored" in stat or "committed" in stat:
            a_pred = (team_a_weighted[stat] + team_b_weighted[stat.replace("scored", "conceded").replace("committed", "suffered")]) * 1.2 / 2
            b_pred = (team_b_weighted[stat] + team_a_weighted[stat.replace("scored", "conceded").replace("committed", "suffered")]) / 1.2 / 2
        elif "conceded" in stat or "suffered" in stat:
            a_pred = (team_a_weighted[stat] + team_b_weighted[stat.replace("conceded", "scored").replace("suffered", "committed")]) / 1.2 / 2
            b_pred = (team_b_weighted[stat] + team_a_weighted[stat.replace("conceded", "scored").replace("suffered", "committed")]) * 1.2 / 2
        else:
            a_pred = (team_a_weighted[stat] + team_b_simple[stat]) / 2
            b_pred = (team_b_weighted[stat] + team_a_simple[stat]) / 2
        predicted_stats[stat] = {"team_a": a_pred, "team_b": b_pred}
        a_std = np.std([team_a_weighted[stat], team_b_weighted[stat.replace("scored", "conceded").replace("committed", "suffered") if ("scored" in stat or "committed" in stat) else stat]])
        b_std = np.std([team_b_weighted[stat], team_a_weighted[stat.replace("scored", "conceded").replace("committed", "suffered") if ("scored" in stat or "committed" in stat) else stat]])
        z = norm.ppf(0.925)
        confidence_intervals[stat] = {
            "team_a": (a_pred - z * a_std / np.sqrt(2), a_pred + z * a_std / np.sqrt(2)),
            "team_b": (b_pred - z * b_std / np.sqrt(2), b_pred + z * b_std / np.sqrt(2))
        }
    return predicted_stats, confidence_intervals

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
    # Calcular probabilidade de mais de 2.5 gols
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
    params = {"fixture": fixture_id}
    try:
        response = requests.get(url, headers=HEADERS, params=params)
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

# Função para comparar odds e previsões (exibir probabilidades implícitas e previstas)
def compare_odds(predicted_stats, score_pred, odds):
    if not predicted_stats or score_pred["score"] == "N/A":
        st.warning("Não há previsões válidas para comparar com odds.")
        return []
    if not odds:
        st.warning("Nenhuma odd disponível para o jogo selecionado.")
        return []

    comparison_data = []
    # Calcular probabilidade de ambos marcarem
    btts_prob = poisson.pmf(1, predicted_stats["goals_scored"]["team_a"]) * poisson.pmf(1, predicted_stats["goals_scored"]["team_b"])
    # Calcular probabilidade de mais/menos de 2.5 gols
    over_2_5_prob = score_pred["total_goals_prob"]
    under_2_5_prob = 1 - over_2_5_prob

    # Processar os mercados principais
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
            # Processar apenas os mercados principais
            if bet_name in ["Match Winner", "Both Teams To Score", "Goals Over/Under"]:
                for value in values:
                    odd = float(value.get("odd", 0))
                    if odd <= 0:
                        continue  # Pular odds inválidas
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
                        "implied_prob": implied_prob * 100,  # Converter para porcentagem
                        "predicted_prob": predicted_prob * 100  # Converter para porcentagem
                    })

    return comparison_data

# Função para exportar resultados
def export_results(team_a, team_b, simple_a, weighted_a, simple_b, weighted_b, predicted_stats, confidence_intervals, score_pred, comparison_data):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Médias"
    ws.append(["Time", "Estatística", "Média Simples", "Média Ponderada"])
    for stat in simple_a.keys():
        ws.append([team_a["team"]["name"], stat, round(simple_a[stat], 1), round(weighted_a[stat], 1)])
        ws.append([team_b["team"]["name"], stat, round(simple_b[stat], 1), round(weighted_b[stat], 1)])
    ws = wb.create_sheet("Previsões")
    ws.append(["Estatística", f"{team_a['team']['name']} (Prev)", f"{team_a['team']['name']} (IC 85%)", f"{team_b['team']['name']} (Prev)", f"{team_b['team']['name']} (IC 85%)"])
    for stat in predicted_stats.keys():
        ws.append([
            stat,
            round(predicted_stats[stat]["team_a"], 1),
            f"[{round(confidence_intervals[stat]['team_a'][0], 1)}, {round(confidence_intervals[stat]['team_a'][1], 1)}]",
            round(predicted_stats[stat]["team_b"], 1),
            f"[{round(confidence_intervals[stat]['team_b'][0], 1)}, {round(confidence_intervals[stat]['team_b'][1], 1)}]"
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

    # Testar chave da API
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

    # Aba 1: Seleção de Times
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
                teams_a = search_team(team_a_name)
                teams_b = search_team(team_b_name)
                st.session_state["teams_a"] = teams_a
                st.session_state["teams_b"] = teams_b
                if not teams_a:
                    st.error("Nenhum time encontrado para o Time A.")
                if not teams_b:
                    st.error("Nenhum time encontrado para o Time B.")
        if "teams_a" in st.session_state and "teams_b" in st.session_state and st.session_state["teams_a"] and st.session_state["teams_b"]:
            st.subheader("Selecione os Times")
            team_a_options = [f"{t['team']['name']} ({t['team']['country']})" for t in st.session_state["teams_a"]]
            team_b_options = [f"{t['team']['name']} ({t['team']['country']})" for t in st.session_state["teams_b"]]
            team_a_selected = st.selectbox("Time A", team_a_options, key="team_a_select")
            team_b_selected = st.selectbox("Time B", team_b_options, key="team_b_select")
            if st.button("Confirmar Seleção"):
                st.session_state["team_a"] = next(t for t in st.session_state["teams_a"] if f"{t['team']['name']} ({t['team']['country']})" == team_a_selected)
                st.session_state["team_b"] = next(t for t in st.session_state["teams_b"] if f"{t['team']['name']} ({t['team']['country']})" == team_b_selected)
                st.session_state["season_a"] = season_a
                st.session_state["season_b"] = season_b
                st.success("Times selecionados com sucesso!")

    # Aba 2: Jogos Analisados
    with tabs[1]:
        if "team_a" in st.session_state and "team_b" in st.session_state:
            team_a_id = st.session_state["team_a"]["team"]["id"]
            team_b_id = st.session_state["team_b"]["team"]["id"]
            season_a = st.session_state["season_a"]
            season_b = st.session_state["season_b"]
            games_a = get_team_games(team_a_id, season_a, home=True)
            games_b = get_team_games(team_b_id, season_b, home=False)
            if games_a:
                st.write(f"Jogos do Time A ({st.session_state['team_a']['team']['name']} - Mandante):")
                for game in games_a:
                    game_date = datetime.strptime(game["fixture"]["date"], "%Y-%m-%dT%H:%M:%S+00:00")
                    formatted_date = game_date.strftime("%d/%m/%Y %H:%M")
                    home_team = game["teams"]["home"]["name"]
                    away_team = game["teams"]["away"]["name"]
                    home_goals = game["goals"]["home"] if game["goals"]["home"] is not None else 0
                    away_goals = game["goals"]["away"] if game["goals"]["away"] is not None else 0
                    league_name = game["league"]["name"]
                    mapped_name = LEAGUE_MAPPING.get(league_name, league_name)
                    weight = weights.get(mapped_name, 0.50)
                    title = f"{home_team} {home_goals} x {away_goals} {away_team} - {formatted_date} (fator ponderação {weight:.2f} - {league_name})"
                    with st.expander(title):
                        stats = get_game_stats(game["fixture"]["id"])
                        if stats:
                            data = []
                            for stat in stats:
                                if stat["team"]["id"] == team_a_id:
                                    for s in stat["statistics"]:
                                        value = s["value"]
                                        if s["type"].lower() == "ball possession" and isinstance(value, str):
                                            value = value.replace("%", "").strip()
                                            value = float(value) if value else 0.0
                                        else:
                                            try:
                                                value = float(value) if value is not None else 0.0
                                            except (ValueError, TypeError):
                                                value = 0.0
                                        data.append([s["type"], value])
                            df_stats = pd.DataFrame(data, columns=["Estatística", "Valor"])
                            df_stats["Valor"] = df_stats["Valor"].round(1)
                            st.dataframe(df_stats)
                        else:
                            st.warning("Nenhuma estatística disponível para este jogo, mas gols podem estar em /fixtures.")
            else:
                st.warning(f"Nenhum jogo finalizado encontrado para {st.session_state['team_a']['team']['name']} na temporada {season_a}. Tente outra temporada, como 2024.")
            if games_b:
                st.write(f"Jogos do Time B ({st.session_state['team_b']['team']['name']} - Visitante):")
                for game in games_b:
                    game_date = datetime.strptime(game["fixture"]["date"], "%Y-%m-%dT%H:%M:%S+00:00")
                    formatted_date = game_date.strftime("%d/%m/%Y %H:%M")
                    home_team = game["teams"]["home"]["name"]
                    away_team = game["teams"]["away"]["name"]
                    home_goals = game["goals"]["home"] if game["goals"]["home"] is not None else 0
                    away_goals = game["goals"]["away"] if game["goals"]["away"] is not None else 0
                    league_name = game["league"]["name"]
                    mapped_name = LEAGUE_MAPPING.get(league_name, league_name)
                    weight = weights.get(mapped_name, 0.50)
                    title = f"{home_team} {home_goals} x {away_goals} {away_team} - {formatted_date} (fator ponderação {weight:.2f} - {league_name})"
                    with st.expander(title):
                        stats = get_game_stats(game["fixture"]["id"])
                        if stats:
                            data = []
                            for stat in stats:
                                if stat["team"]["id"] == team_b_id:
                                    for s in stat["statistics"]:
                                        value = s["value"]
                                        if s["type"].lower() == "ball possession" and isinstance(value, str):
                                            value = value.replace("%", "").strip()
                                            value = float(value) if value else 0.0
                                        else:
                                            try:
                                                value = float(value) if value is not None else 0.0
                                            except (ValueError, TypeError):
                                                value = 0.0
                                        data.append([s["type"], value])
                            df_stats = pd.DataFrame(data, columns=["Estatística", "Valor"])
                            df_stats["Valor"] = df_stats["Valor"].round(1)
                            st.dataframe(df_stats)
                        else:
                            st.warning("Nenhuma estatística disponível para este jogo, mas gols podem estar em /fixtures.")
            else:
                st.warning(f"Nenhum jogo finalizado encontrado para {st.session_state['team_b']['team']['name']} na temporada {season_b}. Tente outra temporada, como 2024.")
        else:
            st.info("Selecione os times na aba 'Seleção de Times' para ver os jogos.")

    # Aba 3: Médias
    with tabs[2]:
        if "team_a" in st.session_state and "team_b" in st.session_state:
            team_a_id = st.session_state["team_a"]["team"]["id"]
            team_b_id = st.session_state["team_b"]["team"]["id"]
            season_a = st.session_state["season_a"]
            season_b = st.session_state["season_b"]
            games_a = get_team_games(team_a_id, season_a, home=True)
            games_b = get_team_games(team_b_id, season_b, home=False)
            if games_a and games_b:
                simple_a, weighted_a = calculate_averages(games_a, team_a_id, weights)
                simple_b, weighted_b = calculate_averages(games_b, team_b_id, weights)
                st.session_state["simple_a"] = simple_a
                st.session_state["weighted_a"] = weighted_a
                st.session_state["simple_b"] = simple_b
                st.session_state["weighted_b"] = weighted_b
                df_a = pd.DataFrame({
                    "Estatística": simple_a.keys(),
                    "Média Simples": [round(v, 1) for v in simple_a.values()],
                    "Média Ponderada": [round(v, 1) for v in weighted_a.values()]
                })
                df_b = pd.DataFrame({
                    "Estatística": simple_b.keys(),
                    "Média Simples": [round(v, 1) for v in simple_b.values()],
                    "Média Ponderada": [round(v, 1) for v in weighted_b.values()]
                })
                st.write(f"Médias do Time A ({st.session_state['team_a']['team']['name']}):")
                st.dataframe(df_a)
                st.write(f"Médias do Time B ({st.session_state['team_b']['team']['name']}):")
                st.dataframe(df_b)
            else:
                st.warning("Não foi possível calcular médias. Verifique se há jogos finalizados na aba 'Jogos Analisados'.")
        else:
            st.info("Selecione os times na aba 'Seleção de Times' para calcular as médias.")

    # Aba 4: Estatísticas Previstas
    with tabs[3]:
        if "simple_a" in st.session_state:
            predicted_stats, confidence_intervals = predict_stats(
                st.session_state["simple_a"], st.session_state["weighted_a"],
                st.session_state["simple_b"], st.session_state["weighted_b"]
            )
            st.session_state["predicted_stats"] = predicted_stats
            st.session_state["confidence_intervals"] = confidence_intervals
            if predicted_stats:
                df_pred = pd.DataFrame([
                    {
                        "Estatística": stat,
                        f"{st.session_state['team_a']['team']['name']}": round(pred["team_a"], 1),
                        f"{st.session_state['team_a']['team']['name']} (IC 85%)": f"[{round(confidence_intervals[stat]['team_a'][0], 1)}, {round(confidence_intervals[stat]['team_a'][1], 1)}]",
                        f"{st.session_state['team_b']['team']['name']}": round(pred["team_b"], 1),
                        f"{st.session_state['team_b']['team']['name']} (IC 85%)": f"[{round(confidence_intervals[stat]['team_b'][0], 1)}, {round(confidence_intervals[stat]['team_b'][1], 1)}]"
                    }
                    for stat, pred in predicted_stats.items()
                ])
                st.dataframe(df_pred)
            else:
                st.warning("Não foi possível gerar previsões devido à falta de dados de gols.")
        else:
            st.info("Calcule as médias na aba 'Médias' para gerar previsões.")

    # Aba 5: Placar Provável
    with tabs[4]:
        if "weighted_a" in st.session_state:
            score_pred = predict_score(st.session_state["weighted_a"], st.session_state["weighted_b"])
            st.session_state["score_pred"] = score_pred
            if score_pred["score"] != "N/A":
                st.write(f"Placar mais provável: {score_pred['score']}")
                st.write(f"Probabilidade de Vitória: {score_pred['probs']['win']*100:.1f}%")
                st.write(f"Probabilidade de Empate: {score_pred['probs']['draw']*100:.1f}%")
                st.write(f"Probabilidade de Derrota: {score_pred['probs']['loss']*100:.1f}%")
                st.write(f"Intervalo de Confiança (Gols {st.session_state['team_a']['team']['name']}): [{score_pred['ci']['team_a'][0]:.1f}, {score_pred['ci']['team_a'][1]:.1f}]")
                st.write(f"Intervalo de Confiança (Gols {st.session_state['team_b']['team']['name']}): [{score_pred['ci']['team_b'][0]:.1f}, {score_pred['ci']['team_b'][1]:.1f}]")
            else:
                st.warning("Não foi possível prever o placar devido à falta de dados de gols.")
        else:
            st.info("Calcule as médias na aba 'Médias' para prever o placar.")

    # Aba 6: Odds x Previsão
    with tabs[5]:
        if "team_a" in st.session_state and "team_b" in st.session_state and "season_a" in st.session_state and "predicted_stats" in st.session_state:
            team_a_id = st.session_state["team_a"]["team"]["id"]
            team_b_id = st.session_state["team_b"]["team"]["id"]
            season = st.session_state["season_a"]
            fixture_id = find_next_fixture(team_a_id, team_b_id, season)
            odds = get_odds(fixture_id)
            comparison_data = compare_odds(st.session_state["predicted_stats"], st.session_state["score_pred"], odds)
            st.session_state["comparison_data"] = comparison_data
            if comparison_data:
                df_odds = pd.DataFrame(comparison_data)
                df_odds["implied_prob"] = df_odds["implied_prob"].round(1)
                df_odds["predicted_prob"] = df_odds["predicted_prob"].round(1)
                df_odds["odd"] = df_odds["odd"].round(1)
                st.dataframe(df_odds)
            else:
                st.warning("Nenhuma odd disponível para os mercados principais ou os dados retornados não atendem aos critérios de comparação.")
        else:
            st.info("Selecione os times na aba 'Seleção de Times' e complete as previsões nas abas anteriores para comparar odds.")

    # Aba 7: Exportar
    with tabs[6]:
        if "team_a" in st.session_state and "predicted_stats" in st.session_state and st.session_state["predicted_stats"]:
            if st.button("Exportar Resultados"):
                filename = export_results(
                    st.session_state["team_a"], st.session_state["team_b"],
                    st.session_state["simple_a"], st.session_state["weighted_a"],
                    st.session_state["simple_b"], st.session_state["weighted_b"],
                    st.session_state["predicted_stats"], st.session_state["confidence_intervals"],
                    st.session_state["score_pred"], st.session_state["comparison_data"]
                )
                with open(filename, "rb") as f:
                    st.download_button("Baixar .xlsx", f, file_name=filename)
                st.success("Resultados exportados com sucesso!")
        else:
            st.info("Complete as previsões nas abas anteriores para exportar os resultados.")

if __name__ == "__main__":
    main()