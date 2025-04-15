"""PrevFut - Módulo de Previsão de Resultados de Futebol

Este módulo fornece funcionalidades para:
- Buscar dados de times e partidas via API-Football
- Calcular previsões estatísticas de resultados
- Analisar oportunidades de apostas
- Processar estatísticas de jogos

Requer uma chave de API válida da API-Football configurada como variável de ambiente.
"""

from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, List, Any
from requests.exceptions import RequestException
from scipy.stats import poisson
from loguru import logger
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import requests
import json
import os

# Carregar variáveis de ambiente
load_dotenv()

# Configuração de logging
logger.add(
    "prevfut.log",
    rotation="500 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Configuração da API
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logger.error("API_KEY não encontrada nas variáveis de ambiente")
    raise ValueError("A variável de ambiente API_KEY não está definida. Configure-a no arquivo .env ou no Streamlit Cloud Secrets.")

from config import (
    BASE_URL,
    REQUEST_LIMIT_PER_MINUTE,
    REQUEST_INTERVAL,
    CACHE_FILE,
    ESTATISTICAS_MODELO,
    COMPETICAO_PESOS,
    LOG_CONFIG,
    THRESHOLD_VALOR,
    MIN_PROBABILIDADE,
    MAX_ODD
)

# Configuração dos headers da API
HEADERS = {"x-apisports-key": API_KEY}

# Configuração do logger
logger.configure(**LOG_CONFIG)

def load_cache() -> Dict[str, Any]:
    """Carrega o cache do arquivo JSON.
    
    Returns:
        Dict[str, Any]: Dicionário com dados em cache ou vazio se não existir.
    """
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                logger.debug(f"Cache carregado de {CACHE_FILE}")
                return json.load(f)
    except Exception as e:
        logger.warning(f"Erro ao carregar cache: {e}")
    return {}

def save_cache(cache: Dict[str, Any]) -> None:
    """Salva o cache em arquivo JSON.
    
    Args:
        cache (Dict[str, Any]): Dicionário com dados para cache.
    """
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
            logger.debug(f"Cache salvo em {CACHE_FILE}")
    except Exception as e:
        logger.error(f"Erro ao salvar cache: {e}")

def make_api_request(url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Faz requisição à API com cache e tratamento de erros.
    
    Args:
        url (str): URL da API.
        params (Dict[str, Any]): Parâmetros da requisição.
    
    Returns:
        Optional[Dict[str, Any]]: Dados da resposta ou None se houver erro.
    """
    cache = load_cache()
    cache_key = f"{url}_{json.dumps(params, sort_keys=True)}"
    
    # Verificar cache
    if cache_key in cache:
        logger.debug(f"Dados encontrados em cache para: {url}")
        return cache[cache_key]
    
    try:
        # Respeitar limite de requisições
        time.sleep(REQUEST_INTERVAL)
        
        # Fazer requisição
        logger.info(f"Fazendo requisição para: {url}")
        response = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        # Verificar erros da API
        if data.get("errors"):
            logger.error(f"Erro da API: {data['errors']}")
            return None
        
        # Salvar no cache
        cache[cache_key] = data
        save_cache(cache)
        logger.debug(f"Dados salvos em cache para: {url}")
        
        return data
        
    except RequestException as e:
        logger.error(f"Erro na requisição HTTP: {e}")
    except Exception as e:
        logger.error(f"Erro inesperado: {e}")
    
    return None

def buscar_id_time(nome_busca: str) -> Tuple[Optional[int], Optional[str]]:
    url = f"{BASE_URL}/teams"
    data = make_api_request(url, {"search": nome_busca})
    
    if not data:
        print("❌ Nenhum time correspondente encontrado.")
        return None, None
        
    times = data.get("response", [])
    candidatos: List[Tuple[int, str]] = []

    for t in times:
        nome_api = t["team"]["name"].lower()
        if nome_busca.lower() in nome_api:
            candidatos.append((t["team"]["id"], t["team"]["name"]))

    if not candidatos:
        print("❌ Nenhum time correspondente encontrado.")
        return None, None

    print("\n🔍 Times encontrados:")
    for idx, (tid, nome) in enumerate(candidatos, 1):
        print(f"{idx}. {nome}")
    print(f"Selecionando automaticamente: {candidatos[0][1]}")
    return candidatos[0]

    while True:
        try:
            escolha = int(input("Digite o número do time desejado: "))
            if 1 <= escolha <= len(candidatos):
                return candidatos[escolha - 1]
            else:
                print("❗ Número fora do intervalo. Tente novamente.")
        except ValueError:
            print("❗ Entrada inválida. Digite apenas o número da opção.")

def detectar_temporada_atual(time_id: int) -> Optional[int]:
    url = f"{BASE_URL}/teams/seasons"
    data = make_api_request(url, {"team": time_id})
    
    if not data:
        return None
        
    temporadas = sorted(data.get("response", []), reverse=True)
    hoje_iso = datetime.now().isoformat()

    for temporada in temporadas:
        url_fixture = f"{BASE_URL}/fixtures"
        res = make_api_request(url_fixture, {
            "team": time_id,
            "season": temporada
        })
        
        if not res:
            continue
            
        jogos = res.get("response", [])
        jogos_passados = [j for j in jogos if j["fixture"]["date"] <= hoje_iso]
        if jogos_passados:
            return temporada

    return temporadas[0] if temporadas else None

def determinar_peso_adversario(competicao: str) -> float:
    return COMPETICAO_PESOS.get(competicao, 0.5)

def buscar_jogos_com_season(time_id: int, nome_oficial: str, season: int, limite: int = 15) -> Optional[pd.DataFrame]:
    url = f"{BASE_URL}/fixtures"
    data = make_api_request(url, {
        "team": time_id,
        "season": season
    })
    
    if not data:
        return None

    todos_os_jogos = data.get("response", [])
    hoje_iso = datetime.now().isoformat()
    jogos_passados = [j for j in todos_os_jogos if j["fixture"]["date"] <= hoje_iso]
    jogos = sorted(jogos_passados, key=lambda x: x["fixture"]["date"], reverse=True)[:limite]

    print(f"\n📅 Temporada selecionada para {nome_oficial}: {season}")
    print(f"🔍 {len(jogos)} jogos já realizados retornados para {nome_oficial}:\n")

    dados = []

    for jogo in jogos:
        fixture_id = jogo["fixture"]["id"]
        data = jogo["fixture"]["date"][:10]
        mandante = jogo["teams"]["home"]["name"]
        visitante = jogo["teams"]["away"]["name"]
        gols_mandante = jogo["goals"]["home"] if jogo["goals"]["home"] is not None else 0
        gols_visitante = jogo["goals"]["away"] if jogo["goals"]["away"] is not None else 0
        is_mandante = mandante == nome_oficial
        competicao = jogo.get("league", {}).get("name", "Desconhecida")
        peso_adversario = determinar_peso_adversario(competicao)

        print(f"🗓️ {data} — {mandante} {gols_mandante} x {gols_visitante} {visitante} (Peso adversário: {peso_adversario}, Competição: {competicao})")

        estat_url = f"{BASE_URL}/fixtures/statistics"
        estat_data = make_api_request(estat_url, {"fixture": fixture_id})
        
        jogo_dados = {
            "data": data,
            "mandante": mandante,
            "visitante": visitante,
            "gols_mandante": gols_mandante,
            "gols_visitante": gols_visitante,
            "peso_adversario": peso_adversario,
            "is_mandante": is_mandante,
            "competicao": competicao
        }

        if estat_data:
            estatisticas = estat_data.get("response", [])
            for time in estatisticas:
                team_name = time["team"]["name"]
                is_team_mandante = team_name == mandante
                prefixo = "mandante" if is_team_mandante else "visitante"
                for estat in time["statistics"]:
                    tipo = estat["type"].lower().replace(' ', '_')
                    if tipo == "corner_kicks":
                        tipo = "corners"
                    if tipo in ESTATISTICAS_MODELO:
                        chave = f"{tipo}_{prefixo}"
                        valor = estat["value"]
                        if valor is None:
                            valor = 0
                        elif tipo == "ball_possession" and isinstance(valor, str):
                            valor = float(valor.strip('%')) if valor.strip('%') else 0.0
                        elif tipo in ["shots_on_goal", "total_shots", "corners", "fouls", "yellow_cards", "red_cards", "offsides"]:
                            try:
                                valor = int(valor)
                                if tipo == "shots_on_goal" and valor > 15:
                                    valor = 0
                                elif tipo == "total_shots" and valor > 25:
                                    valor = 0
                                elif tipo == "corners" and valor > 15:
                                    valor = 0
                                elif tipo == "offsides" and valor > 10:  # Limite razoável para impedimentos
                                    valor = 0
                            except (ValueError, TypeError):
                                valor = 0
                        if prefixo == ("mandante" if is_mandante else "visitante"):
                            valor = valor * peso_adversario if tipo != "ball_possession" else valor
                        else:
                            valor = valor / peso_adversario if peso_adversario > 0 and tipo != "ball_possession" else valor
                        jogo_dados[chave] = valor

        for stat in ESTATISTICAS_MODELO:
            for p in ["mandante", "visitante"]:
                chave = f"{stat}_{p}"
                if chave not in jogo_dados:
                    jogo_dados[chave] = 0

        dados.append(jogo_dados)

    df = pd.DataFrame(dados)
    return df if not df.empty else None

def calcular_media_ajustada(df: pd.DataFrame, time_nome: str, is_mandante: bool = None) -> Dict[str, str]:
    estatisticas_finais = {}
    
    if is_mandante is not None:
        df = df[df["is_mandante"] == is_mandante]
    
    if df.empty:
        for stat_key in ESTATISTICAS_MODELO:
            estatisticas_finais[f"{ESTATISTICAS_MODELO[stat_key]} Feita"] = "N/D"
            estatisticas_finais[f"{ESTATISTICAS_MODELO[stat_key]} Sofrida"] = "N/D"
        return estatisticas_finais

    for stat_key in ESTATISTICAS_MODELO:
        is_time_mandante = df.iloc[0]["mandante"] == time_nome
        prefixo = "mandante" if is_time_mandante else "visitante"
        prefixo_oposto = "visitante" if is_time_mandante else "mandante"
        col_feita = f"{stat_key}_{prefixo}"
        col_sofrida = f"{stat_key}_{prefixo_oposto}"

        for col, sufixo in [(col_feita, "Feita"), (col_sofrida, "Sofrida")]:
            if col in df.columns:
                try:
                    valid_data = df[[col]].dropna(subset=[col])
                    if not valid_data.empty:
                        valores = pd.to_numeric(valid_data[col], errors='coerce')
                        if valores.isna().all():
                            estatisticas_finais[f"{ESTATISTICAS_MODELO[stat_key]} {sufixo}"] = "N/D"
                            continue
                        media = np.mean(valores)
                        desvio = np.std(valores) if len(valores) > 1 else 0.0
                        media = np.round(media, 2)
                        desvio = np.round(desvio, 2)
                        z_score = 1.282
                        erro = z_score * desvio / np.sqrt(len(valores)) if len(valores) > 0 else 0.0
                        ic_inf = np.round(max(0, media - erro), 2)
                        ic_sup = np.round(media + erro, 2)
                        estatisticas_finais[f"{ESTATISTICAS_MODELO[stat_key]} {sufixo}"] = f"{media} ± {desvio} (IC 80%: [{ic_inf}, {ic_sup}])"
                    else:
                        estatisticas_finais[f"{ESTATISTICAS_MODELO[stat_key]} {sufixo}"] = "N/D"
                except Exception as e:
                    print(f"⚠️ Erro ao calcular {col}: {e}")
                    estatisticas_finais[f"{ESTATISTICAS_MODELO[stat_key]} {sufixo}"] = "N/D"
            else:
                estatisticas_finais[f"{ESTATISTICAS_MODELO[stat_key]} {sufixo}"] = "N/D"

    return estatisticas_finais

def prever_placar(time_a: str, df_a: pd.DataFrame, time_b: str, df_b: pd.DataFrame, time_a_mandante: bool) -> Dict[str, Any]:
    # Calcular médias ajustadas para gols
    lambda_a_casa = float(calcular_media_ajustada(df_a, time_a, is_mandante=True).get("Gols Feita", "0 ± 0").split(" ± ")[0])
    lambda_a_fora = float(calcular_media_ajustada(df_a, time_a, is_mandante=False).get("Gols Feita", "0 ± 0").split(" ± ")[0])
    lambda_a = (lambda_a_casa + lambda_a_fora) / 2 if not time_a_mandante else lambda_a_casa

    lambda_b_casa = float(calcular_media_ajustada(df_b, time_b, is_mandante=True).get("Gols Feita", "0 ± 0").split(" ± ")[0])
    lambda_b_fora = float(calcular_media_ajustada(df_b, time_b, is_mandante=False).get("Gols Feita", "0 ± 0").split(" ± ")[0])
    lambda_b = (lambda_b_casa + lambda_b_fora) / 2 if time_a_mandante else lambda_b_fora

    # Prever placar mais provável
    max_gols = 10
    prob_a = [poisson.pmf(i, lambda_a) for i in range(max_gols + 1)]
    prob_b = [poisson.pmf(i, lambda_b) for i in range(max_gols + 1)]

    placar_mais_provavel = (0, 0)
    maior_prob = 0.0
    for i in range(max_gols + 1):
        for j in range(max_gols + 1):
            prob = prob_a[i] * prob_b[j]
            if prob > maior_prob:
                maior_prob = prob
                placar_mais_provavel = (i, j)

    # Calcular médias ajustadas para outras estatísticas (totais, sem distinguir mandante/visitante)
    medias_estatisticas = {}
    for stat_key in ESTATISTICAS_MODELO:
        if stat_key == "gols":
            continue  # Já calculado acima
        media_a = float(calcular_media_ajustada(df_a, time_a).get(f"{ESTATISTICAS_MODELO[stat_key]} Feita", "0 ± 0").split(" ± ")[0])
        media_b = float(calcular_media_ajustada(df_b, time_b).get(f"{ESTATISTICAS_MODELO[stat_key]} Feita", "0 ± 0").split(" ± ")[0])
        media_total = media_a + media_b  # Total combinado para o jogo
        medias_estatisticas[stat_key] = media_total

    return {
        "lambda_a": lambda_a,
        "lambda_b": lambda_b,
        "placar_mais_provavel": placar_mais_provavel,
        "medias_estatisticas": medias_estatisticas
    }

def calcular_probabilidades_1x2(lambda_a: float, lambda_b: float) -> Tuple[float, float, float]:
    max_gols = 10
    prob_a = [poisson.pmf(i, lambda_a) for i in range(max_gols + 1)]
    prob_b = [poisson.pmf(i, lambda_b) for i in range(max_gols + 1)]

    prob_vitoria_a = 0.0
    prob_empate = 0.0
    prob_vitoria_b = 0.0

    for i in range(max_gols + 1):
        for j in range(max_gols + 1):
            prob = prob_a[i] * prob_b[j]
            if i > j:
                prob_vitoria_a += prob
            elif i == j:
                prob_empate += prob
            else:
                prob_vitoria_b += prob

    return prob_vitoria_a, prob_empate, prob_vitoria_b

# Função movada para fora da função acima
def calcular_probabilidade_over_under(lambda_a: float, lambda_b: float, threshold: float = 2.5) -> Tuple[float, float]:
    max_gols = 10
    prob_a = [poisson.pmf(i, lambda_a) for i in range(max_gols + 1)]
    prob_b = [poisson.pmf(i, lambda_b) for i in range(max_gols + 1)]

    prob_over = 0.0
    for i in range(max_gols + 1):
        for j in range(max_gols + 1):
            total_gols = i + j
            prob = prob_a[i] * prob_b[j]
            if total_gols > threshold:
                prob_over += prob
    prob_under = 1.0 - prob_over
    return prob_over, prob_under

def calcular_probabilidade_btts(lambda_a: float, lambda_b: float) -> Tuple[float, float]:
    max_gols = 10
    prob_a = [poisson.pmf(i, lambda_a) for i in range(max_gols + 1)]
    prob_b = [poisson.pmf(i, lambda_b) for i in range(max_gols + 1)]

    prob_btts_yes = 0.0
    for i in range(1, max_gols + 1):  # Pelo menos 1 gol do Time A
        for j in range(1, max_gols + 1):  # Pelo menos 1 gol do Time B
            prob = prob_a[i] * prob_b[j]
            prob_btts_yes += prob
    prob_btts_no = 1.0 - prob_btts_yes
    return prob_btts_yes, prob_btts_no

def calcular_probabilidade_over_under_estatistica(media: float, threshold: float) -> Tuple[float, float]:
    prob_over = 1.0 - poisson.cdf(threshold - 0.5, media)  # Ajuste para continuidade
    prob_under = 1.0 - prob_over
    return prob_over, prob_under

def buscar_odds(time_a: str, time_b: str, time_id_a: int, time_id_b: int, temporada_a: int) -> Optional[Dict[str, Any]]:
    # Passo 1: Buscar jogos futuros entre os dois times
    hoje = datetime.now()
    data_inicio = hoje.strftime("%Y-%m-%d")
    data_fim = (hoje + timedelta(days=30)).strftime("%Y-%m-%d")  # Buscar jogos nos próximos 30 dias

    url_fixtures = f"{BASE_URL}/fixtures"
    params = {
        "team": time_id_a,
        "season": temporada_a,
        "from": data_inicio,
        "to": data_fim,
        "status": "NS"  # NS = Not Started (jogos não iniciados)
    }
    fixtures_data = make_api_request(url_fixtures, params)
    
    if not fixtures_data:
        print("❌ Não foi possível encontrar jogos futuros para o time A.")
        return None

    fixture_id = None
    jogos = fixtures_data.get("response", [])
    for jogo in jogos:
        home_team_id = jogo["teams"]["home"]["id"]
        away_team_id = jogo["teams"]["away"]["id"]
        if (home_team_id == time_id_a and away_team_id == time_id_b) or \
           (home_team_id == time_id_b and away_team_id == time_id_a):
            fixture_id = jogo["fixture"]["id"]
            break

    if not fixture_id:
        print(f"❌ Jogo {time_a} vs {time_b} não encontrado nos próximos 30 dias.")
        return None

    # Passo 2: Buscar odds para múltiplos mercados
    url_odds = f"{BASE_URL}/odds"
    params = {
        "fixture": fixture_id
    }
    odds_data = make_api_request(url_odds, params)

    if not odds_data or not odds_data.get("response"):
        print(f"❌ Odds não disponíveis para o jogo {time_a} vs {time_b}.")
        return None

    response = odds_data["response"][0]
    bookmakers = response.get("bookmakers", [])
    if not bookmakers:
        print("❌ Nenhum bookmaker disponível para este jogo.")
        return None

    bookmaker = bookmakers[0]  # Usar o primeiro bookmaker
    bets = bookmaker.get("bets", [])

    odds_dict = {}

    # Mercado 1x2 (Match Winner)
    bet_1x2 = next((bet for bet in bets if bet["id"] == 1), None)
    if bet_1x2:
        odds_dict["h2h"] = {
            "outcomes": [
                {"name": time_a if jogo["teams"]["home"]["id"] == time_id_a else time_b, "price": float(bet_1x2["values"][0]["odd"])},  # Home
                {"name": "Draw", "price": float(bet_1x2["values"][1]["odd"])},  # Draw
                {"name": time_b if jogo["teams"]["away"]["id"] == time_id_b else time_a, "price": float(bet_1x2["values"][2]["odd"])}   # Away
            ]
        }

    # Mercado Over/Under 2.5 Gols
    bet_over_under = next((bet for bet in bets if bet["id"] == 3 and bet["values"][0]["value"] == "Over 2.5"), None)
    if bet_over_under:
        odds_dict["over_under_2.5"] = {
            "outcomes": [
                {"name": "Over 2.5", "price": float(bet_over_under["values"][0]["odd"])},
                {"name": "Under 2.5", "price": float(bet_over_under["values"][1]["odd"])}
            ]
        }

    # Mercado Ambas as Equipes Marcam (BTTS)
    bet_btts = next((bet for bet in bets if bet["id"] == 13), None)
    if bet_btts:
        odds_dict["btts"] = {
            "outcomes": [
                {"name": "Yes", "price": float(bet_btts["values"][0]["odd"])},
                {"name": "No", "price": float(bet_btts["values"][1]["odd"])}
            ]
        }

    # Mercado Over/Under Escanteios (ex.: 9.5, pode variar)
    bet_corners = next((bet for bet in bets if bet["id"] == 34 and "Over" in bet["values"][0]["value"]), None)
    if bet_corners:
        threshold = float(bet_corners["values"][0]["value"].split()[-1])  # Ex.: "Over 9.5" -> 9.5
        odds_dict["corners"] = {
            "threshold": threshold,
            "outcomes": [
                {"name": f"Over {threshold}", "price": float(bet_corners["values"][0]["odd"])},
                {"name": f"Under {threshold}", "price": float(bet_corners["values"][1]["odd"])}
            ]
        }

    # Mercado Over/Under Cartões (ex.: 4.5, pode variar)
    bet_cards = next((bet for bet in bets if bet["id"] == 39 and "Over" in bet["values"][0]["value"]), None)
    if bet_cards:
        threshold = float(bet_cards["values"][0]["value"].split()[-1])  # Ex.: "Over 4.5" -> 4.5
        odds_dict["cards"] = {
            "threshold": threshold,
            "outcomes": [
                {"name": f"Over {threshold}", "price": float(bet_cards["values"][0]["odd"])},
                {"name": f"Under {threshold}", "price": float(bet_cards["values"][1]["odd"])}
            ]
        }

    # Mercado Over/Under Finalizações Totais (Total Shots)
    bet_total_shots = next((bet for bet in bets if bet["id"] == 42 and "Over" in bet["values"][0]["value"]), None)
    if bet_total_shots:
        threshold = float(bet_total_shots["values"][0]["value"].split()[-1])  # Ex.: "Over 20" -> 20
        odds_dict["total_shots"] = {
            "threshold": threshold,
            "outcomes": [
                {"name": f"Over {threshold}", "price": float(bet_total_shots["values"][0]["odd"])},
                {"name": f"Under {threshold}", "price": float(bet_total_shots["values"][1]["odd"])}
            ]
        }

    # Mercado Over/Under Chutes ao Gol (Shots on Goal)
    bet_shots_on_goal = next((bet for bet in bets if bet["id"] == 43 and "Over" in bet["values"][0]["value"]), None)
    if bet_shots_on_goal:
        threshold = float(bet_shots_on_goal["values"][0]["value"].split()[-1])  # Ex.: "Over 8" -> 8
        odds_dict["shots_on_goal"] = {
            "threshold": threshold,
            "outcomes": [
                {"name": f"Over {threshold}", "price": float(bet_shots_on_goal["values"][0]["odd"])},
                {"name": f"Under {threshold}", "price": float(bet_shots_on_goal["values"][1]["odd"])}
            ]
        }

    # Mercado Over/Under Faltas (Fouls)
    bet_fouls = next((bet for bet in bets if bet["id"] == 45 and "Over" in bet["values"][0]["value"]), None)
    if bet_fouls:
        threshold = float(bet_fouls["values"][0]["value"].split()[-1])  # Ex.: "Over 25" -> 25
        odds_dict["fouls"] = {
            "threshold": threshold,
            "outcomes": [
                {"name": f"Over {threshold}", "price": float(bet_fouls["values"][0]["odd"])},
                {"name": f"Under {threshold}", "price": float(bet_fouls["values"][1]["odd"])}
            ]
        }

    if not odds_dict:
        print("❌ Nenhum mercado de aposta disponível.")
        return None

    return {"bookmakers": [{"markets": [{"key": key, **value} for key, value in odds_dict.items()]}]}

    # Definir bet como o primeiro elemento de bets
    bet = bets[0]

    odds = {
        "bookmakers": [
            {
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": time_a if jogo["teams"]["home"]["id"] == time_id_a else time_b, "price": float(bet["values"][0]["odd"])},  # Home
                            {"name": "Draw", "price": float(bet["values"][1]["odd"])},  # Draw
                            {"name": time_b if jogo["teams"]["away"]["id"] == time_id_b else time_a, "price": float(bet["values"][2]["odd"])}   # Away
                        ]
                    }
                ]
            }
        ]
    }
    return odds

def calcular_probabilidade_implicita(odd: float) -> float:
    if odd <= 1.0:
        return 0.0
    return (1 / odd) * 100

def identificar_oportunidades(prob_vitoria_a: float, prob_empate: float, prob_vitoria_b: float, 
                             prob_over: float, prob_under: float, prob_btts_yes: float, prob_btts_no: float,
                             prob_corners_over: float, prob_corners_under: float, prob_cards_over: float, prob_cards_under: float,
                             prob_total_shots_over: float, prob_total_shots_under: float,
                             prob_shots_on_goal_over: float, prob_shots_on_goal_under: float,
                             prob_fouls_over: float, prob_fouls_under: float,
                             odds: Dict[str, Any], time_a: str, time_b: str) -> List[str]:
    oportunidades = []
    bookmakers = odds.get("bookmakers", [])
    if not bookmakers:
        return ["❌ Nenhuma odd disponível para comparação."]

    markets = bookmakers[0].get("markets", [])
    if not markets:
        return ["❌ Nenhum mercado disponível."]

    # Mercado 1x2 (h2h)
    h2h_market = next((market for market in markets if market["key"] == "h2h"), None)
    if h2h_market:
        outcomes = h2h_market["outcomes"]
        odd_vitoria_a = odd_empate = odd_vitoria_b = None
        for outcome in outcomes:
            if outcome["name"].lower() == time_a.lower():
                odd_vitoria_a = outcome["price"]
            elif outcome["name"].lower() == time_b.lower():
                odd_vitoria_b = outcome["price"]
            elif outcome["name"].lower() == "draw":
                odd_empate = outcome["price"]

        if all([odd_vitoria_a, odd_empate, odd_vitoria_b]):
            prob_implicita_a = calcular_probabilidade_implicita(odd_vitoria_a)
            prob_implicita_empate = calcular_probabilidade_implicita(odd_empate)
            prob_implicita_b = calcular_probabilidade_implicita(odd_vitoria_b)

            prob_vitoria_a *= 100  # Converter para porcentagem
            prob_empate *= 100
            prob_vitoria_b *= 100

            print(f"\n📊 Comparação de Probabilidades para {time_a} vs {time_b} (1x2):")
            print(f"Vitória {time_a}: Prevista {prob_vitoria_a:.2f}% | Implícita {prob_implicita_a:.2f}%")
            print(f"Empate: Prevista {prob_empate:.2f}% | Implícita {prob_implicita_empate:.2f}%")
            print(f"Vitória {time_b}: Prevista {prob_vitoria_b:.2f}% | Implícita {prob_implicita_b:.2f}%")

            if prob_vitoria_a > prob_implicita_a:
                oportunidades.append(f"✅ Oportunidade (1x2): Vitória de {time_a} (Odd: {odd_vitoria_a:.2f}) - Probabilidade Prevista: {prob_vitoria_a:.2f}% > Implícita: {prob_implicita_a:.2f}%")
            if prob_empate > prob_implicita_empate:
                oportunidades.append(f"✅ Oportunidade (1x2): Empate (Odd: {odd_empate:.2f}) - Probabilidade Prevista: {prob_empate:.2f}% > Implícita: {prob_implicita_empate:.2f}%")
            if prob_vitoria_b > prob_implicita_b:
                oportunidades.append(f"✅ Oportunidade (1x2): Vitória de {time_b} (Odd: {odd_vitoria_b:.2f}) - Probabilidade Prevista: {prob_vitoria_b:.2f}% > Implícita: {prob_implicita_b:.2f}%")

    # Mercado Over/Under 2.5 Gols
    ou_market = next((market for market in markets if market["key"] == "over_under_2.5"), None)
    if ou_market:
        outcomes = ou_market["outcomes"]
        odd_over = next(outcome["price"] for outcome in outcomes if outcome["name"] == "Over 2.5")
        odd_under = next(outcome["price"] for outcome in outcomes if outcome["name"] == "Under 2.5")

        prob_implicita_over = calcular_probabilidade_implicita(odd_over)
        prob_implicita_under = calcular_probabilidade_implicita(odd_under)

        prob_over *= 100
        prob_under *= 100

        print(f"\n📊 Comparação de Probabilidades (Over/Under 2.5 Gols):")
        print(f"Over 2.5: Prevista {prob_over:.2f}% | Implícita {prob_implicita_over:.2f}%")
        print(f"Under 2.5: Prevista {prob_under:.2f}% | Implícita {prob_implicita_under:.2f}%")

        if prob_over > prob_implicita_over:
            oportunidades.append(f"✅ Oportunidade (Over/Under 2.5): Over 2.5 Gols (Odd: {odd_over:.2f}) - Probabilidade Prevista: {prob_over:.2f}% > Implícita: {prob_implicita_over:.2f}%")
        if prob_under > prob_implicita_under:
            oportunidades.append(f"✅ Oportunidade (Over/Under 2.5): Under 2.5 Gols (Odd: {odd_under:.2f}) - Probabilidade Prevista: {prob_under:.2f}% > Implícita: {prob_implicita_under:.2f}%")

    # Mercado Ambas as Equipes Marcam (BTTS)
    btts_market = next((market for market in markets if market["key"] == "btts"), None)
    if btts_market:
        outcomes = btts_market["outcomes"]
        odd_yes = next(outcome["price"] for outcome in outcomes if outcome["name"] == "Yes")
        odd_no = next(outcome["price"] for outcome in outcomes if outcome["name"] == "No")

        prob_implicita_yes = calcular_probabilidade_implicita(odd_yes)
        prob_implicita_no = calcular_probabilidade_implicita(odd_no)

        prob_btts_yes *= 100
        prob_btts_no *= 100

        print(f"\n📊 Comparação de Probabilidades (Ambas as Equipes Marcam):")
        print(f"Sim: Prevista {prob_btts_yes:.2f}% | Implícita {prob_implicita_yes:.2f}%")
        print(f"Não: Prevista {prob_btts_no:.2f}% | Implícita {prob_implicita_no:.2f}%")

        if prob_btts_yes > prob_implicita_yes:
            oportunidades.append(f"✅ Oportunidade (BTTS): Sim (Odd: {odd_yes:.2f}) - Probabilidade Prevista: {prob_btts_yes:.2f}% > Implícita: {prob_implicita_yes:.2f}%")
        if prob_btts_no > prob_implicita_no:
            oportunidades.append(f"✅ Oportunidade (BTTS): Não (Odd: {odd_no:.2f}) - Probabilidade Prevista: {prob_btts_no:.2f}% > Implícita: {prob_implicita_no:.2f}%")

    # Mercado Over/Under Escanteios
    corners_market = next((market for market in markets if market["key"] == "corners"), None)
    if corners_market:
        threshold = corners_market["threshold"]
        outcomes = corners_market["outcomes"]
        odd_over = next(outcome["price"] for outcome in outcomes if outcome["name"] == f"Over {threshold}")
        odd_under = next(outcome["price"] for outcome in outcomes if outcome["name"] == f"Under {threshold}")

        prob_implicita_over = calcular_probabilidade_implicita(odd_over)
        prob_implicita_under = calcular_probabilidade_implicita(odd_under)

        prob_corners_over *= 100
        prob_corners_under *= 100

        print(f"\n📊 Comparação de Probabilidades (Over/Under {threshold} Escanteios):")
        print(f"Over {threshold}: Prevista {prob_corners_over:.2f}% | Implícita {prob_implicita_over:.2f}%")
        print(f"Under {threshold}: Prevista {prob_corners_under:.2f}% | Implícita {prob_implicita_under:.2f}%")

        if prob_corners_over > prob_implicita_over:
            oportunidades.append(f"✅ Oportunidade (Escanteios): Over {threshold} (Odd: {odd_over:.2f}) - Probabilidade Prevista: {prob_corners_over:.2f}% > Implícita: {prob_implicita_over:.2f}%")
        if prob_corners_under > prob_implicita_under:
            oportunidades.append(f"✅ Oportunidade (Escanteios): Under {threshold} (Odd: {odd_under:.2f}) - Probabilidade Prevista: {prob_corners_under:.2f}% > Implícita: {prob_implicita_under:.2f}%")

    # Mercado Over/Under Cartões
    cards_market = next((market for market in markets if market["key"] == "cards"), None)
    if cards_market:
        threshold = cards_market["threshold"]
        outcomes = cards_market["outcomes"]
        odd_over = next(outcome["price"] for outcome in outcomes if outcome["name"] == f"Over {threshold}")
        odd_under = next(outcome["price"] for outcome in outcomes if outcome["name"] == f"Under {threshold}")

        prob_implicita_over = calcular_probabilidade_implicita(odd_over)
        prob_implicita_under = calcular_probabilidade_implicita(odd_under)

        prob_cards_over *= 100
        prob_cards_under *= 100

        print(f"\n📊 Comparação de Probabilidades (Over/Under {threshold} Cartões):")
        print(f"Over {threshold}: Prevista {prob_cards_over:.2f}% | Implícita {prob_implicita_over:.2f}%")
        print(f"Under {threshold}: Prevista {prob_cards_under:.2f}% | Implícita {prob_implicita_under:.2f}%")

        if prob_cards_over > prob_implicita_over:
            oportunidades.append(f"✅ Oportunidade (Cartões): Over {threshold} (Odd: {odd_over:.2f}) - Probabilidade Prevista: {prob_cards_over:.2f}% > Implícita: {prob_implicita_over:.2f}%")
        if prob_cards_under > prob_implicita_under:
            oportunidades.append(f"✅ Oportunidade (Cartões): Under {threshold} (Odd: {odd_under:.2f}) - Probabilidade Prevista: {prob_cards_under:.2f}% > Implícita: {prob_implicita_under:.2f}%")

        # Mercado Over/Under Finalizações Totais
    total_shots_market = next((market for market in markets if market["key"] == "total_shots"), None)
    if total_shots_market:
        threshold = total_shots_market["threshold"]
        outcomes = total_shots_market["outcomes"]
        odd_over = next(outcome["price"] for outcome in outcomes if outcome["name"] == f"Over {threshold}")
        odd_under = next(outcome["price"] for outcome in outcomes if outcome["name"] == f"Under {threshold}")

        prob_implicita_over = calcular_probabilidade_implicita(odd_over)
        prob_implicita_under = calcular_probabilidade_implicita(odd_under)

        prob_total_shots_over *= 100
        prob_total_shots_under *= 100

        print(f"\n📊 Comparação de Probabilidades (Over/Under {threshold} Finalizações Totais):")
        print(f"Over {threshold}: Prevista {prob_total_shots_over:.2f}% | Implícita {prob_implicita_over:.2f}%")
        print(f"Under {threshold}: Prevista {prob_total_shots_under:.2f}% | Implícita {prob_implicita_under:.2f}%")

        if prob_total_shots_over > prob_implicita_over:
            oportunidades.append(f"✅ Oportunidade (Finalizações Totais): Over {threshold} (Odd: {odd_over:.2f}) - Probabilidade Prevista: {prob_total_shots_over:.2f}% > Implícita: {prob_implicita_over:.2f}%")
        if prob_total_shots_under > prob_implicita_under:
            oportunidades.append(f"✅ Oportunidade (Finalizações Totais): Under {threshold} (Odd: {odd_under:.2f}) - Probabilidade Prevista: {prob_total_shots_under:.2f}% > Implícita: {prob_implicita_under:.2f}%")

    # Mercado Over/Under Chutes ao Gol
    shots_on_goal_market = next((market for market in markets if market["key"] == "shots_on_goal"), None)
    if shots_on_goal_market:
        threshold = shots_on_goal_market["threshold"]
        outcomes = shots_on_goal_market["outcomes"]
        odd_over = next(outcome["price"] for outcome in outcomes if outcome["name"] == f"Over {threshold}")
        odd_under = next(outcome["price"] for outcome in outcomes if outcome["name"] == f"Under {threshold}")

        prob_implicita_over = calcular_probabilidade_implicita(odd_over)
        prob_implicita_under = calcular_probabilidade_implicita(odd_under)

        prob_shots_on_goal_over *= 100
        prob_shots_on_goal_under *= 100

        print(f"\n📊 Comparação de Probabilidades (Over/Under {threshold} Chutes ao Gol):")
        print(f"Over {threshold}: Prevista {prob_shots_on_goal_over:.2f}% | Implícita {prob_implicita_over:.2f}%")
        print(f"Under {threshold}: Prevista {prob_shots_on_goal_under:.2f}% | Implícita {prob_implicita_under:.2f}%")

        if prob_shots_on_goal_over > prob_implicita_over:
            oportunidades.append(f"✅ Oportunidade (Chutes ao Gol): Over {threshold} (Odd: {odd_over:.2f}) - Probabilidade Prevista: {prob_shots_on_goal_over:.2f}% > Implícita: {prob_implicita_over:.2f}%")
        if prob_shots_on_goal_under > prob_implicita_under:
            oportunidades.append(f"✅ Oportunidade (Chutes ao Gol): Under {threshold} (Odd: {odd_under:.2f}) - Probabilidade Prevista: {prob_shots_on_goal_under:.2f}% > Implícita: {prob_implicita_under:.2f}%")

    # Mercado Over/Under Faltas
    fouls_market = next((market for market in markets if market["key"] == "fouls"), None)
    if fouls_market:
        threshold = fouls_market["threshold"]
        outcomes = fouls_market["outcomes"]
        odd_over = next(outcome["price"] for outcome in outcomes if outcome["name"] == f"Over {threshold}")
        odd_under = next(outcome["price"] for outcome in outcomes if outcome["name"] == f"Under {threshold}")

        prob_implicita_over = calcular_probabilidade_implicita(odd_over)
        prob_implicita_under = calcular_probabilidade_implicita(odd_under)

        prob_fouls_over *= 100
        prob_fouls_under *= 100

        print(f"\n📊 Comparação de Probabilidades (Over/Under {threshold} Faltas):")
        print(f"Over {threshold}: Prevista {prob_fouls_over:.2f}% | Implícita {prob_implicita_over:.2f}%")
        print(f"Under {threshold}: Prevista {prob_fouls_under:.2f}% | Implícita {prob_implicita_under:.2f}%")

        if prob_fouls_over > prob_implicita_over:
            oportunidades.append(f"✅ Oportunidade (Faltas): Over {threshold} (Odd: {odd_over:.2f}) - Probabilidade Prevista: {prob_fouls_over:.2f}% > Implícita: {prob_implicita_over:.2f}%")
        if prob_fouls_under > prob_implicita_under:
            oportunidades.append(f"✅ Oportunidade (Faltas): Under {threshold} (Odd: {odd_under:.2f}) - Probabilidade Prevista: {prob_fouls_under:.2f}% > Implícita: {prob_implicita_under:.2f}%")
        
        
        if not oportunidades:
            oportunidades.append("ℹ️ Nenhuma oportunidade de aposta identificada.")

    return oportunidades

def processar_confronto(nome_a: str, time_id_a: int, temporada_a: int, nome_b: str, time_id_b: int, temporada_b: int, time_a_mandante: bool) -> None:
    df_a = buscar_jogos_com_season(time_id_a, nome_a, temporada_a)
    df_b = buscar_jogos_com_season(time_id_b, nome_b, temporada_b)

    if df_a is None or df_b is None:
        print("❌ Não foi possível obter dados suficientes para a previsão.")
        return

    print(f"\n🔹 Estatísticas de {nome_a}:")
    estatisticas_a = calcular_media_ajustada(df_a, nome_a)
    for estat, valor in estatisticas_a.items():
        print(f"{estat}: {valor}")

    print(f"\n🔸 Estatísticas de {nome_b}:")
    estatisticas_b = calcular_media_ajustada(df_b, nome_b)
    for estat, valor in estatisticas_b.items():
        print(f"{estat}: {valor}")

    print("\n📈 Previsão estatística:")
    previsao = prever_placar(nome_a, df_a, nome_b, df_b, time_a_mandante)
    lambda_a = previsao["lambda_a"]
    lambda_b = previsao["lambda_b"]
    placar_mais_provavel = previsao["placar_mais_provavel"]
    medias_estatisticas = previsao["medias_estatisticas"]

    print(f"\n🎯 Previsão de placar mais provável ({nome_a} vs {nome_b}):")
    print(f"Placar: {nome_a} {placar_mais_provavel[0]} x {placar_mais_provavel[1]} {nome_b}")

    # Calcular probabilidades para os mercados
    prob_vitoria_a, prob_empate, prob_vitoria_b = calcular_probabilidades_1x2(lambda_a, lambda_b)
    prob_over, prob_under = calcular_probabilidade_over_under(lambda_a, lambda_b, 2.5)
    prob_btts_yes, prob_btts_no = calcular_probabilidade_btts(lambda_a, lambda_b)
    
    # Inicializar probabilidades para outros mercados
    prob_corners_over = prob_corners_under = 0.0
    prob_cards_over = prob_cards_under = 0.0
    prob_total_shots_over = prob_total_shots_under = 0.0
    prob_shots_on_goal_over = prob_shots_on_goal_under = 0.0
    prob_fouls_over = prob_fouls_under = 0.0

    # Buscar odds apenas uma vez
    odds = buscar_odds(nome_a, nome_b, time_id_a, time_id_b, temporada_a)
    
    if odds:
        # Ajustar probabilidades com base nos thresholds das odds
        markets = odds["bookmakers"][0]["markets"]
        
        # Escanteios
        corners_market = next((market for market in markets if market["key"] == "corners"), None)
        if corners_market:
            threshold_corners = corners_market["threshold"]
            media_corners = medias_estatisticas.get("corners", 0)
            prob_corners_over, prob_corners_under = calcular_probabilidade_over_under_estatistica(media_corners, threshold_corners)
        
        # Cartões
        cards_market = next((market for market in markets if market["key"] == "cards"), None)
        if cards_market:
            threshold_cards = cards_market["threshold"]
            media_yellow = medias_estatisticas.get("yellow_cards", 0)
            media_red = medias_estatisticas.get("red_cards", 0)
            media_cards = media_yellow + media_red * 2
            prob_cards_over, prob_cards_under = calcular_probabilidade_over_under_estatistica(media_cards, threshold_cards)

        # Finalizações Totais
        total_shots_market = next((market for market in markets if market["key"] == "total_shots"), None)
        if total_shots_market:
            threshold_shots = total_shots_market["threshold"]
            media_shots = medias_estatisticas.get("total_shots", 0)
            prob_total_shots_over, prob_total_shots_under = calcular_probabilidade_over_under_estatistica(media_shots, threshold_shots)

        # Chutes ao Gol
        shots_market = next((market for market in markets if market["key"] == "shots_on_goal"), None)
        if shots_market:
            threshold_shots_on = shots_market["threshold"]
            media_shots_on = medias_estatisticas.get("shots_on_goal", 0)
            prob_shots_on_goal_over, prob_shots_on_goal_under = calcular_probabilidade_over_under_estatistica(media_shots_on, threshold_shots_on)

        # Faltas
        fouls_market = next((market for market in markets if market["key"] == "fouls"), None)
        if fouls_market:
            threshold_fouls = fouls_market["threshold"]
            media_fouls = medias_estatisticas.get("fouls", 0)
            prob_fouls_over, prob_fouls_under = calcular_probabilidade_over_under_estatistica(media_fouls, threshold_fouls)

        print("\n💡 Oportunidades de Apostas:")
        oportunidades = identificar_oportunidades(
            prob_vitoria_a, prob_empate, prob_vitoria_b,
            prob_over, prob_under, prob_btts_yes, prob_btts_no,
            prob_corners_over, prob_corners_under, 
            prob_cards_over, prob_cards_under,
            prob_total_shots_over, prob_total_shots_under,
            prob_shots_on_goal_over, prob_shots_on_goal_under,
            prob_fouls_over, prob_fouls_under,
            odds, nome_a, nome_b
        )
        for oportunidade in oportunidades:
            print(oportunidade)
    else:
        print("\n❌ Não foi possível obter odds para este confronto.")