import requests
import pandas as pd
from time import sleep
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import poisson
from typing import Tuple, Optional, Dict, List, Any
import json
import os
from requests.exceptions import RequestException

# Obter API_KEY para a API-Football
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise ValueError("A variável de ambiente API_KEY não está definida. Configure-a no Streamlit Cloud Secrets.")

BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
CACHE_FILE = "api_cache.json"
REQUEST_LIMIT_PER_MINUTE = 100
REQUEST_INTERVAL = 60 / REQUEST_LIMIT_PER_MINUTE

ESTATISTICAS_MODELO = {
    "shots_on_goal": "Chutes no Gol",
    "total_shots": "Finalizações Totais",
    "ball_possession": "Posse de Bola (%)",
    "corners": "Escanteios",
    "fouls": "Faltas",
    "yellow_cards": "Cartões Amarelos",
    "red_cards": "Cartões Vermelhos",
    "gols": "Gols"
}

COMPETICAO_PESOS = {
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

def load_cache() -> Dict[str, Any]:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache: Dict[str, Any]) -> None:
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def make_api_request(url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cache = load_cache()
    cache_key = f"{url}_{json.dumps(params, sort_keys=True)}"
    
    if cache_key in cache:
        return cache[cache_key]
    
    try:
        sleep(REQUEST_INTERVAL)
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("errors"):
            print(f"❌ Erro da API: {data['errors']}")
            return None
            
        cache[cache_key] = data
        save_cache(cache)
        return data
    except RequestException as e:
        print(f"❌ Falha na requisição: {e}")
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
                        elif tipo in ["shots_on_goal", "total_shots", "corners", "fouls", "yellow_cards", "red_cards"]:
                            try:
                                valor = int(valor)
                                if tipo in ["shots_on_goal"] and valor > 15:
                                    valor = 0
                                elif tipo == "total_shots" and valor > 25:
                                    valor = 0
                                elif tipo == "corners" and valor > 15:
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

def prever_placar(time_a: str, df_a: pd.DataFrame, time_b: str, df_b: pd.DataFrame, time_a_mandante: bool) -> Tuple[float, float, Tuple[int, int]]:
    lambda_a_casa = float(calcular_media_ajustada(df_a, time_a, is_mandante=True).get("Gols Feita", "0 ± 0").split(" ± ")[0])
    lambda_a_fora = float(calcular_media_ajustada(df_a, time_a, is_mandante=False).get("Gols Feita", "0 ± 0").split(" ± ")[0])
    lambda_a = (lambda_a_casa + lambda_a_fora) / 2 if not time_a_mandante else lambda_a_casa

    lambda_b_casa = float(calcular_media_ajustada(df_b, time_b, is_mandante=True).get("Gols Feita", "0 ± 0").split(" ± ")[0])
    lambda_b_fora = float(calcular_media_ajustada(df_b, time_b, is_mandante=False).get("Gols Feita", "0 ± 0").split(" ± ")[0])
    lambda_b = (lambda_b_casa + lambda_b_fora) / 2 if time_a_mandante else lambda_b_fora

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

    return lambda_a, lambda_b, placar_mais_provavel

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

    # Passo 2: Buscar odds para o jogo encontrado
    url_odds = f"{BASE_URL}/odds"
    params = {
        "fixture": fixture_id,
        "bet": 1  # ID 1 geralmente é "Match Winner" (1x2)
    }
    odds_data = make_api_request(url_odds, params)

    if not odds_data or not odds_data.get("response"):
        print(f"❌ Odds não disponíveis para o jogo {time_a} vs {time_b}.")
        return None

    # Estrutura de resposta simulada para compatibilidade com a função identificar_oportunidades
    response = odds_data["response"][0]
    bookmakers = response.get("bookmakers", [])
    if not bookmakers:
        print("❌ Nenhum bookmaker disponível para este jogo.")
        return None

    # Usar o primeiro bookmaker (ex.: Bet365)
    bookmaker = bookmakers[0]
    bets = bookmaker.get("bets", [])
    if not bets or bets[0]["id"] != 1:  # Verificar se é o mercado "Match Winner"
        print("❌ Mercado '1x2' não disponível.")
        return None

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

def identificar_oportunidades(prob_vitoria_a: float, prob_empate: float, prob_vitoria_b: float, odds: Dict[str, Any], time_a: str, time_b: str) -> List[str]:
    oportunidades = []
    bookmakers = odds.get("bookmakers", [])
    if not bookmakers:
        return ["❌ Nenhuma odd disponível para comparação."]

    markets = bookmakers[0].get("markets", [])
    if not markets or markets[0]["key"] != "h2h":
        return ["❌ Mercado '1x2' não disponível."]

    outcomes = markets[0]["outcomes"]
    odd_vitoria_a = odd_empate = odd_vitoria_b = None

    for outcome in outcomes:
        if outcome["name"].lower() == time_a.lower():
            odd_vitoria_a = outcome["price"]
        elif outcome["name"].lower() == time_b.lower():
            odd_vitoria_b = outcome["price"]
        elif outcome["name"].lower() == "draw":
            odd_empate = outcome["price"]

    if not all([odd_vitoria_a, odd_empate, odd_vitoria_b]):
        return ["❌ Odds incompletas para o mercado '1x2'."]

    prob_implicita_a = calcular_probabilidade_implicita(odd_vitoria_a)
    prob_implicita_empate = calcular_probabilidade_implicita(odd_empate)
    prob_implicita_b = calcular_probabilidade_implicita(odd_vitoria_b)

    prob_vitoria_a *= 100  # Converter para porcentagem
    prob_empate *= 100
    prob_vitoria_b *= 100

    print(f"\n📊 Comparação de Probabilidades para {time_a} vs {time_b}:")
    print(f"Vitória {time_a}: Prevista {prob_vitoria_a:.2f}% | Implícita {prob_implicita_a:.2f}%")
    print(f"Empate: Prevista {prob_empate:.2f}% | Implícita {prob_implicita_empate:.2f}%")
    print(f"Vitória {time_b}: Prevista {prob_vitoria_b:.2f}% | Implícita {prob_implicita_b:.2f}%")

    if prob_vitoria_a > prob_implicita_a:
        oportunidades.append(f"✅ Oportunidade: Vitória de {time_a} (Odd: {odd_vitoria_a:.2f}) - Probabilidade Prevista: {prob_vitoria_a:.2f}% > Implícita: {prob_implicita_a:.2f}%")
    if prob_empate > prob_implicita_empate:
        oportunidades.append(f"✅ Oportunidade: Empate (Odd: {odd_empate:.2f}) - Probabilidade Prevista: {prob_empate:.2f}% > Implícita: {prob_implicita_empate:.2f}%")
    if prob_vitoria_b > prob_implicita_b:
        oportunidades.append(f"✅ Oportunidade: Vitória de {time_b} (Odd: {odd_vitoria_b:.2f}) - Probabilidade Prevista: {prob_vitoria_b:.2f}% > Implícita: {prob_implicita_b:.2f}%")

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
    lambda_a, lambda_b, placar_mais_provavel = prever_placar(nome_a, df_a, nome_b, df_b, time_a_mandante)

    print(f"\n🎯 Previsão de placar mais provável ({nome_a} vs {nome_b}):")
    print(f"Placar: {nome_a} {placar_mais_provavel[0]} x {placar_mais_provavel[1]} {nome_b}")

    prob_vitoria_a, prob_empate, prob_vitoria_b = calcular_probabilidades_1x2(lambda_a, lambda_b)

    odds = buscar_odds(nome_a, nome_b, time_id_a, time_id_b, temporada_a)
    if odds:
        print("\n💡 Oportunidades de Apostas:")
        oportunidades = identificar_oportunidades(prob_vitoria_a, prob_empate, prob_vitoria_b, odds, nome_a, nome_b)
        for oportunidade in oportunidades:
            print(oportunidade)
    else:
        print("\n❌ Não foi possível obter odds para este confronto.")