import requests
import pandas as pd
from time import sleep
from datetime import datetime
import numpy as np
from scipy.stats import poisson
from typing import Tuple, Optional, Dict, List, Any
import json
import os
from requests.exceptions import RequestException

# Obter API_KEY de variável de ambiente
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
                                # Validação mais rigorosa
                                if tipo in ["shots_on_goal"] and valor > 15:
                                    valor = 0
                                elif tipo == "total_shots" and valor > 25:
                                    valor = 0
                                elif tipo == "corners" and valor > 15:
                                    valor = 0
                            except (ValueError, TypeError):
                                valor = 0
                        # Etapa 1: Normalizar pelo peso do adversário
                        if prefixo == ("mandante" if is_mandante else "visitante"):
                            valor = valor * peso_adversario if tipo != "ball_possession" else valor
                        else:
                            valor = valor / peso_adversario if peso_adversario > 0 and tipo != "ball_possession" else valor
                        jogo_dados[chave] = valor

        # Garantir que todas as estatísticas estejam presentes
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
    
    # Filtrar por mando se especificado
    if is_mandante is not None:
        df = df[df["is_mandante"] == is_mandante]
    
    if df.empty:
        for stat_key in ESTATISTICAS_MODELO:
            estatisticas_finais[f"{ESTATISTICAS_MODELO[stat_key]} Feita"] = "N/D"
            estatisticas_finais[f"{ESTATISTICAS_MODELO[stat_key]} Sofrida"] = "N/D"
        return estatisticas_finais

    for stat_key in ESTATISTICAS_MODELO:
        # Corrigir prefixo para time_nome
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
                estatisticas_fin