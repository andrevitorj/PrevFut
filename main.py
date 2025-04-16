import streamlit as st
import pandas as pd
import requests
import json

# Carregar pesos das competições
with open('pesos.json', 'r', encoding='utf-8') as f:
    PESOS = json.load(f)

API_KEY = st.secrets["API_FOOTBALL_KEY"] if "API_FOOTBALL_KEY" in st.secrets else "SUA_API_KEY_AQUI"
API_URL = "https://v3.football.api-sports.io/"

st.set_page_config(page_title="Previsão Estatística Futebol", layout="wide", page_icon="⚽")
st.title("Previsão Estatística de Partidas de Futebol")

# Tabs principais
abas = st.tabs([
    "Seleção dos Times e Temporada",
    "Jogos Analisados",
    "Médias",
    "Estatísticas Previstas",
    "Placar Provável",
    "Odds x Previsão",
    "Exportar Resultado"
])

with abas[0]:
    st.header("Seleção dos Times e Temporada")
    st.markdown("Digite os nomes dos times e selecione as temporadas (2020-2025):")
    time_a_nome = st.text_input("Time A (Mandante)")
    time_b_nome = st.text_input("Time B (Visitante)")
    col1, col2 = st.columns(2)
    with col1:
        temporada_a = st.selectbox("Temporada Time A", options=[str(y) for y in range(2025, 2019, -1)])
    with col2:
        temporada_b = st.selectbox("Temporada Time B", options=[str(y) for y in range(2025, 2019, -1)])
    buscar = st.button("Buscar Times")
    times_a, times_b = [], []
    erro_a, erro_b = None, None
    headers = {
        "x-apisports-key": API_KEY
    }
    def buscar_times(nome, temporada):
        try:
            url = f"{API_URL}teams?search={nome}&season={temporada}"
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            if data.get("response"):
                return data["response"], None
            else:
                return [], "Nenhum time encontrado."
        except Exception as e:
            return [], f"Erro na busca: {e}"
    if buscar and time_a_nome and time_b_nome:
        with st.spinner("Buscando times..."):
            times_a, erro_a = buscar_times(time_a_nome, temporada_a)
            times_b, erro_b = buscar_times(time_b_nome, temporada_b)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Times compatíveis - Mandante")
        if erro_a:
            st.error(erro_a)
        elif times_a:
            opcoes_a = [f"{t['team']['name']} ({t['team']['country']}, {t['league']['name']})" for t in times_a]
            idx_a = st.selectbox("Selecione o Time A", options=range(len(opcoes_a)), format_func=lambda i: opcoes_a[i])
            st.image(times_a[idx_a]['team']['logo'], width=60)
        else:
            st.info("Digite e busque pelo nome do time.")
    with col2:
        st.subheader("Times compatíveis - Visitante")
        if erro_b:
            st.error(erro_b)
        elif times_b:
            opcoes_b = [f"{t['team']['name']} ({t['team']['country']}, {t['league']['name']})" for t in times_b]
            idx_b = st.selectbox("Selecione o Time B", options=range(len(opcoes_b)), format_func=lambda i: opcoes_b[i])
            st.image(times_b[idx_b]['team']['logo'], width=60)
        else:
            st.info("Digite e busque pelo nome do time.")

with abas[1]:
    st.header("Jogos Analisados")
    if 'times_a' in locals() and 'times_b' in locals() and times_a and times_b:
        if 'idx_a' in locals() and 'idx_b' in locals():
            time_a_id = times_a[idx_a]['team']['id']
            time_b_id = times_b[idx_b]['team']['id']
            temporada_a = times_a[idx_a]['league']['season']
            temporada_b = times_b[idx_b]['league']['season']
            headers = {"x-apisports-key": API_KEY}
            def buscar_jogos(time_id, temporada, mando):
                try:
                    if mando == 'casa':
                        url = f"{API_URL}fixtures?team={time_id}&season={temporada}&venue=home&status=FT"
                    else:
                        url = f"{API_URL}fixtures?team={time_id}&season={temporada}&venue=away&status=FT"
                    r = requests.get(url, headers=headers, timeout=10)
                    r.raise_for_status()
                    data = r.json()
                    jogos = data.get('response', [])
                    return jogos[-10:] if len(jogos) > 10 else jogos, None
                except Exception as e:
                    return [], f"Erro ao buscar jogos: {e}"
            with st.spinner("Buscando jogos do mandante..."):
                jogos_a, erro_ja = buscar_jogos(time_a_id, temporada_a, 'casa')
            with st.spinner("Buscando jogos do visitante..."):
                jogos_b, erro_jb = buscar_jogos(time_b_id, temporada_b, 'fora')
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Mandante - Últimos 10 jogos em casa")
                if erro_ja:
                    st.error(erro_ja)
                elif jogos_a:
                    for j in jogos_a:
    with st.expander(f"{j['fixture']['date'][:10]} - vs {j['teams']['away']['name']} ({j['goals']['home']}x{j['goals']['away']})"):
        st.write(f"Liga: {j['league']['name']} ({j['league']['country']})")
        # Buscar estatísticas detalhadas via API
        try:
            stats_url = f"{API_URL}fixtures/statistics?fixture={j['fixture']['id']}"
            r_stats = requests.get(stats_url, headers=headers, timeout=10)
            r_stats.raise_for_status()
            stats_data = r_stats.json().get('response', [])
        except Exception as e:
            stats_data = []
        if stats_data:
            for team_stats in stats_data:
                st.markdown(f"**{team_stats['team']['name']}**")
                for item in team_stats['statistics']:
                    nome = item['type']
                    valor = item['value'] if item['value'] is not None else '-'
                    st.write(f"{nome}: {valor}")
        else:
            st.info("Sem estatísticas detalhadas disponíveis para este jogo.")
                else:
                    st.info("Nenhum jogo encontrado.")
            with col2:
                st.subheader("Visitante - Últimos 10 jogos fora")
                if erro_jb:
                    st.error(erro_jb)
                elif jogos_b:
                    for j in jogos_b:
    with st.expander(f"{j['fixture']['date'][:10]} - vs {j['teams']['home']['name']} ({j['goals']['home']}x{j['goals']['away']})"):
        st.write(f"Liga: {j['league']['name']} ({j['league']['country']})")
        # Buscar estatísticas detalhadas via API
        try:
            stats_url = f"{API_URL}fixtures/statistics?fixture={j['fixture']['id']}"
            r_stats = requests.get(stats_url, headers=headers, timeout=10)
            r_stats.raise_for_status()
            stats_data = r_stats.json().get('response', [])
        except Exception as e:
            stats_data = []
        if stats_data:
            for team_stats in stats_data:
                st.markdown(f"**{team_stats['team']['name']}**")
                for item in team_stats['statistics']:
                    nome = item['type']
                    valor = item['value'] if item['value'] is not None else '-'
                    st.write(f"{nome}: {valor}")
        else:
            st.info("Sem estatísticas detalhadas disponíveis para este jogo.")
                else:
                    st.info("Nenhum jogo encontrado.")
        else:
            st.info("Selecione os times nas abas anteriores.")
    else:
        st.info("Selecione e busque os times na aba anterior.")

with abas[2]:
    st.header("Médias (Simples e Ponderadas)")
    def calcular_medias(jogos, tipo_mando):
        if not jogos:
            return pd.DataFrame(), pd.DataFrame()
        estatisticas = {}
        estatisticas_pond = {}
        for j in jogos:
            # Buscar estatísticas detalhadas
            try:
                stats_url = f"{API_URL}fixtures/statistics?fixture={j['fixture']['id']}"
                r_stats = requests.get(stats_url, headers=headers, timeout=10)
                r_stats.raise_for_status()
                stats_data = r_stats.json().get('response', [])
            except Exception as e:
                stats_data = []
            # Identificar adversário e competição
            if tipo_mando == 'casa':
                adversario = j['teams']['away']['name']
                campeonato = j['league']['name']
            else:
                adversario = j['teams']['home']['name']
                campeonato = j['league']['name']
            peso = PESOS.get(campeonato, PESOS.get('Outro', 0.5))
            for team_stats in stats_data:
                for item in team_stats['statistics']:
                    nome = item['type']
                    valor = item['value']
                    if valor is None or not isinstance(valor, (int, float)):
                        continue
                    # Médias simples
                    estatisticas.setdefault(team_stats['team']['name'], {}).setdefault(nome, []).append(valor)
                    # Médias ponderadas
                    if team_stats['team']['name'] == (j['teams']['home']['name'] if tipo_mando == 'casa' else j['teams']['away']['name']):
                        # Feitas
                        estatisticas_pond.setdefault(team_stats['team']['name'], {}).setdefault(nome, []).append(valor * peso)
                    else:
                        # Sofridas
                        estatisticas_pond.setdefault(team_stats['team']['name'], {}).setdefault(nome, []).append(valor / peso if peso else valor)
        # Médias simples
        medias_simples = {}
        for time, stats in estatisticas.items():
            medias_simples[time] = {k: round(sum(v)/len(v),2) for k,v in stats.items() if v}
        # Médias ponderadas
        medias_pond = {}
        for time, stats in estatisticas_pond.items():
            medias_pond[time] = {k: round(sum(v)/len(v),2) for k,v in stats.items() if v}
        return pd.DataFrame(medias_simples).T, pd.DataFrame(medias_pond).T
    # Buscar jogos já carregados
    jogos_a = []
    jogos_b = []
    if 'jogos_a' in locals():
        jogos_a = jogos_a
    if 'jogos_b' in locals():
        jogos_b = jogos_b
    st.subheader('Mandante')
    df_simples_a, df_pond_a = calcular_medias(jogos_a, 'casa')
    st.write('Médias Simples')
    st.dataframe(df_simples_a)
    st.write('Médias Ponderadas')
    st.dataframe(df_pond_a)
    st.subheader('Visitante')
    df_simples_b, df_pond_b = calcular_medias(jogos_b, 'fora')
    st.write('Médias Simples')
    st.dataframe(df_simples_b)
    st.write('Médias Ponderadas')
    st.dataframe(df_pond_b)

with abas[3]:
    st.header("Estatísticas Previstas")
    def prever_estatisticas(df_pond_a, df_pond_b):
        if df_pond_a.empty or df_pond_b.empty:
            return pd.DataFrame()
        # Assumir que o nome do time mandante e visitante é o primeiro index
        time_a = df_pond_a.index[0]
        time_b = df_pond_b.index[0]
        stats = sorted(list(set(df_pond_a.columns).intersection(set(df_pond_b.columns))))
        previsao = {}
        for stat in stats:
            # Mandante
            feita_a = df_pond_a.loc[time_a, stat] if stat in df_pond_a.columns else 0
            sofrida_b = df_pond_b.loc[time_b, stat] if stat in df_pond_b.columns else 0
            prev_a = ((feita_a + sofrida_b) * 1.2) / 2
            # Visitante
            feita_b = df_pond_b.loc[time_b, stat] if stat in df_pond_b.columns else 0
            sofrida_a = df_pond_a.loc[time_a, stat] if stat in df_pond_a.columns else 0
            prev_b = ((feita_b + sofrida_a) / 1.2) / 2
            previsao[stat] = {
                'Mandante': round(prev_a,2),
                'Visitante': round(prev_b,2)
            }
        return pd.DataFrame(previsao).T
    # Recuperar médias ponderadas já calculadas
    if 'df_pond_a' in locals() and 'df_pond_b' in locals():
        df_prev = prever_estatisticas(df_pond_a, df_pond_b)
        if not df_prev.empty:
            st.write('Previsão das Estatísticas da Partida')
            st.dataframe(df_prev)
            # Calcular intervalo de confiança 85% (aprox. 1.44 * std, se disponível)
            st.write('Intervalo de Confiança (aprox. 85%)')
            # Para cada estatística, estimar desvio padrão simples das amostras (mandante e visitante)
            def ic_85(media, std):
                return (round(media - 1.44*std,2), round(media + 1.44*std,2))
            ic_dict = {}
            for stat in df_prev.index:
                std_a = df_pond_a[stat].std() if stat in df_pond_a.columns else 0
                std_b = df_pond_b[stat].std() if stat in df_pond_b.columns else 0
                ic_dict[stat] = {
                    'Mandante': ic_85(df_prev.loc[stat, 'Mandante'], std_a),
                    'Visitante': ic_85(df_prev.loc[stat, 'Visitante'], std_b)
                }
            st.dataframe(pd.DataFrame(ic_dict).T)
        else:
            st.info('Calcule as médias ponderadas primeiro.')
    else:
        st.info('Calcule as médias ponderadas primeiro.')

with abas[4]:
    st.header("Placar Provável")
    import numpy as np
    from scipy.stats import poisson
    def prever_placar(df_prev):
        if df_prev is None or df_prev.empty:
            return None, None, None, None
        # Usar previsão de gols
        if 'Gols' in df_prev.index:
            exp_gols_a = df_prev.loc['Gols', 'Mandante']
            exp_gols_b = df_prev.loc['Gols', 'Visitante']
        elif 'Goals' in df_prev.index:
            exp_gols_a = df_prev.loc['Goals', 'Mandante']
            exp_gols_b = df_prev.loc['Goals', 'Visitante']
        else:
            return None, None, None, None
        # Distribuição de Poisson para gols
        max_gols = 6
        prob_matrix = np.zeros((max_gols+1, max_gols+1))
        for i in range(max_gols+1):
            for j in range(max_gols+1):
                prob_matrix[i,j] = poisson.pmf(i, exp_gols_a) * poisson.pmf(j, exp_gols_b)
        # Placar mais provável
        idx = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
        placar_provavel = f"{idx[0]} x {idx[1]}"
        # Probabilidades
        prob_vitoria_a = np.sum(np.tril(prob_matrix, -1))
        prob_vitoria_b = np.sum(np.triu(prob_matrix, 1))
        prob_empate = np.trace(prob_matrix)
        probs = {
            'Vitória Mandante': round(prob_vitoria_a*100,1),
            'Empate': round(prob_empate*100,1),
            'Vitória Visitante': round(prob_vitoria_b*100,1)
        }
        # Intervalo de confiança 85% para gols
        def intervalo_poisson(mu):
            std = np.sqrt(mu)
            return (round(mu-1.44*std,2), round(mu+1.44*std,2))
        ic_a = intervalo_poisson(exp_gols_a)
        ic_b = intervalo_poisson(exp_gols_b)
        return placar_provavel, probs, ic_a, ic_b
    # Recuperar previsão de estatísticas
    if 'df_prev' in locals() and df_prev is not None and not df_prev.empty:
        placar, probs, ic_a, ic_b = prever_placar(df_prev)
        if placar:
            st.write(f'**Placar mais provável:** {placar}')
            st.write('**Probabilidades:**')
            st.write(probs)
            st.write('**Intervalo de confiança (85%) para gols:**')
            st.write({
                'Mandante': ic_a,
                'Visitante': ic_b
            })
        else:
            st.info('Calcule a previsão de gols primeiro.')
    else:
        st.info('Calcule a previsão de gols primeiro.')

with abas[5]:
    st.header("Odds x Previsão")
    import math
    def buscar_odds(fixture_id):
        try:
            url = f"{API_URL}odds?fixture={fixture_id}"
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json().get('response', [])
            return data
        except Exception as e:
            return []
    def odds_para_prob(odd):
        try:
            return round(100/(float(odd)),2) if odd and float(odd)>0 else 0
        except:
            return 0
    # Buscar fixture_id do último jogo futuro entre os times (se disponível)
    fixture_id = None
    if 'jogos_a' in locals() and jogos_a:
        fixture_id = jogos_a[0]['fixture']['id']
    odds_data = buscar_odds(fixture_id) if fixture_id else []
    if odds_data:
        mercados = {}
        for league in odds_data:
            for bookmaker in league.get('bookmakers', []):
                for bet in bookmaker.get('bets', []):
                    mercados[bet['name']] = bet['values']
        # Exibir mercados principais
        st.subheader('Odds dos principais mercados')
        for mercado, valores in mercados.items():
            st.markdown(f'**{mercado}**')
            odds_table = {v['value']: v['odd'] for v in valores}
            st.write(odds_table)
        # Comparar odds 1x2 com previsão
        if 'Match Winner' in mercados or 'Resultado' in mercados:
            mercado_nome = 'Match Winner' if 'Match Winner' in mercados else 'Resultado'
            odds_1x2 = {v['value']: float(v['odd']) for v in mercados[mercado_nome]}
            probs_1x2 = {k: odds_para_prob(v) for k,v in odds_1x2.items()}
            st.write('Probabilidades implícitas das odds (1x2):')
            st.write(probs_1x2)
            # Comparar com modelo
            if 'df_prev' in locals() and df_prev is not None and not df_prev.empty:
                # Usar as probabilidades do modelo calculadas anteriormente
                if 'prob_vitoria_a' in locals() and 'prob_empate' in locals() and 'prob_vitoria_b' in locals():
                    st.write('Probabilidades do modelo:')
                    st.write({
                        'Vitória Mandante': prob_vitoria_a,
                        'Empate': prob_empate,
                        'Vitória Visitante': prob_vitoria_b
                    })
                # Destacar oportunidades de valor
                st.write('**Oportunidades de Valor:**')
                for k in ['Home', 'Draw', 'Away', 'Mandante', 'Empate', 'Visitante']:
                    if k in probs_1x2:
                        odd_prob = probs_1x2[k]
                        # Valor esperado simples: modelo - odd implícita
                        if k == 'Home' or k == 'Mandante':
                            modelo_prob = prob_vitoria_a if 'prob_vitoria_a' in locals() else 0
                        elif k == 'Draw' or k == 'Empate':
                            modelo_prob = prob_empate if 'prob_empate' in locals() else 0
                        elif k == 'Away' or k == 'Visitante':
                            modelo_prob = prob_vitoria_b if 'prob_vitoria_b' in locals() else 0
                        else:
                            modelo_prob = 0
                        if modelo_prob > odd_prob:
                            st.success(f"Aposta de valor em {k}: Modelo={modelo_prob:.1f}% > Odd Implícita={odd_prob:.1f}%")
    else:
        st.info('Odds não encontradas para o confronto.')

with abas[6]:
    st.header("Exportar Resultado")
    import io
    from datetime import datetime
    import xlsxwriter
    st.info("Ao final, exporte os dados para .xlsx.")
    def exportar_para_excel():
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Médias Simples e Ponderadas
            if 'df_simples_a' in locals() and not df_simples_a.empty:
                df_simples_a.to_excel(writer, sheet_name='Medias_Simples_Mandante')
            if 'df_simples_b' in locals() and not df_simples_b.empty:
                df_simples_b.to_excel(writer, sheet_name='Medias_Simples_Visitante')
            if 'df_pond_a' in locals() and not df_pond_a.empty:
                df_pond_a.to_excel(writer, sheet_name='Medias_Ponderadas_Mandante')
            if 'df_pond_b' in locals() and not df_pond_b.empty:
                df_pond_b.to_excel(writer, sheet_name='Medias_Ponderadas_Visitante')
            # Estatísticas previstas
            if 'df_prev' in locals() and not df_prev.empty:
                df_prev.to_excel(writer, sheet_name='Estatisticas_Previstas')
            # Intervalo de confiança
            if 'ic_dict' in locals() and ic_dict:
                pd.DataFrame(ic_dict).T.to_excel(writer, sheet_name='IC_85')
            # Placar provável
            if 'placar' in locals() and placar:
                pd.DataFrame({'Placar Provável': [placar]}).to_excel(writer, sheet_name='Placar_Provavel')
            if 'probs' in locals() and probs:
                pd.DataFrame([probs]).to_excel(writer, sheet_name='Probabilidades_Placar')
            # Odds e destaques de valor
            if 'mercados' in locals() and mercados:
                for mercado, valores in mercados.items():
                    pd.DataFrame(valores).to_excel(writer, sheet_name=f'Odds_{mercado[:20]}')
            # Oportunidades de valor
            # (Opcional: pode-se criar uma lista durante a análise de odds e salvar aqui)
            writer.save()
        output.seek(0)
        return output
    if st.button('Exportar para Excel (.xlsx)'):
        excel_data = exportar_para_excel()
        st.download_button(
            label='Baixar arquivo Excel',
            data=excel_data,
            file_name=f'previsao_futebol_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
