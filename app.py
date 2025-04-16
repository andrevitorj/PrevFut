import streamlit as st
import pandas as pd
from io import StringIO
import sys
from typing import Tuple, Optional, Dict, Any
from PrevFut_1_6 import (
    processar_confronto,
    buscar_id_time,
    detectar_temporada_atual,
    load_cache
)

# Configuração da página
st.set_page_config(
    page_title="Previsão de Placar de Futebol",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título e descrição
st.title("⚽ Previsão de Placar de Futebol - Versão 1.6")
st.markdown("Digite os times e selecione o mandante para prever o placar, estatísticas e identificar oportunidades de apostas.")

# Formulário de input
with st.form(key="previsao_form"):
    col1, col2 = st.columns(2)
    with col1:
        nome_input_a = st.text_input("Time A (ex.: Aston Villa)", key="time_a")
    with col2:
        nome_input_b = st.text_input("Time B (ex.: Paris Saint Germain)", key="time_b")
    
    mando = st.radio("Mandante:", ["Time A", "Time B"], horizontal=True)
    submit_button = st.form_submit_button("Prever Placar e Analisar Oportunidades")

# Processar previsão
if submit_button:
    if not nome_input_a or not nome_input_b:
        st.error("Por favor, preencha os nomes dos dois times.")
    else:
        with st.spinner("Buscando dados, calculando previsão e analisando odds..."):
            # Capturar saída de print
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            
            # Buscar IDs e temporadas
            time_id_a, nome_a = buscar_id_time(nome_input_a)
            time_id_b, nome_b = buscar_id_time(nome_input_b)
            
            if time_id_a and time_id_b:
                temp_a = detectar_temporada_atual(time_id_a)
                temp_b = detectar_temporada_atual(time_id_b)
                if temp_a and temp_b:
                    time_a_mandante = mando == "Time A"
                    processar_confronto(nome_a, time_id_a, temp_a, nome_b, time_id_b, temp_b, time_a_mandante)
                else:
                    st.error("Não foi possível detectar temporadas para um ou ambos os times.")
            else:
                st.error("Um ou ambos os times não foram encontrados.")
            
            # Restaurar stdout e obter saída
            sys.stdout = old_stdout
            output = mystdout.getvalue()
            
            # Processar resultados
            if output:
                # Criar abas
                tab1, tab2, tab3 = st.tabs(["Previsão", "Estatísticas Detalhadas", "Histórico de Partidas"])
                
                with tab1:
                    st.subheader("📊 Previsão do Confronto")
                    
                    # Seção 1.1 - Placar mais provável
                    placar_section = next(s for s in output.split("\n\n") if s.startswith("🎯 Previsão de placar"))
                    st.markdown(f"**{placar_section.splitlines()[0]}**")
                    st.write(placar_section.splitlines()[1])
                    
                    # Seção 1.2 - Odds dentro do IC 80%
                    st.subheader("📊 Odds com Valor Esperado (IC 80%)")
                    oportunidades_section = next(s for s in output.split("\n\n") if s.startswith("💡 Oportunidades de Apostas"))
                    for line in oportunidades_section.splitlines()[1:]:
                        if "Oportunidade" in line:
                            st.success(line.split(":")[1].strip())
                        else:
                            st.write(line)
                    
                    # Seção 1.3 - Todas as estatísticas previstas
                    st.subheader("📈 Estatísticas Previstas")
                    estat_section = next(s for s in output.split("\n\n") if s.startswith("📈 Previsão estatística"))
                    st.markdown(f"**{estat_section.splitlines()[0]}**")
                    
                    # Ordem de importância das estatísticas
                    estat_ordem = [
                        "Gols Feita", "Gols Sofrida",
                        "Chutes no Gol Feita", "Chutes no Gol Sofrida",
                        "Finalizações Totais Feita", "Finalizações Totais Sofrida",
                        "Posse de Bola (%) Feita", "Posse de Bola (%) Sofrida",
                        "Escanteios Feita", "Escanteios Sofrida",
                        "Faltas Feita", "Faltas Sofrida",
                        "Cartões Amarelos Feita", "Cartões Amarelos Sofrida",
                        "Cartões Vermelhos Feita", "Cartões Vermelhos Sofrida",
                        "Impedimentos Feita", "Impedimentos Sofrida"
                    ]
                    
                    # Processar estatísticas para cada time
                    stats_data = []
                    for section in output.split("\n\n"):
                        if section.startswith("🔹 Estatísticas de") or section.startswith("🔸 Estatísticas de"):
                            time_nome = section.split(":")[0].split("de ")[1]
                            for line in section.splitlines()[1:]:
                                if ":" in line:
                                    stat, value = line.split(":", 1)
                                    stat = stat.strip()
                                    value = value.strip()
                                    # Extrair valores do IC
                                    if "±" in value:
                                        media = float(value.split("±")[0].strip())
                                        desvio = float(value.split("±")[1].split("(")[0].strip())
                                        ic_inf = media - desvio
                                        ic_sup = media + desvio
                                        stats_data.append({
                                            "Time": time_nome,
                                            "Estatística": stat,
                                            "Média": media,
                                            "IC Inferior": ic_inf,
                                            "IC Superior": ic_sup
                                        })
                    
                    # Exibir estatísticas ordenadas
                    for stat in estat_ordem:
                        for time in [nome_a, nome_b]:
                            stat_data = next((item for item in stats_data if item["Estatística"] == stat and item["Time"] == time), None)
                            if stat_data:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric(
                                        label=f"{stat} - {time}",
                                        value=f"{stat_data['Média']:.2f}",
                                        delta=f"IC: {stat_data['IC Inferior']:.2f}-{stat_data['IC Superior']:.2f}"
                                    )
                
                with tab2:
                    st.subheader("📋 Estatísticas Detalhadas por Time")
                    
                    # Seção 2.1 - Estatísticas ordenadas por importância
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"### {nome_a} (Mandante)" if time_a_mandante else f"### {nome_a} (Visitante)")
                        df_a = pd.DataFrame([item for item in stats_data if item["Time"] == nome_a])
                        df_a = df_a[df_a["Estatística"].isin(estat_ordem)].sort_values(
                            by="Estatística", 
                            key=lambda x: x.map({v:i for i,v in enumerate(estat_ordem)})
                        )
                        st.dataframe(
                            df_a[["Estatística", "Média", "IC Inferior", "IC Superior"]].rename(columns={
                                "Estatística": "Estatística",
                                "Média": "Média",
                                "IC Inferior": "Mínimo",
                                "IC Superior": "Máximo"
                            }),
                            hide_index=True,
                            use_container_width=True
                        )
                    
                    with col2:
                        st.markdown(f"### {nome_b} (Mandante)" if not time_a_mandante else f"### {nome_b} (Visitante)")
                        df_b = pd.DataFrame([item for item in stats_data if item["Time"] == nome_b])
                        df_b = df_b[df_b["Estatística"].isin(estat_ordem)].sort_values(
                            by="Estatística", 
                            key=lambda x: x.map({v:i for i,v in enumerate(estat_ordem)})
                        )
                        st.dataframe(
                            df_b[["Estatística", "Média", "IC Inferior", "IC Superior"]].rename(columns={
                                "Estatística": "Estatística",
                                "Média": "Média",
                                "IC Inferior": "Mínimo",
                                "IC Superior": "Máximo"
                            }),
                            hide_index=True,
                            use_container_width=True
                        )
                
                with tab3:
                    st.subheader("📅 Histórico de Partidas Recentes")
                    
                    # Seção 3 - Histórico de partidas
                    for section in output.split("\n\n"):
                        if "jogos já realizados retornados para" in section:
                            st.markdown(f"**{section.splitlines()[0]}**")
                            jogos = section.splitlines()[1:]
                            for jogo in jogos:
                                if jogo.strip():
                                    st.write(jogo)
            else:
                st.warning("Nenhuma saída gerada. Verifique os inputs e tente novamente.")