import streamlit as st
import pandas as pd
from PrevFut_1_6 import processar_confronto, buscar_id_time, detectar_temporada_atual
from io import StringIO
import sys

# Configurar página
st.set_page_config(page_title="Previsão de Placar de Futebol", layout="wide")

# Título e descrição
st.title("⚽ Previsão de Placar de Futebol - Versão 1.6")
st.markdown("Digite os times e selecione o mandante para prever o placar, estatísticas e identificar oportunidades de apostas com base em odds de mercado.")

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
            
            # Exibir resultados
            if output:
                st.subheader("📊 Resultado da Previsão")
                
                # Separar seções da saída
                sections = output.split("\n\n")
                for section in sections:
                    if section.startswith("📈 Previsão estatística"):
                        st.markdown(f"**{section.splitlines()[0]}**")
                    elif section.startswith("🔹 Estatísticas de") or section.startswith("🔸 Estatísticas de"):
                        # Extrair nome do time
                        time_nome = section.split(":")[0].split("de ")[1]
                        st.markdown(f"**{time_nome}**")
                        # Criar tabela para estatísticas
                        stats_lines = section.splitlines()[1:]
                        stats_data = []
                        for line in stats_lines:
                            if ":" in line:
                                stat, value = line.split(":", 1)
                                stats_data.append({"Estatística": stat.strip(), "Valor": value.strip()})
                        if stats_data:
                            st.table(stats_data)
                    elif section.startswith("🎯 Previsão de placar"):
                        st.markdown(f"**{section.splitlines()[0]}**")
                        for line in section.splitlines()[1:]:
                            st.write(line)
                    elif section.startswith("💡 Oportunidades de Apostas"):
                        st.markdown(f"**{section.splitlines()[0]}**")
                        for line in section.splitlines()[1:]:
                            st.write(line)
                    elif section.startswith("📊 Comparação de Probabilidades"):
                        st.markdown(f"**{section.splitlines()[0]}**")
                        for line in section.splitlines()[1:]:
                            st.write(line)
                    elif section.strip():
                        st.write(section)
            else:
                st.warning("Nenhuma saída gerada. Verifique os inputs e tente novamente.")