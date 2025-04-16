# Web App de Previsão Estatística de Partidas de Futebol

Este projeto é um aplicativo Streamlit para previsão estatística de partidas de futebol, utilizando dados da API-Football e exportação para Excel.

## Como rodar localmente

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Crie o arquivo `.streamlit/secrets.toml` ou use `.env` com sua API Key:
   ```toml
   API_FOOTBALL_KEY = "SUA_API_KEY_AQUI"
   ```
   Ou edite o arquivo `.env.example`.
3. Execute o app:
   ```bash
   streamlit run main.py
   ```

## Estrutura
- `main.py`: Código principal da aplicação
- `pesos.json`: Pesos de dificuldade por competição
- `requirements.txt`: Dependências
- `.env.example`: Exemplo de configuração de API Key

## Deploy no Streamlit Cloud
1. Faça o push de todos os arquivos para o seu repositório GitHub.
2. No Streamlit Cloud, conecte o repositório e defina a variável de ambiente `API_FOOTBALL_KEY`.
3. O app estará pronto para uso.

## Funcionalidades
- Seleção manual de times e temporadas
- Coleta dos 10 últimos jogos (mandante/visitante)
- Estatísticas detalhadas
- Médias simples e ponderadas
- Previsão estatística e placar provável (Poisson)
- Odds x previsão e apostas de valor
- Exportação para Excel

---

Dúvidas ou sugestões? Abra uma issue no repositório!
