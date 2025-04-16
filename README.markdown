# Previsão Estatística de Partidas de Futebol

Uma aplicação web desenvolvida com Streamlit para prever estatísticas de partidas de futebol com base em dados da API-Football.

## Como Executar

1. Clone o repositório:

   ```bash
   git clone https://github.com/your-username/football-stats-prediction.git
   ```

2. Crie um arquivo `.env` na raiz do projeto com sua chave da API-Football:

   ```plaintext
   API_KEY=your_api_key_here
   ```

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

4. Execute a aplicação:

   ```bash
   streamlit run app.py
   ```

## Deploy no Streamlit Cloud

1. Faça push do repositório para o GitHub.
2. Acesse Streamlit Cloud e conecte seu repositório.
3. Configure a variável de ambiente `API_KEY` nas configurações do app no Streamlit Cloud.
4. Faça o deploy.

## Estrutura do Projeto

- `app.py`: Código principal da aplicação.
- `requirements.txt`: Dependências do projeto.
- `pesos.json`: Pesos das competições para cálculos ponderados.
- `.env.example`: Exemplo de configuração de variáveis de ambiente.
- `assets/`: Pasta para armazenar logos ou outros recursos.

## Funcionalidades

- Seleção de times e temporadas.
- Análise dos 10 últimos jogos de cada time.
- Cálculo de médias simples e ponderadas.
- Previsão de estatísticas com intervalos de confiança.
- Previsão de placar usando modelo de Poisson bivariado.
- Comparação com odds da API-Football para identificar apostas de valor.
- Exportação dos resultados em formato .xlsx.