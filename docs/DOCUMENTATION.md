# Documentação do PrevFut

## Visão Geral

O PrevFut é uma aplicação web desenvolvida em Python usando Streamlit para prever resultados de partidas de futebol e identificar oportunidades de apostas. O sistema utiliza dados históricos e estatísticas em tempo real da API-Football para fazer suas análises.

## Estrutura do Projeto

```
PrevFut/
├── app.py                 # Interface Streamlit
├── PrevFut_1_6.py        # Lógica principal
├── config.py             # Configurações
├── requirements.txt      # Dependências
├── .env.example         # Template de variáveis de ambiente
├── .gitignore           # Arquivos ignorados pelo git
├── tests/               # Testes unitários
│   └── test_prevfut.py
├── docs/                # Documentação
│   └── DOCUMENTATION.md
└── README.md           # Instruções básicas
```

## Componentes Principais

### 1. Interface (app.py)

A interface web é construída usando Streamlit e oferece:
- Seleção de times com autocomplete
- Escolha do mandante
- Exibição de resultados em abas organizadas
- Cache de dados para melhor performance
- Feedback visual do processamento

### 2. Lógica Principal (PrevFut_1_6.py)

Módulo principal com as seguintes funcionalidades:
- Integração com API-Football
- Sistema de cache para requisições
- Cálculos estatísticos
- Análise de probabilidades
- Identificação de oportunidades

#### Funções Principais:

##### `buscar_id_time(nome_busca: str) -> Tuple[Optional[int], Optional[str]]`
Busca o ID de um time na API-Football.
- **Parâmetros**: nome do time
- **Retorno**: ID do time e nome oficial

##### `processar_confronto(nome_a, time_id_a, temporada_a, nome_b, time_id_b, temporada_b, time_a_mandante)`
Processa todas as informações do confronto.
- Busca estatísticas
- Calcula probabilidades
- Analisa oportunidades
- Gera previsões

##### `calcular_probabilidades_1x2(lambda_a: float, lambda_b: float)`
Calcula probabilidades de vitória, empate e derrota.
- Usa distribuição de Poisson
- Considera força dos times

### 3. Configurações (config.py)

Centraliza todas as configurações do sistema:
- Constantes da API
- Pesos por competição
- Configurações de cache
- Mapeamento de estatísticas
- Thresholds para análise

## Fluxo de Dados

1. **Entrada de Dados**
   - Usuário seleciona times
   - Sistema busca IDs na API

2. **Processamento**
   - Busca estatísticas recentes
   - Calcula médias e tendências
   - Aplica modelo estatístico
   - Analisa odds de mercado

3. **Saída**
   - Previsão de placar
   - Estatísticas comparativas
   - Oportunidades identificadas

## Cache

O sistema implementa dois níveis de cache:
1. **Cache de API**: Salva respostas da API-Football
2. **Cache Streamlit**: Otimiza componentes da interface

## Logging

Sistema de logging estruturado:
- Rotação de arquivos
- Níveis de log configuráveis
- Formato padronizado
- Rastreamento de erros

## Testes

Suite de testes unitários cobrindo:
- Funções de cache
- Requisições à API
- Cálculos estatísticos
- Formatação de dados

## Variáveis de Ambiente

Requeridas:
- `API_KEY`: Chave da API-Football

Opcionais:
- `DEBUG`: Modo debug (True/False)
- `CACHE_EXPIRY`: Tempo de expiração do cache

## Manutenção

### Atualizando Dependências
```bash
pip install -r requirements.txt --upgrade
```

### Rodando Testes
```bash
pytest tests/
```

### Limpando Cache
```bash
rm api_cache.json
```

## Troubleshooting

### Erros Comuns

1. **API Key Inválida**
   - Verifique o arquivo .env
   - Confirme a chave no painel da API-Football

2. **Cache Corrompido**
   - Delete o arquivo api_cache.json
   - Reinicie a aplicação

3. **Time Não Encontrado**
   - Verifique a grafia exata
   - Use o nome oficial do time

## Próximas Melhorias

1. Implementar mais mercados de apostas
2. Adicionar análise de forma recente
3. Melhorar precisão das previsões
4. Expandir cobertura de testes
5. Adicionar mais validações de entrada
