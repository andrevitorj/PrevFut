"""Testes unitários para o módulo PrevFut."""

import pytest
import os
import json
from unittest.mock import patch, MagicMock
from PrevFut_1_6 import (
    load_cache,
    save_cache,
    make_api_request,
    buscar_id_time,
    calcular_probabilidades_1x2,
    calcular_probabilidade_over_under,
    calcular_probabilidade_btts
)

# Fixtures
@pytest.fixture
def mock_cache_file(tmp_path):
    """Cria um arquivo de cache temporário para testes."""
    cache_file = tmp_path / "test_cache.json"
    test_data = {
        "test_key": {"response": [{"team": {"id": 1, "name": "Test Team"}}]}
    }
    cache_file.write_text(json.dumps(test_data))
    return str(cache_file)

@pytest.fixture
def mock_api_response():
    """Mock de resposta da API."""
    return {
        "response": [
            {
                "team": {
                    "id": 123,
                    "name": "Test Team"
                }
            }
        ]
    }

# Testes de Cache
def test_load_cache_existing_file(mock_cache_file):
    """Testa carregamento de cache existente."""
    with patch('PrevFut_1_6.CACHE_FILE', mock_cache_file):
        cache = load_cache()
        assert cache["test_key"]["response"][0]["team"]["name"] == "Test Team"

def test_load_cache_nonexistent_file():
    """Testa carregamento quando cache não existe."""
    with patch('PrevFut_1_6.CACHE_FILE', 'nonexistent.json'):
        cache = load_cache()
        assert cache == {}

def test_save_cache(tmp_path):
    """Testa salvamento do cache."""
    cache_file = tmp_path / "save_test_cache.json"
    test_data = {"test": "data"}
    
    with patch('PrevFut_1_6.CACHE_FILE', str(cache_file)):
        save_cache(test_data)
        assert json.loads(cache_file.read_text()) == test_data

# Testes de API
@patch('requests.get')
def test_make_api_request_success(mock_get, mock_api_response):
    """Testa requisição à API com sucesso."""
    mock_get.return_value.json.return_value = mock_api_response
    mock_get.return_value.raise_for_status = MagicMock()
    
    result = make_api_request("test_url", {})
    assert result == mock_api_response

@patch('requests.get')
def test_make_api_request_error(mock_get):
    """Testa requisição à API com erro."""
    mock_get.side_effect = Exception("API Error")
    result = make_api_request("test_url", {})
    assert result is None

# Testes de Cálculos
def test_calcular_probabilidades_1x2():
    """Testa cálculo de probabilidades 1X2."""
    prob_v, prob_e, prob_d = calcular_probabilidades_1x2(1.5, 1.0)
    assert 0 <= prob_v <= 1
    assert 0 <= prob_e <= 1
    assert 0 <= prob_d <= 1
    assert abs(prob_v + prob_e + prob_d - 1) < 0.01  # Soma deve ser próxima de 1

def test_calcular_probabilidade_over_under():
    """Testa cálculo de probabilidades Over/Under."""
    prob_over, prob_under = calcular_probabilidade_over_under(2.0, 1.5, 2.5)
    assert 0 <= prob_over <= 1
    assert 0 <= prob_under <= 1
    assert abs(prob_over + prob_under - 1) < 0.01

def test_calcular_probabilidade_btts():
    """Testa cálculo de probabilidades BTTS."""
    prob_yes, prob_no = calcular_probabilidade_btts(1.5, 1.5)
    assert 0 <= prob_yes <= 1
    assert 0 <= prob_no <= 1
    assert abs(prob_yes + prob_no - 1) < 0.01

if __name__ == '__main__':
    pytest.main([__file__])
