"""Testes da calibração conformal com dados simulados (sem baixar o dataset)."""

import numpy as np
import pytest

from entrega.intervalo import ajuste_conformal, excessos, resumo


def test_excessos_dentro_e_fora():
    real = np.array([10, 30, 50])
    baixo, alto = np.array([20, 20, 20]), np.array([40, 40, 40])
    # 10 está 10 abaixo do mínimo; 30 está 10 dentro; 50 está 10 acima do máximo
    assert excessos(real, baixo, alto).tolist() == [10, -10, 10]


def test_ajuste_conformal_garante_cobertura_em_dados_novos():
    """Propriedade central: depois de somar a folga, ~80% dos casos NOVOS ficam dentro."""
    rng = np.random.default_rng(1)
    def sorteia(n):
        real = rng.normal(30, 8, n)
        return real, np.full(n, 27.0), np.full(n, 33.0)  # intervalo "chute" estreito demais

    real_cal, baixo, alto = sorteia(5000)
    folga = ajuste_conformal(excessos(real_cal, baixo, alto))
    assert folga > 0  # o chute estreito precisa ser alargado

    real_novo, baixo, alto = sorteia(20000)
    dentro = ((real_novo >= baixo - folga) & (real_novo <= alto + folga)).mean()
    assert dentro == pytest.approx(0.80, abs=0.02)


def test_ajuste_conformal_pode_ser_negativo_quando_intervalo_largo_demais():
    rng = np.random.default_rng(2)
    real = rng.normal(30, 1, 2000)
    folga = ajuste_conformal(excessos(real, np.full(2000, 10.0), np.full(2000, 50.0)))
    assert folga < 0  # intervalo exagerado: pode encolher


def test_resumo_separa_os_dois_tipos_de_erro():
    real = np.array([10, 25, 25, 50])
    r = resumo(real, np.full(4, 20.0), np.full(4, 30.0))
    assert r["cobertura_pct"] == 50.0
    assert r["chegou_depois_do_maximo_pct"] == 25.0 and r["chegou_antes_do_minimo_pct"] == 25.0
    assert r["largura_media_min"] == 10.0
