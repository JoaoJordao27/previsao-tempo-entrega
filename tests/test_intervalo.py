"""Testes da calibração conformal com dados simulados (sem baixar o dataset)."""

import numpy as np
import pytest

from entrega.intervalo import ajuste_conformal, ajuste_inteiro, excessos, limites_inteiros, resumo


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


def test_limites_inteiros_arredondam_para_fora_e_ficam_acima_de_1():
    baixo, alto = limites_inteiros(np.array([20.2, 0.3]), np.array([30.1, 5.0]), folga=0.0)
    assert baixo.tolist() == [20.0, 1.0] and alto.tolist() == [31.0, 5.0]
    # quantis cruzados são reordenados, e a folga alarga os dois lados
    b, a = limites_inteiros(np.array([30.0]), np.array([20.0]), folga=1.0)
    assert (b[0], a[0]) == (19.0, 31.0)  # reordena para [20, 30] e alarga 1 min de cada lado


def test_ajuste_inteiro_cobre_a_meta_em_dados_inteiros():
    """Com alvo inteiro (minutos), calibrar no contínuo e arredondar distorce a cobertura; aqui ela fica na meta."""
    rng = np.random.default_rng(4)
    def sorteia(n):
        real = np.round(rng.normal(30, 8, n))
        return real, np.full(n, 27.4), np.full(n, 33.6)

    real_cal, baixo, alto = sorteia(6000)
    folga = ajuste_inteiro(real_cal, baixo, alto)
    real_novo, baixo, alto = sorteia(30000)
    lo, hi = limites_inteiros(baixo, alto, folga)
    cobertura = ((real_novo >= lo) & (real_novo <= hi)).mean()
    assert 0.79 <= cobertura <= 0.86  # em dados inteiros a cobertura avança em degraus: cumpre a meta sem passar muito


def test_arredondar_depois_do_continuo_muda_a_cobertura():
    """O problema que motivou a calibração em inteiros: a mesma folga contínua rende cobertura bem maior ao arredondar."""
    rng = np.random.default_rng(5)
    real = np.round(rng.normal(30, 8, 40000))
    baixo, alto = np.full(40000, 24.0), np.full(40000, 36.0)
    cont = ((real >= baixo - 0.4) & (real <= alto + 0.4)).mean()
    lo, hi = limites_inteiros(baixo, alto, 0.4)
    inteiro = ((real >= lo) & (real <= hi)).mean()
    assert inteiro > cont
