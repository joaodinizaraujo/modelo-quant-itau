"""
Indicador de notícia (-1 a 1) para o modelo AÇÃO × WTI.

Este módulo consome um CSV de notícias JÁ PONTUADAS e devolve uma série diária
alinhada ao calendário do modelo, pronta para virar o 4º voto do score de convicção.

    fonte, manchete, datahora, sinal

O classificador (manchete -> sinal) NÃO está aqui: nesta POC a coluna `sinal` é
preenchida à mão. A rubrica abaixo é a especificação dessa rotulação e, mais
adiante, vira o prompt do classificador automático sem reescrita.


================================================================================
RUBRICA DE CLASSIFICAÇÃO
================================================================================

CONVENÇÃO DE SINAL
------------------
    +1  altista para o complexo de petróleo (WTI / ações de energia sobem)
    -1  baixista
    vazio (nulo)  irrelevante para o modelo -> a linha é descartada

O sinal é sempre a direção do PETRÓLEO, nunca "boa notícia / má notícia".
Lucro recorde da XOM é boa notícia para a XOM, mas não diz nada sobre o crude.

  Limitação conhecida: crude em alta é bullish para upstream (XOM, CVX, COP...)
  mas comprime o crack spread do refino (MPC, PSX, VLO). Como este sinal é macro
  — uma série só para os 13 tickers — essa assimetria não é capturada. O
  `multiplicador_risco` por segmento no modelo (1.0 / 0.6 / 0.3) já atenua
  parcialmente. Um sinal por ticker é candidato a uma v5, não se resolve aqui.


GATE DE RELEVÂNCIA — três perguntas, todas precisam de "sim"
------------------------------------------------------------
Se qualquer uma falhar, o sinal é NULO e a notícia é descartada.

1. CANAL DE TRANSMISSÃO. A manchete toca algum destes?
     Oferta       OPEP+ (cotas, cortes, compliance), sanções (Rússia, Irã,
                  Venezuela), interrupção física (furacão no Golfo, ataque a
                  refinaria/oleoduto, greve), xisto americano, rig count
     Demanda      atividade de China/EUA/Europa enquadrada em consumo de
                  energia, driving season, aviação, recessão
     Estoques     EIA/API semanais, SPR (liberação ou recompra)
     Geopolítica  conflito em região produtora, Estreito de Ormuz, Mar Vermelho, Donald Trump brincando com o cenário americano
     Regulatório  tarifas, licenças de perfuração, windfall tax, política
                  energética dos EUA
     Setorial     M&A grande entre majors, corte coletivo de capex

2. NOVIDADE. Traz informação nova? Descrição de preço passado ("petróleo fecha
   em alta de 2%") é EFEITO, não causa -> nulo. Opinião de analista sem fato
   novo -> nulo.

3. HORIZONTE. O efeito é plausível em DIAS A SEMANAS? O modelo usa z-score de 20
   dias e segura posição por dias. Transição energética para 2050 -> nulo.

LISTA NEGATIVA (nulo direto, sem avaliar): resumo de fechamento de mercado,
análise técnica/gráfica, outra commodity sem ligação, ESG institucional sem
efeito operacional, release corporativo rotineiro (dividendo, troca de CFO),
duplicata de manchete já pontuada no mesmo dia.

REGRA DE AMBIGUIDADE: o input é só a manchete. Se a direção depende de contexto
que não está nela, o sinal é nulo. Na dúvida, nulo — o custo de um rótulo errado
é maior que o de uma notícia perdida.


ESCALA DE MAGNITUDE (ancoragem)
-------------------------------
Sem âncora explícita a rotulação vira ruído. Três níveis, com sinal:

  ±1.0  Choque de 1ª ordem, muda o balanço global de oferta.
        Corte/aumento surpresa e grande da OPEP+; guerra afetando produção ou
        trânsito; embargo amplo.

  ±0.6  Material, mas de escopo limitado ou parcialmente precificado.
        Sanção incremental; furacão com parada temporária; reunião da OPEP+
        dentro do esperado mas com viés.

  ±0.3  Informativo incremental.
        Estoques semanais fora do consenso; dado de demanda da China; rig count.

Valores intermediários são permitidos (o campo é contínuo), mas os três níveis
são a âncora. `0.0` não deve ser usado: se não move nada, é NULO.
"""

import numpy as np
import pandas as pd

COLUNAS = ['fonte', 'manchete', 'datahora', 'sinal']

FUSO_PREGAO = 'America/New_York'
HORA_CORTE = 16          # fechamento do pregão em NY
MEIA_VIDA_DIAS = 3       # em quantos dias o impacto de uma manchete cai pela metade
LIMIAR_NOTICIA = 0.15    # zona morta do voto, no mesmo espírito do LIMIAR_TERMO


def carregar_csv(caminho, verbose=True):
    """Lê o CSV de notícias pontuadas e devolve só as linhas utilizáveis.

    Linhas com `sinal` vazio são notícias que passaram pelo classificador e foram
    consideradas irrelevantes — ficam no arquivo para auditoria, mas saem daqui.
    """
    noticias = pd.read_csv(caminho)

    faltando = [c for c in COLUNAS if c not in noticias.columns]
    if faltando:
        raise ValueError(f'CSV sem as colunas obrigatórias: {faltando}')

    noticias = noticias[COLUNAS].copy()
    noticias['datahora'] = _parse_datahora(noticias['datahora'])
    noticias['sinal'] = pd.to_numeric(noticias['sinal'], errors='coerce')

    total = len(noticias)
    noticias = noticias.dropna(subset=['datahora', 'sinal'])
    noticias['sinal'] = noticias['sinal'].clip(-1.0, 1.0)
    noticias = noticias.sort_values('datahora').reset_index(drop=True)

    if verbose:
        print(f'📰 Notícias lidas: {total} | pontuadas: {len(noticias)} | '
              f'descartadas (sinal nulo): {total - len(noticias)}')
    return noticias


def _parse_datahora(coluna):
    # A base pode vir toda com fuso explícito (ideal) ou toda sem fuso (aí
    # assumimos o horário de NY). Misturar os dois é ambíguo demais para adivinhar.
    bruto = coluna.astype('string').str.strip()
    tem_fuso = bruto.str.contains(r'(?:Z|[+-]\d{2}:?\d{2})$', na=False)

    if tem_fuso.all():
        return pd.to_datetime(bruto, utc=True).dt.tz_convert(FUSO_PREGAO)
    if not tem_fuso.any():
        return pd.to_datetime(bruto).dt.tz_localize(
            FUSO_PREGAO, ambiguous=True, nonexistent='shift_forward')
    raise ValueError('A coluna datahora mistura valores com e sem fuso horário. '
                     'Padronize a base (de preferência com offset explícito).')


def serie_diaria(noticias, calendario, meia_vida=MEIA_VIDA_DIAS, hora_corte=HORA_CORTE):
    """Agrega as notícias no calendário do modelo e aplica o decaimento.

    `calendario` é o índice de pregão do modelo (df.index): datas tz-naive,
    normalizadas à meia-noite.

    Duas decisões que evitam look-ahead e ruído:

    - Corte de sessão: notícia publicada até `hora_corte` conta para o próprio
      dia; depois disso, para o pregão seguinte. Fim de semana e feriado saem de
      graça, porque a data-alvo é mapeada para o primeiro dia do calendário >= ela.
    - Agregação do dia por SOMA CLIPADA em [-1, 1]: manchetes concordantes
      acumulam até saturar (a média achataria cinco manchetes na mesma direção).
    """
    calendario = pd.DatetimeIndex(calendario)
    diario = pd.Series(0.0, index=calendario)

    if len(noticias) > 0:
        local = noticias['datahora'].dt.tz_convert(FUSO_PREGAO).dt.tz_localize(None)
        passou_do_corte = (local.dt.hour >= hora_corte).astype(int)
        data_alvo = local.dt.normalize() + pd.to_timedelta(passou_do_corte, unit='D')

        pos = calendario.searchsorted(data_alvo)
        dentro = pos < len(calendario)  # notícias além do fim do calendário caem fora

        por_dia = noticias.loc[dentro, 'sinal'].groupby(calendario[pos[dentro]]).sum()
        diario = por_dia.clip(-1.0, 1.0).reindex(calendario).fillna(0.0)

    # Estoque de notícia: entra cheio no dia e decai com meia-vida de `meia_vida`
    # dias. Loop explícito em vez de ewm() porque ewm normaliza e amorteceria o
    # impacto do próprio dia da manchete.
    fator = 0.5 ** (1 / meia_vida)
    acumulado = diario.copy()
    for i in range(1, len(acumulado)):
        acumulado.iloc[i] = np.clip(
            acumulado.iloc[i] + acumulado.iloc[i - 1] * fator, -1.0, 1.0)

    return acumulado


def regime_noticia(serie, limiar=LIMIAR_NOTICIA):
    """Converte o score contínuo no voto discreto -1 / 0 / +1."""
    regime = pd.Series(0, index=serie.index)
    regime[serie > limiar] = 1
    regime[serie < -limiar] = -1
    return regime


def carregar_regime(caminho, calendario, meia_vida=MEIA_VIDA_DIAS,
                    limiar=LIMIAR_NOTICIA, verbose=True):
    """Atalho: CSV -> voto -1/0/+1 alinhado ao calendário. É o que o modelo chama."""
    noticias = carregar_csv(caminho, verbose=verbose)
    serie = serie_diaria(noticias, calendario, meia_vida=meia_vida)
    regime = regime_noticia(serie, limiar=limiar)

    if verbose:
        print(f'   Tilt bullish (+1): {(regime == 1).mean():.1%} dos dias')
        print(f'   Tilt bearish (-1): {(regime == -1).mean():.1%} dos dias')
        print(f'   Zona morta / sem notícia: {(regime == 0).mean():.1%} dos dias')
    return regime


if __name__ == '__main__':
    # Exercita o pipeline contra um calendário sintético de dias úteis.
    CAMINHO = 'noticias_exemplo.csv'

    noticias = carregar_csv(CAMINHO)
    calendario = pd.bdate_range('2024-01-01', '2025-12-31')
    serie = serie_diaria(noticias, calendario)
    regime = regime_noticia(serie)

    print(f'\n📅 Calendário sintético: {len(calendario)} dias úteis '
          f'({calendario[0].date()} → {calendario[-1].date()})')
    print(f'   Tilt bullish (+1): {(regime == 1).mean():.1%} dos dias')
    print(f'   Tilt bearish (-1): {(regime == -1).mean():.1%} dos dias')
    print(f'   Zona morta / sem notícia: {(regime == 0).mean():.1%} dos dias')

    print('\n🔎 Mapeamento de cada notícia para o dia de pregão efetivo:')
    local = noticias['datahora'].dt.tz_convert(FUSO_PREGAO).dt.tz_localize(None)
    alvo = local.dt.normalize() + pd.to_timedelta((local.dt.hour >= HORA_CORTE).astype(int), unit='D')
    pos = calendario.searchsorted(alvo)
    for i in range(len(noticias)):
        efetivo = calendario[pos[i]].date() if pos[i] < len(calendario) else 'fora do calendário'
        print(f'   {local.iloc[i]:%Y-%m-%d %H:%M} ({local.iloc[i]:%a}) '
              f'sinal={noticias["sinal"].iloc[i]:+.1f} -> {efetivo}')

    janela = serie[serie != 0].head(20)
    print('\n📉 Primeiros 20 dias com sinal acumulado (dá para ver o decaimento):')
    for data, valor in janela.items():
        print(f'   {data.date()}  {valor:+.3f}  {"#" * int(abs(valor) * 40)}')
