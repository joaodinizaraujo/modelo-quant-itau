"""
Indicador de notícia (-1 a 1, em passos de 0,1) para o modelo AÇÃO × WTI.

O módulo faz duas coisas, nesta ordem:

1. CLASSIFICAR — lê um CSV de manchetes cruas (`jornal, manchete, data, hora`),
   manda para o Gemini com a rubrica e grava um CSV pontuado.
       python news_signal.py classificar brutas.csv pontuadas.csv

2. CONSUMIR — lê o CSV pontuado (as mesmas colunas + `sinal`) e devolve uma
   série diária alinhada ao calendário do modelo, pronta para virar o 4º voto
   do score de convicção da v4.

`data` e `hora` são colunas separadas e são combinadas na leitura. Se `hora`
trouxer offset de fuso (`09:15:00-05:00`), ele é respeitado; sem offset, assume-se
o horário de Nova York.

Os critérios de classificação estão na constante RUBRICA, logo abaixo. Ela é a
ÚNICA fonte da verdade: é o que documenta o método e é literalmente o texto
enviado à LLM. Editar a rubrica muda o prompt na mesma hora — e invalida o cache
de classificações, então nada de rótulo velho misturado com novo.
"""

import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

RUBRICA = """\
Você classifica manchetes de notícia quanto ao impacto sobre o preço do petróleo.

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
     Geopolítica  conflito em região produtora, Estreito de Ormuz, Mar Vermelho;
                  e ação do governo dos EUA com efeito DIRETO sobre oferta ou
                  demanda de energia (tarifas, sanções, licenças de perfuração,
                  ação militar contra país produtor). Política doméstica sem esse
                  elo — eleição, pesquisa, disputa partidária, nomeação, escândalo
                  — é NULA, por mais que envolva figuras de destaque.
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
A magnitude é um INTEIRO DE 1 A 10, em décimos: 10 = 1,0 e 3 = 0,3. O sinal
final é sempre múltiplo de 0,1. Sem âncora explícita uma escala de 1 a 10 vira
ruído, então use estes três pontos para se calibrar:

  10  Choque de 1ª ordem, muda o balanço global de oferta.
      Corte/aumento surpresa e grande da OPEP+; guerra afetando produção ou
      trânsito; embargo amplo.

   6  Material, mas de escopo limitado ou parcialmente precificado.
      Sanção incremental; furacão com parada temporária; reunião da OPEP+
      dentro do esperado mas com viés.

   3  Informativo incremental.
      Estoques semanais fora do consenso; dado de demanda da China; rig count.

Os valores entre as âncoras servem para graduar dentro da faixa: um choque um
pouco menor que o máximo é 8 ou 9; um incremental mais forte que o típico é 4.
O 0 não existe na escala — se não move nada, a notícia é NULA (canal NENHUM),
não magnitude zero.
"""

INSTRUCOES_SAIDA = """

================================================================================
COMO RESPONDER
================================================================================

Você recebe um JSON com uma lista de manchetes, cada uma com seu `id`. Devolva
uma classificação para CADA id recebido — nem a mais, nem a menos.

REGRAS INEGOCIÁVEIS:

1. Julgue APENAS pelo texto da manchete. Você não sabe o que aconteceu depois
   dela, e não deve usar nenhum conhecimento externo sobre o evento, sobre o
   preço do petróleo na época ou sobre o desfecho. Se a manchete não diz, você
   não sabe.

2. Não invente números. Não infira magnitudes, volumes ou percentuais que não
   estejam escritos na manchete.

3. Avalie cada manchete ISOLADAMENTE. Não use uma manchete da lista para
   interpretar outra, nem trate a lista como uma sequência de eventos.

4. O campo `trecho` deve ser um pedaço COPIADO LITERALMENTE da manchete, palavra
   por palavra, que sustenta o canal escolhido. Não parafraseie, não corrija, não
   traduza. Ele é conferido automaticamente contra o texto original: se não bater
   exatamente, a classificação é descartada.

5. Quando o canal for NENHUM, ou quando a manchete não passar no gate de
   novidade ou de horizonte, a direção deve ser INDEFINIDA. A notícia será
   descartada de qualquer forma — não force uma direção para "aproveitar" a linha.

6. Na dúvida, descarte. Um rótulo errado custa mais caro que uma notícia perdida.
"""

# =============================================================================
# CONFIGURAÇÃO DO CLASSIFICADOR
# =============================================================================

MODELO_PADRAO = 'gemini-3.5-flash-lite'
TAMANHO_LOTE = 20
TENTATIVAS = 3

CANAIS = ['OFERTA', 'DEMANDA', 'ESTOQUES', 'GEOPOLITICA', 'REGULATORIO',
          'SETORIAL', 'NENHUM']
DIRECAO = {'ALTISTA': 1, 'BAIXISTA': -1, 'INDEFINIDA': 0}

# =============================================================================
# FILTRO LOCAL — corta o que nem vale mandar para a LLM
# =============================================================================
# A base real é um firehose de política americana: 21.285 manchetes, das quais
# 90% citam Trump. Mandar tudo custa ~1.065 chamadas para receber nulo na imensa
# maioria. Estes termos derrubam para ~20% da base (~218 chamadas).
#
# A precisão é baixa DE PROPÓSITO ("cooking oil" passa). O filtro existe para
# cortar custo; quem decide relevância é o gate da rubrica, na LLM. O erro caro
# é o inverso — descartar manchete que valia —, então cada termo aqui foi medido
# contra rótulos reais, e `auditar` remede a qualquer momento.
#
# `trump` NÃO está na lista: aparece em 90,2% da base e não filtraria nada.
TERMOS_RELEVANTES = {
    'OFERTA': r'\b(?:oil|crude|opec\+?|petroleum|barrels?|wti|brent|shale|fracking'
              r'|refin\w+|pipelines?|rig count|drill\w+|output cut|production cut'
              r'|natural gas|lng)\b',
    'DEMANDA': r'\b(?:gasoline|diesel|jet fuel|fuel|heating|recession|china\w*)\b',
    'ESTOQUES': r'\b(?:inventor\w+|stockpiles?|strategic petroleum reserve|spr)\b',
    # Além dos países, os líderes e capitais: `auditar` mostrou manchetes sobre a
    # guerra que citam só "Putin"/"Zelenskyy" e escapavam (custo: +252 manchetes).
    'GEOPOLITICA': r'\b(?:iran\w*|russia\w*|venezuela\w*|saudi\w*|hormuz|red sea|tankers?'
                   r'|sanction\w*|embargo|ukrain\w*|israel\w*|houthi\w*|nigeria\w*|libya\w*'
                   r'|iraq\w*|kuwait\w*|qatar\w*|emirates|uae|gaza\w*|hamas|hezbollah'
                   r'|yemen\w*|syria\w*|leban\w+|opec'
                   r'|middle east|putin|zelensky\w*|kremlin|moscow|kyiv|tehran|khamenei'
                   r'|riyadh|maduro|netanyahu)\b',
    # `energy`, `climate` e `emissions` entraram depois de o `auditar` mostrar que
    # a política energética dos EUA estava escapando por eles (custo: +104 manchetes).
    'REGULATORIO': r'\b(?:tariffs?|drilling permit|offshore lease|windfall tax|keystone'
                   r'|anwr|energy|climate|emissions?|federal reserve|interest rates?)\b',
    'SETORIAL': r'\b(?:exxon\w*|chevron|conocophillips|occidental|halliburton'
                r'|schlumberger|capex)\b',
}

COLUNAS_AUDITORIA = ['canal', 'direcao', 'magnitude_decimos', 'trecho', 'justificativa',
                     'motivo_descarte', 'canais_filtro', 'modelo', 'versao_rubrica']

# Versão calculada, não digitada: mexer na rubrica OU nos termos do filtro invalida
# o cache sozinho. Com uma constante manual dava para esquecer de subir e ficar com
# linhas antigas travadas no cache depois de alargar o filtro.
VERSAO_RUBRICA = hashlib.sha256(
    (RUBRICA + INSTRUCOES_SAIDA + repr(sorted(TERMOS_RELEVANTES.items()))).encode()
).hexdigest()[:8]

# Schema JSON puro em vez de Pydantic: é a mesma coisa para o SDK
# (`response_json_schema` recebe um dict), mas não obriga um import a mais só
# para `import news_signal` funcionar em ambiente sem o google-genai.
# A ordem das propriedades importa: as evidências vêm antes do veredito, para
# o modelo escolher o canal e copiar o trecho ANTES de decidir direção e nota.
ESQUEMA_ITEM = {
    'type': 'object',
    'properties': {
        'id': {'type': 'integer'},
        'canal': {'type': 'string', 'enum': CANAIS},
        'trecho': {'type': 'string'},
        'passa_novidade': {'type': 'boolean'},
        'passa_horizonte': {'type': 'boolean'},
        'direcao': {'type': 'string', 'enum': list(DIRECAO)},
        'magnitude_decimos': {'type': 'integer', 'minimum': 1, 'maximum': 10},
        'justificativa': {'type': 'string'},
    },
    'required': ['id', 'canal', 'trecho', 'passa_novidade', 'passa_horizonte',
                 'direcao', 'magnitude_decimos', 'justificativa'],
    'propertyOrdering': ['id', 'canal', 'trecho', 'passa_novidade',
                         'passa_horizonte', 'direcao', 'magnitude_decimos',
                         'justificativa'],
}

ESQUEMA_RESPOSTA = {
    'type': 'object',
    'properties': {'classificacoes': {'type': 'array', 'items': ESQUEMA_ITEM}},
    'required': ['classificacoes'],
}

# =============================================================================
# CONFIGURAÇÃO DO CONSUMIDOR
# =============================================================================

# Contrato do CSV bruto. O CSV pontuado é isto + `sinal` + as colunas de auditoria.
COLUNAS_ENTRADA = ['jornal', 'manchete', 'data', 'hora']
COLUNAS = COLUNAS_ENTRADA + ['sinal']

FUSO_PREGAO = 'America/New_York'
HORA_CORTE = 16          # fechamento do pregão em NY
MEIA_VIDA_DIAS = 3       # em quantos dias o impacto de uma manchete cai pela metade
LIMIAR_NOTICIA = 0.15    # zona morta do voto, no mesmo espírito do LIMIAR_TERMO
PASSO_SINAL = 0.1        # o sinal é discreto: só múltiplos de 0,1


def filtrar(manchetes):
    """Decide quais manchetes valem uma chamada à LLM.

    Devolve (mascara, canais): a máscara booleana e, por linha, quais canais
    casaram ('GEOPOLITICA|OFERTA'). O segundo vai para o CSV, para dar para
    auditar depois por que cada manchete foi enviada.
    """
    texto = manchetes.astype('string').fillna('')
    acertos = {canal: texto.str.contains(padrao, case=False, na=False, regex=True)
               for canal, padrao in TERMOS_RELEVANTES.items()}

    mascara = pd.Series(False, index=texto.index)
    for acerto in acertos.values():
        mascara |= acerto

    canais = pd.Series('', index=texto.index)
    for canal, acerto in acertos.items():
        canais = canais.mask(acerto, canais.where(canais == '', canais + '|') + canal)
    return mascara, canais


def classificar_csv(entrada, saida, modelo=MODELO_PADRAO, tamanho_lote=TAMANHO_LOTE,
                    usar_cache=True, verbose=True):
    """Lê manchetes cruas, pontua com a LLM e grava o CSV que o modelo consome.

    Entrada:  jornal, manchete, data, hora
    Saída:    as mesmas + sinal + colunas de auditoria (canal, trecho, motivo...)

    Notícias irrelevantes ficam no arquivo com `sinal` vazio — é o que permite
    auditar o gate depois.
    """
    brutas = pd.read_csv(entrada, encoding='utf-8-sig')
    faltando = [c for c in COLUNAS_ENTRADA if c not in brutas.columns]
    if faltando:
        raise ValueError(f'CSV de entrada sem as colunas obrigatórias: {faltando}')
    brutas = brutas.reset_index(drop=True)

    relevantes, canais_filtro = filtrar(brutas['manchete'])
    if verbose:
        _relatar_filtro(brutas, relevantes, tamanho_lote)

    cache = _ler_cache(saida, modelo) if usar_cache else {}
    pendentes = [i for i in brutas.index
                 if relevantes.iloc[i] and _chave(brutas.loc[i]) not in cache]

    if verbose:
        print(f'\n📰 {len(cache)} reaproveitadas do cache | {len(pendentes)} a classificar')

    resultados = {}
    if pendentes:
        cliente = _criar_cliente()
        for inicio in range(0, len(pendentes), tamanho_lote):
            bloco = pendentes[inicio:inicio + tamanho_lote]
            obtidos = _classificar_lote(cliente, brutas, bloco, modelo)

            # Ids que não voltaram no lote: reprocessa um a um antes de desistir.
            for i in [i for i in bloco if int(i) not in obtidos]:
                obtidos.update(_classificar_lote(cliente, brutas, [i], modelo))

            resultados.update(obtidos)
            if verbose:
                print(f'   lote {inicio // tamanho_lote + 1}: '
                      f'{len(obtidos)}/{len(bloco)} classificadas')

    linhas = [_montar_linha(brutas.loc[i], cache, resultados, modelo,
                            bool(relevantes.iloc[i]), canais_filtro.iloc[i])
              for i in brutas.index]
    pontuadas = pd.DataFrame(linhas, columns=COLUNAS + COLUNAS_AUDITORIA)
    pontuadas.to_csv(saida, index=False)

    if verbose:
        _resumir(pontuadas, saida)
    return pontuadas


def _montar_linha(bruta, cache, resultados, modelo, relevante, canais):
    em_cache = cache.get(_chave(bruta))
    if em_cache is not None:
        return em_cache

    if not relevante:
        # Cortada antes da LLM. Fica no arquivo com sinal vazio: nada some, e dá
        # para auditar depois exatamente o que o filtro descartou.
        sinal, motivo, resultado = None, 'filtro_local', {}
    else:
        resultado = resultados.get(int(bruta.name))
        if resultado is None:
            sinal, motivo, resultado = None, 'sem_resposta', {}
        else:
            sinal, motivo = _resultado_para_sinal(resultado, bruta['manchete'])

    return {
        'jornal': bruta['jornal'],
        'manchete': bruta['manchete'],
        'data': bruta['data'],
        'hora': bruta['hora'],
        'sinal': sinal,
        'canal': resultado.get('canal', ''),
        'direcao': resultado.get('direcao', ''),
        'magnitude_decimos': resultado.get('magnitude_decimos', ''),
        'trecho': resultado.get('trecho', ''),
        'justificativa': resultado.get('justificativa', ''),
        'motivo_descarte': motivo,
        'canais_filtro': canais,
        'modelo': modelo,
        'versao_rubrica': VERSAO_RUBRICA,
    }


def _relatar_filtro(brutas, relevantes, tamanho_lote):
    total, passam = len(brutas), int(relevantes.sum())
    print(f'🔎 Filtro local: {passam} de {total} manchetes seguem para a LLM '
          f'({passam / total:.1%})')

    por_canal = {canal: int(brutas['manchete'].str.contains(padrao, case=False,
                                                           na=False, regex=True).sum())
                 for canal, padrao in TERMOS_RELEVANTES.items()}
    resumo = ' | '.join(f'{c} {n}' for c, n in
                        sorted(por_canal.items(), key=lambda x: -x[1]) if n)
    print(f'   {resumo}')
    print(f'   {total - passam} descartadas localmente (motivo_descarte=\'filtro_local\')')
    print(f'   ~{-(-passam // tamanho_lote)} chamadas à LLM em lotes de {tamanho_lote}')


def _resultado_para_sinal(resultado, manchete):
    """Aplica as travas e devolve (sinal, motivo_descarte).

    A LLM nunca emite o número: ela escolhe direção e um inteiro de 1 a 10, e a
    conta é feita aqui. Isso garante a grade de 0,1 por construção.
    """
    if resultado.get('canal') == 'NENHUM':
        return None, 'canal_nenhum'
    if not resultado.get('passa_novidade'):
        return None, 'sem_novidade'
    if not resultado.get('passa_horizonte'):
        return None, 'horizonte_longo'
    if resultado.get('direcao') not in ('ALTISTA', 'BAIXISTA'):
        return None, 'direcao_indefinida'

    decimos = resultado.get('magnitude_decimos')
    if not isinstance(decimos, int) or not 1 <= decimos <= 10:
        return None, 'fora_da_faixa'

    # Âncora textual: a justificativa tem que estar escrita na própria manchete.
    if not _trecho_confere(resultado.get('trecho', ''), manchete):
        return None, 'trecho_nao_encontrado'

    return round(DIRECAO[resultado['direcao']] * decimos / 10, 1), ''


def _normalizar(texto):
    return re.sub(r'\s+', ' ', str(texto)).strip().casefold()


def _trecho_confere(trecho, manchete):
    normalizado = _normalizar(trecho)
    return bool(normalizado) and normalizado in _normalizar(manchete)


def _chave(linha):
    return (str(linha['manchete']).strip(), str(linha['data']).strip(),
            str(linha['hora']).strip())


def _ler_cache(saida, modelo):
    """Reaproveita classificações anteriores da mesma rubrica e do mesmo modelo."""
    if not os.path.exists(saida):
        return {}
    anterior = pd.read_csv(saida, encoding='utf-8-sig')
    if 'versao_rubrica' not in anterior.columns:
        return {}
    valido = anterior[(anterior['modelo'] == modelo)
                      & (anterior['versao_rubrica'].astype(str) == VERSAO_RUBRICA)]
    return {_chave(linha): linha.to_dict() for _, linha in valido.iterrows()}


def _criar_cliente():
    try:
        from google import genai
    except ImportError as e:
        raise ImportError('O classificador precisa do SDK do Gemini. '
                          'Instale com: uv add google-genai') from e

    chave = os.environ.get('GEMINI_API_KEY')
    if not chave:
        raise RuntimeError('Defina GEMINI_API_KEY no ambiente. '
                           "No Colab: os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')")
    return genai.Client(api_key=chave)


def _classificar_lote(cliente, brutas, ids, modelo):
    """Uma chamada à LLM. Devolve {id: resultado} só com os ids realmente enviados.

    O casamento é sempre por id, nunca por posição na lista: um lote devolvido
    fora de ordem corromperia a base inteira em silêncio.
    """
    lote = [{'id': int(i), 'manchete': str(brutas.at[i, 'manchete'])} for i in ids]
    enviados = {item['id'] for item in lote}

    texto = _gerar(cliente, json.dumps(lote, ensure_ascii=False), modelo)
    payload = json.loads(texto)

    obtidos = {}
    for item in payload.get('classificacoes', []):
        identificador = item.get('id')
        if identificador in enviados and identificador not in obtidos:
            obtidos[identificador] = item
    return obtidos


def _gerar(cliente, conteudo, modelo):
    from google.genai import types

    for tentativa in range(TENTATIVAS):
        try:
            resposta = cliente.models.generate_content(
                model=modelo,
                contents=conteudo,
                config=types.GenerateContentConfig(
                    system_instruction=RUBRICA + INSTRUCOES_SAIDA,
                    temperature=0,
                    response_mime_type='application/json',
                    response_json_schema=ESQUEMA_RESPOSTA,
                ),
            )
            return resposta.text
        except Exception:
            if tentativa == TENTATIVAS - 1:
                raise
            time.sleep(2 ** tentativa)


def _resumir(pontuadas, saida):
    pontuada = pontuadas['sinal'].notna()
    print(f'\n✅ {saida}: {pontuada.sum()} pontuadas, {(~pontuada).sum()} descartadas')

    motivos = pontuadas.loc[~pontuada, 'motivo_descarte'].value_counts()
    for motivo, quantidade in motivos.items():
        alerta = '  ⚠️ (a LLM justificou com algo fora do texto)' if motivo == 'trecho_nao_encontrado' else ''
        print(f'   {motivo}: {quantidade}{alerta}')

    fora_da_grade = pontuadas.loc[pontuada, 'sinal'].mul(10).mod(1).abs().gt(1e-9).sum()
    print(f'   sinais fora da grade de 0,1: {fora_da_grade}')


def auditar_filtro(entrada, amostra=300, modelo=MODELO_PADRAO, semente=42):
    """Mede quanto sinal o filtro local está jogando fora.

    Sorteia `amostra` manchetes DESCARTADAS pelo filtro, roda a LLM nelas e
    conta quantas teriam recebido sinal. Perto de zero = filtro calibrado.
    Rode de novo toda vez que mexer em TERMOS_RELEVANTES.
    """
    brutas = pd.read_csv(entrada, encoding='utf-8-sig')
    relevantes, _ = filtrar(brutas['manchete'])

    descartadas = brutas[~relevantes]
    if descartadas.empty:
        print('O filtro não descartou nada — nada a auditar.')
        return

    n = min(amostra, len(descartadas))
    sorteadas = descartadas.sample(n, random_state=semente).reset_index(drop=True)
    print(f'🔍 Auditando {n} de {len(descartadas)} manchetes descartadas pelo filtro...')

    cliente = _criar_cliente()
    resultados = {}
    for inicio in range(0, n, TAMANHO_LOTE):
        bloco = list(range(inicio, min(inicio + TAMANHO_LOTE, n)))
        resultados.update(_classificar_lote(cliente, sorteadas, bloco, modelo))

    perdidas = []
    for i in range(n):
        resultado = resultados.get(i)
        if resultado is None:
            continue
        sinal, _ = _resultado_para_sinal(resultado, sorteadas.at[i, 'manchete'])
        if sinal is not None:
            perdidas.append((sinal, resultado['canal'], sorteadas.at[i, 'manchete']))

    print(f'\n   {len(perdidas)} de {n} ({len(perdidas) / n:.1%}) teriam recebido sinal.')
    if perdidas:
        print('   Manchetes que o filtro está perdendo — se houver padrão, '
              'falta termo em TERMOS_RELEVANTES:')
        for sinal, canal, manchete in sorted(perdidas, key=lambda x: -abs(x[0])):
            print(f'      [{sinal:+.1f} {canal}] {manchete[:88]}')
    else:
        print('   Filtro calibrado: nada de valor na amostra descartada.')


def comparar_com_gabarito(gerado, gabarito):
    """Mede a classificação contra um CSV rotulado à mão. É como se calibra a rubrica.

    Casa as duas bases pela manchete e reporta três coisas: o acerto do gate
    (descartar ou não), o acerto da direção entre as que ambos pontuaram, e o
    erro de magnitude. O gate é o número que mais importa — errar nele coloca
    ruído na base ou joga fora informação boa.
    """
    a = pd.read_csv(gerado, encoding='utf-8-sig')[['manchete', 'sinal', 'motivo_descarte']]
    b = pd.read_csv(gabarito, encoding='utf-8-sig')[['manchete', 'sinal']].rename(columns={'sinal': 'sinal_gabarito'})
    par = a.merge(b, on='manchete', how='inner')
    print(f'🔬 {len(par)} manchetes casadas com o gabarito')

    tem = par['sinal'].notna()
    tem_gab = par['sinal_gabarito'].notna()
    print(f'\n   Gate (descartar ou não): {(tem == tem_gab).mean():.1%} de acerto')
    print(f'      falsos positivos (pontuou o que era nulo):  {(tem & ~tem_gab).sum()}')
    print(f'      falsos negativos (descartou o que valia):   {(~tem & tem_gab).sum()}')

    ambos = par[tem & tem_gab]
    if len(ambos):
        mesma = np.sign(ambos['sinal']) == np.sign(ambos['sinal_gabarito'])
        erro = (ambos['sinal'].abs() - ambos['sinal_gabarito'].abs()).abs()
        print(f'\n   Direção: {mesma.mean():.1%} de acerto ({len(ambos)} comparáveis)')
        print(f'   Magnitude: erro médio {erro.mean():.2f}, máximo {erro.max():.2f}')

    print('\n   Divergências:')
    for _, linha in par[(tem != tem_gab) | (tem & tem_gab & (np.sign(par['sinal'].fillna(0))
                                                            != np.sign(par['sinal_gabarito'].fillna(0))))].iterrows():
        gerado_txt = f"{linha['sinal']:+.1f}" if pd.notna(linha['sinal']) else f"nulo ({linha['motivo_descarte']})"
        gab_txt = f"{linha['sinal_gabarito']:+.1f}" if pd.notna(linha['sinal_gabarito']) else 'nulo'
        print(f'      gerado {gerado_txt:>28}  | gabarito {gab_txt:>5}  | {linha["manchete"][:60]}')


def carregar_csv(caminho, verbose=True):
    """Lê o CSV de notícias pontuadas e devolve só as linhas utilizáveis.

    Linhas com `sinal` vazio são notícias que passaram pelo classificador e foram
    consideradas irrelevantes — ficam no arquivo para auditoria, mas saem daqui.
    """
    noticias = pd.read_csv(caminho, encoding='utf-8-sig')

    faltando = [c for c in COLUNAS if c not in noticias.columns]
    if faltando:
        raise ValueError(f'CSV sem as colunas obrigatórias: {faltando}')

    noticias = noticias[COLUNAS].copy()
    noticias['datahora'] = _parse_datahora(_combinar_data_hora(noticias))
    noticias['sinal'] = pd.to_numeric(noticias['sinal'], errors='coerce')

    total = len(noticias)
    noticias = noticias.dropna(subset=['datahora', 'sinal'])
    # O sinal é discreto (múltiplos de 0,1). O classificador já produz assim; o
    # round aqui só impede que um CSV editado à mão introduza 0.47 pela porta dos fundos.
    noticias['sinal'] = noticias['sinal'].clip(-1.0, 1.0).round(1)
    noticias = noticias.sort_values('datahora').reset_index(drop=True)

    if verbose:
        print(f'📰 Notícias lidas: {total} | pontuadas: {len(noticias)} | '
              f'descartadas (sinal nulo): {total - len(noticias)}')
    return noticias


def _combinar_data_hora(quadro):
    """Junta as colunas `data` e `hora` do CSV numa string só, para o parse."""
    data = quadro['data'].astype('string').str.strip()
    hora = quadro['hora'].astype('string').str.strip().fillna('00:00')
    return data + ' ' + hora


def _parse_datahora(coluna):
    # A base pode vir toda com fuso explícito (ideal) ou toda sem fuso (aí
    # assumimos o horário de NY). Misturar os dois é ambíguo demais para adivinhar.
    bruto = coluna.astype('string').str.strip()
    tem_fuso = bruto.str.contains(r'(?:Z|[+-]\d{2}:?\d{2})$', na=False)

    # Data com barra é ambígua entre dd/mm e mm/dd. Assumimos dd/mm (convenção
    # brasileira) e avisamos, porque errar aqui embaralha a base em silêncio.
    dia_primeiro = bool(bruto.str.contains('/', na=False).any())
    if dia_primeiro:
        print('⚠️ Datas com barra detectadas — interpretando como dd/mm/aaaa. '
              'Se a base for mm/dd/aaaa, converta para ISO (aaaa-mm-dd) antes.')

    if tem_fuso.all():
        return pd.to_datetime(bruto, utc=True, dayfirst=dia_primeiro).dt.tz_convert(FUSO_PREGAO)
    if not tem_fuso.any():
        return pd.to_datetime(bruto, dayfirst=dia_primeiro).dt.tz_localize(
            FUSO_PREGAO, ambiguous=True, nonexistent='shift_forward')
    raise ValueError('A coluna hora mistura valores com e sem fuso horário. '
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


def _autoteste(caminho='noticias_pontuadas.csv'):
    # Exercita o consumidor contra um calendário sintético de dias úteis.
    noticias = carregar_csv(caminho)
    inicio = noticias['datahora'].min().tz_localize(None).normalize()
    fim = noticias['datahora'].max().tz_localize(None).normalize() + pd.Timedelta(days=30)
    calendario = pd.bdate_range(inicio, fim)
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


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == 'classificar':
        classificar_csv(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == 'comparar':
        comparar_com_gabarito(sys.argv[2], sys.argv[3])
    elif len(sys.argv) in (3, 4) and sys.argv[1] == 'auditar':
        auditar_filtro(sys.argv[2], int(sys.argv[3]) if len(sys.argv) == 4 else 300)
    elif len(sys.argv) == 2 and sys.argv[1].endswith('.csv'):
        _autoteste(sys.argv[1])
    elif len(sys.argv) > 1:
        print('uso: python news_signal.py classificar <brutas.csv> <pontuadas.csv>')
        print('     python news_signal.py auditar <brutas.csv> [n]  (recall do filtro)')
        print('     python news_signal.py comparar <pontuadas.csv> <gabarito.csv>')
        print('     python news_signal.py [pontuadas.csv]   (autoteste do consumidor)')
        sys.exit(1)
    else:
        _autoteste()
