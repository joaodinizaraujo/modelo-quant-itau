"""Analise de sentimento sobre pysentimiento.

pysentimiento arrasta torch (~2,5 GB), entao ele nao esta nas dependencias base:

    uv sync --extra sentiment

Nenhum import de pysentimiento/torch acontece no topo deste modulo -- so dentro de
`_load()`. Assim `import modelo_quant` continua barato, e este arquivo pode ser
importado sem o extra instalado (o erro so aparece na primeira analise).

    from modelo_quant.sentiment import analyze
    print(analyze("adorei esse produto"))   # Sentiment(label='POS', score=0.98, ...)
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelo_quant.threads import Media

DEFAULT_LANG = "pt"
DEFAULT_TASK = "sentiment"
DEFAULT_BATCH_SIZE = 32


@dataclass(frozen=True)
class Sentiment:
    """Resultado de uma analise. `label` depende da task.

    Na task "sentiment": "POS", "NEG" ou "NEU".

    As tasks multi-label ("emotion", "hate_speech") podem marcar varios rotulos de
    uma vez -- eles ficam em `labels`, e `label` guarda o de maior probabilidade
    ("NONE" quando nenhum foi marcado).
    """

    label: str
    score: float
    probabilities: dict[str, float] = field(default_factory=dict)
    task: str = DEFAULT_TASK
    lang: str = DEFAULT_LANG
    labels: tuple[str, ...] = ()

    @property
    def polarity(self) -> int:
        """+1 para positivo, -1 para negativo, 0 para o resto. Util para media."""
        return {"POS": 1, "NEG": -1}.get(self.label, 0)


@functools.lru_cache(maxsize=4)
def _load(task: str, lang: str) -> Any:
    """Cria (e cacheia) o analyzer do pysentimiento.

    A primeira chamada baixa o modelo do HuggingFace -- centenas de MB, demora.
    """
    try:
        from pysentimiento import create_analyzer
    except ImportError as e:  # pragma: no cover - testado com __import__ mockado
        raise ImportError(
            "pysentimiento nao esta instalado. Rode: uv sync --extra sentiment"
        ) from e
    return create_analyzer(task=task, lang=lang)


def analyze(text: str, *, task: str = DEFAULT_TASK, lang: str = DEFAULT_LANG) -> Sentiment | None:
    """Sentimento de um texto. Retorna None se o texto for vazio (nao chama o modelo).

    task: "sentiment" (default), "emotion", "hate_speech" ou "irony".
    lang: "pt" (default), "es" ou "en".
    """
    if not text or not text.strip():
        return None
    saida = _load(task, lang).predict(text)
    return _to_sentiment(saida, task, lang)


def analyze_many(
    texts: Sequence[str],
    *,
    task: str = DEFAULT_TASK,
    lang: str = DEFAULT_LANG,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[Sentiment | None]:
    """Analisa varios textos em lote. A saida e alinhada por indice com a entrada.

    Textos vazios saem como None e nao chegam ao modelo.
    """
    resultados: list[Sentiment | None] = [None] * len(texts)

    indices = [i for i, t in enumerate(texts) if t and t.strip()]
    if not indices:
        return resultados

    analyzer = _load(task, lang)
    for lote in itertools.batched(indices, batch_size):
        saidas = analyzer.predict([texts[i] for i in lote])
        for i, saida in zip(lote, saidas, strict=True):
            resultados[i] = _to_sentiment(saida, task, lang)
    return resultados


def analyze_posts(
    posts: Iterable[Media],
    *,
    task: str = DEFAULT_TASK,
    lang: str = DEFAULT_LANG,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[tuple[Media, Sentiment | None]]:
    """Anota posts do ThreadsClient com sentimento, em lotes.

    Consome `posts` de forma lazy -- pode receber o generator do client direto,
    sem materializar tudo na memoria.

    >>> for post, s in analyze_posts(client.search("carne", max_items=50)):  # doctest: +SKIP
    ...     print(s.label if s else "-", post.text)
    """
    for lote in itertools.batched(posts, batch_size):
        sentimentos = analyze_many(
            [p.text or "" for p in lote], task=task, lang=lang, batch_size=batch_size
        )
        yield from zip(lote, sentimentos, strict=True)


def summarize(results: Iterable[Sentiment | None]) -> dict[str, Any]:
    """Resumo agregado: contagem e share por label, polaridade media.

    Retorna {"total", "analisados", "ignorados", "contagem", "share", "polaridade_media"}.
    """
    contagem: dict[str, int] = {}
    analisados: list[Sentiment] = []
    ignorados = 0

    for r in results:
        if r is None:
            ignorados += 1
            continue
        analisados.append(r)
        contagem[r.label] = contagem.get(r.label, 0) + 1

    n = len(analisados)
    return {
        "total": n + ignorados,
        "analisados": n,
        "ignorados": ignorados,
        "contagem": contagem,
        "share": {label: qtd / n for label, qtd in contagem.items()} if n else {},
        "polaridade_media": sum(r.polarity for r in analisados) / n if n else 0.0,
    }


def _to_sentiment(saida: Any, task: str, lang: str) -> Sentiment:
    probas = {str(k): float(v) for k, v in (getattr(saida, "probas", None) or {}).items()}
    bruto = getattr(saida, "output", None)

    if isinstance(bruto, (list, tuple, set)):
        # Tasks multi-label ("emotion", "hate_speech") devolvem uma lista de rotulos,
        # possivelmente vazia quando nada foi detectado.
        labels = tuple(str(x) for x in bruto)
    elif bruto is None:
        labels = (max(probas, key=lambda k: probas[k]),) if probas else ()
    else:
        labels = (str(bruto),)

    label = max(labels, key=lambda k: probas.get(k, 0.0)) if labels else "NONE"
    return Sentiment(
        label=label,
        score=probas.get(label, 0.0),
        probabilities=probas,
        task=task,
        lang=lang,
        labels=labels,
    )
