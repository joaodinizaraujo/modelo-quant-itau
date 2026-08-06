"""Backend NAO-OFICIAL: le o timeline de um perfil pelo endpoint GraphQL do site.

Existe porque a API oficial nao expoe o timeline de um perfil arbitrario (ver
threads.py). Em troca:

  - e fragil: o `doc_id` e o formato da resposta mudam sem aviso;
  - nao e suportado pela Meta e provavelmente contraria os Termos de Uso;
  - pode levar a bloqueio por IP se usado com volume.

Use apenas para dados publicos, com moderacao, e por sua conta e risco. O caminho
oficial (`ThreadsClient.profile_posts` sem `unofficial=True`) e sempre preferivel.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import httpx

from modelo_quant.threads import Media, ThreadsError

log = logging.getLogger(__name__)

WEB_URL = "https://www.threads.com"
GRAPHQL_URL = f"{WEB_URL}/api/graphql"

#: App id publico do cliente web da Threads (fixo).
IG_APP_ID = "238260118697367"

#: Query do tab de posts de um perfil. A Meta rotaciona esse id -- sobrescreva com
#: THREADS_WEB_DOC_ID quando o padrao parar de funcionar.
DEFAULT_DOC_ID = "25073520749683446"

#: Intervalo minimo entre requests, para nao apanhar de rate limit.
MIN_INTERVAL = 1.5

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

_LSD_RE = re.compile(r'"LSD",\[\],\{"token":"([^"]+)"')
_USER_ID_RE = re.compile(r'"user_id":"(\d+)"')
_PROFILE_ID_RE = re.compile(r'"props":\{"user_id":"(\d+)"')


def profile_posts(username: str, *, max_items: int | None = None) -> Iterator[Media]:
    """Timeline publico de um perfil, do mais recente para o mais antigo.

    >>> list(profile_posts("zuck", max_items=5))   # doctest: +SKIP
    """
    doc_id = os.getenv("THREADS_WEB_DOC_ID") or DEFAULT_DOC_ID

    with httpx.Client(
        timeout=20.0,
        follow_redirects=True,
        headers={"User-Agent": _BROWSER_UA, "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8"},
    ) as client:
        lsd, user_id = _abrir_perfil(client, username)

        emitidos = 0
        cursor: str | None = None
        ultimo_request = 0.0

        while True:
            espera = MIN_INTERVAL - (time.monotonic() - ultimo_request)
            if espera > 0:
                time.sleep(espera)
            ultimo_request = time.monotonic()

            payload = _graphql(client, lsd, doc_id, user_id, cursor)

            posts = list(_extrair_posts(payload))
            if not posts:
                return
            for post in posts:
                yield post
                emitidos += 1
                if max_items is not None and emitidos >= max_items:
                    return

            proximo = _extrair_cursor(payload)
            if not proximo or proximo == cursor:
                return
            cursor = proximo


def _abrir_perfil(client: httpx.Client, username: str) -> tuple[str, str]:
    """Busca a pagina do perfil e extrai o token `lsd` e o id numerico do usuario."""
    resposta = client.get(f"{WEB_URL}/@{username}")
    if resposta.status_code == 404:
        raise ThreadsError(f"perfil @{username} nao encontrado")
    if not resposta.is_success:
        raise ThreadsError(f"nao consegui abrir @{username} (HTTP {resposta.status_code})")

    html = resposta.text

    lsd = _LSD_RE.search(html)
    if not lsd:
        raise _endpoint_mudou("nao achei o token 'lsd' no HTML do perfil")

    user_id = _PROFILE_ID_RE.search(html) or _USER_ID_RE.search(html)
    if not user_id:
        raise _endpoint_mudou(f"nao achei o id numerico de @{username} no HTML")

    return lsd.group(1), user_id.group(1)


def _graphql(
    client: httpx.Client, lsd: str, doc_id: str, user_id: str, cursor: str | None
) -> dict[str, Any]:
    variables: dict[str, Any] = {
        "userID": user_id,
        "first": 25,
        "__relay_internal__pv__BarcelonaIsLoggedInrelayprovider": False,
    }
    if cursor:
        variables["after"] = cursor

    resposta = client.post(
        GRAPHQL_URL,
        data={"lsd": lsd, "doc_id": doc_id, "variables": json.dumps(variables)},
        headers={
            "X-IG-App-ID": IG_APP_ID,
            "X-FB-LSD": lsd,
            "Sec-Fetch-Site": "same-origin",
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": WEB_URL,
            "Referer": f"{WEB_URL}/",
        },
    )
    if not resposta.is_success:
        raise ThreadsError(f"GraphQL respondeu HTTP {resposta.status_code}: {resposta.text[:300]}")

    try:
        payload = resposta.json()
    except ValueError as e:
        raise _endpoint_mudou("resposta do GraphQL nao e JSON") from e

    if payload.get("errors"):
        raise ThreadsError(f"GraphQL retornou erros: {payload['errors']}")
    if erro := payload.get("error"):
        raise ThreadsError(f"GraphQL retornou erro: {erro}")
    return payload


def _extrair_posts(payload: dict[str, Any]) -> Iterator[Media]:
    """Varre o payload procurando nos de post e normaliza cada um para Media.

    Vasculhar em vez de indexar um caminho fixo e de proposito: a Meta renomeia as
    chaves de conexao (`xdt_api__v1__feed__user_timeline_graphql_connection`, ...)
    com frequencia, mas a forma do post em si e estavel.
    """
    vistos: set[str] = set()
    for no in _percorrer(payload):
        if not _parece_post(no):
            continue
        media = _para_media(no)
        if media.id in vistos:
            continue
        vistos.add(media.id)
        yield media


def _extrair_cursor(payload: dict[str, Any]) -> str | None:
    for no in _percorrer(payload):
        if "end_cursor" in no and "has_next_page" in no:
            if no.get("has_next_page"):
                return no.get("end_cursor")
            return None
    return None


def _percorrer(valor: Any) -> Iterator[dict[str, Any]]:
    """Itera todos os dicts aninhados do payload."""
    if isinstance(valor, dict):
        yield valor
        for filho in valor.values():
            yield from _percorrer(filho)
    elif isinstance(valor, list):
        for filho in valor:
            yield from _percorrer(filho)


def _parece_post(no: dict[str, Any]) -> bool:
    return ("pk" in no or "id" in no) and ("caption" in no or "text_post_app_info" in no)


def _para_media(no: dict[str, Any]) -> Media:
    caption = no.get("caption") or {}
    usuario = no.get("user") or {}
    codigo = no.get("code")
    taken_at = no.get("taken_at")

    return Media(
        id=str(no.get("pk") or no.get("id")),
        text=caption.get("text") if isinstance(caption, dict) else None,
        username=usuario.get("username"),
        permalink=f"{WEB_URL}/@{usuario.get('username')}/post/{codigo}" if codigo else None,
        timestamp=(
            datetime.fromtimestamp(taken_at, tz=UTC) if isinstance(taken_at, (int, float)) else None
        ),
        media_type=_tipo_de_midia(no),
        is_reply=bool((no.get("text_post_app_info") or {}).get("reply_to_author")),
        is_quote_post=bool(
            (no.get("text_post_app_info") or {}).get("share_info", {}).get("quoted_post")
        ),
        # extra="allow" no Media: guardamos as contagens que so o web expoe.
        like_count=no.get("like_count"),
        reply_count=(no.get("text_post_app_info") or {}).get("direct_reply_count"),
    )


def _tipo_de_midia(no: dict[str, Any]) -> str:
    if no.get("video_versions"):
        return "VIDEO"
    if no.get("carousel_media"):
        return "CAROUSEL_ALBUM"
    if no.get("image_versions2"):
        return "IMAGE"
    return "TEXT"


def _endpoint_mudou(detalhe: str) -> ThreadsError:
    return ThreadsError(
        f"{detalhe}. O endpoint web da Threads provavelmente mudou -- tente atualizar "
        "THREADS_WEB_DOC_ID, ou use o backend oficial (profile_posts sem unofficial=True)."
    )
