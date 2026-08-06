"""Client da Threads Graph API (https://graph.threads.net).

Limitacao importante da API oficial: nao existe endpoint que liste o timeline de um
perfil arbitrario. Os posts de outro perfil so sao alcancaveis via `/keyword_search`
com `author_username`, que exige um termo de busca `q`. Por isso `profile_posts()`
pede um `query` -- e oferece `unofficial=True` para o backend web (ver threads_web.py).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime
from typing import Any, Self, TypeVar

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

load_dotenv()

log = logging.getLogger(__name__)

BASE_URL = "https://graph.threads.net/v1.0"

#: A API rejeita `since`/`until` anteriores ao lancamento da rede (2023-07-05).
MIN_SEARCH_TIMESTAMP = 1688540400
MAX_LIMIT = 100

MEDIA_FIELDS = (
    "id",
    "text",
    "username",
    "permalink",
    "timestamp",
    "media_type",
    "media_url",
    "thumbnail_url",
    "shortcode",
    "has_replies",
    "is_reply",
    "is_quote_post",
)
REPLY_FIELDS = MEDIA_FIELDS + ("root_post", "replied_to", "hide_status", "reply_audience")
PROFILE_FIELDS = (
    "id",
    "username",
    "name",
    "profile_picture_url",
    "biography",
    "follower_count",
    "likes_count",
    "quotes_count",
    "reposts_count",
    "views_count",
    "is_verified",
)

#: Permissoes exigidas por endpoint. Usado para transformar o
#: "Unsupported get request" cru da Meta em algo diagnosticavel.
PERMISSIONS = {
    "keyword_search": ("threads_basic", "threads_keyword_search"),
    "profile_lookup": ("threads_basic", "threads_profile_discovery"),
    "replies": ("threads_basic", "threads_manage_replies"),
    "conversation": ("threads_basic", "threads_manage_replies"),
    "threads": ("threads_basic",),
    "media": ("threads_basic",),
}

_RETRY_STATUS = (429, 500, 502, 503, 504)

_M = TypeVar("_M", bound="Media")


# --------------------------------------------------------------------------- erros


class ThreadsError(Exception):
    """Qualquer falha desta biblioteca."""


class ThreadsAPIError(ThreadsError):
    """Erro devolvido pela Threads API, com o payload de erro da Meta."""

    def __init__(
        self,
        message: str,
        *,
        status: int,
        code: int | None = None,
        subcode: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.subcode = subcode

    def __str__(self) -> str:
        detalhe = f"HTTP {self.status}"
        if self.code is not None:
            detalhe += f", code {self.code}"
        if self.subcode is not None:
            detalhe += f", subcode {self.subcode}"
        return f"{self.message} ({detalhe})"


class ThreadsPermissionError(ThreadsAPIError):
    """O app nao tem a permissao exigida pelo endpoint (ou ela nao foi aprovada)."""

    def __init__(self, message: str, *, endpoint: str | None = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.endpoint = endpoint
        self.required = PERMISSIONS.get(endpoint or "", ())

    def __str__(self) -> str:
        base = super().__str__()
        if not self.required:
            return base
        return (
            f"{base}. O endpoint '{self.endpoint}' exige as permissoes: "
            f"{', '.join(self.required)} -- confira as aprovacoes do app em "
            "https://developers.facebook.com/docs/threads/overview"
        )


def _raise_for_error(response: httpx.Response, endpoint: str) -> None:
    if response.is_success:
        return

    try:
        erro = response.json().get("error", {})
    except ValueError:
        erro = {}

    message = erro.get("message") or response.text[:500] or "erro sem corpo"
    code = erro.get("code")
    subcode = erro.get("error_subcode")

    e_permissao = (
        response.status_code == 403 or code == 10 or (code is not None and 200 <= code <= 299)
    )
    if e_permissao:
        raise ThreadsPermissionError(
            message, endpoint=endpoint, status=response.status_code, code=code, subcode=subcode
        )
    raise ThreadsAPIError(message, status=response.status_code, code=code, subcode=subcode)


# --------------------------------------------------------------------------- models


class _Base(BaseModel):
    # extra="allow" para sobreviver a campos novos da Meta e a `fields=` customizado.
    model_config = ConfigDict(extra="allow")


class Media(_Base):
    """Um post do Threads."""

    id: str
    text: str | None = None
    username: str | None = None
    permalink: str | None = None
    timestamp: datetime | None = None
    media_type: str | None = None
    has_replies: bool | None = None
    is_reply: bool | None = None
    is_quote_post: bool | None = None


class Reply(Media):
    """Um comentario. `root_post`/`replied_to` chegam como {"id": ...}."""

    root_post: dict[str, Any] | None = None
    replied_to: dict[str, Any] | None = None
    hide_status: str | None = None


class Profile(_Base):
    """Metadados publicos de um perfil (endpoint /profile_lookup)."""

    username: str
    id: str | None = None
    name: str | None = None
    biography: str | None = None
    profile_picture_url: str | None = None
    follower_count: int | None = None
    likes_count: int | None = None
    quotes_count: int | None = None
    reposts_count: int | None = None
    views_count: int | None = None
    is_verified: bool | None = None


# --------------------------------------------------------------------------- helpers


def _to_unix(valor: datetime | date | int | None) -> int | None:
    """Normaliza datetime/date/int para timestamp unix, validando o piso da API."""
    if valor is None:
        return None

    if isinstance(valor, bool):  # bool e subclasse de int; quase certamente um bug
        raise ThreadsError(f"timestamp invalido: {valor!r}")
    if isinstance(valor, int):
        ts = valor
    elif isinstance(valor, datetime):
        if valor.tzinfo is None:
            valor = valor.replace(tzinfo=UTC)
        ts = int(valor.timestamp())
    elif isinstance(valor, date):
        ts = int(datetime(valor.year, valor.month, valor.day, tzinfo=UTC).timestamp())
    else:
        raise ThreadsError(f"timestamp invalido: {valor!r}")

    if ts < MIN_SEARCH_TIMESTAMP:
        raise ThreadsError(
            f"timestamp {ts} e anterior ao minimo aceito pela API ({MIN_SEARCH_TIMESTAMP}, 2023-07-05)"
        )
    return ts


def _join_fields(fields: Sequence[str] | str | None, default: Sequence[str]) -> str:
    if fields is None:
        return ",".join(default)
    if isinstance(fields, str):
        return fields
    return ",".join(fields)


def _clamp_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    return max(1, min(limit, MAX_LIMIT))


# --------------------------------------------------------------------------- client


class ThreadsClient:
    """Client sincrono da Threads Graph API.

    O token vem do argumento ou da variavel de ambiente THREADS_ACCESS_TOKEN (.env).

    >>> with ThreadsClient() as c:
    ...     for post in c.search("carne bovina", max_items=10):
    ...         print(post.username, post.text)
    """

    def __init__(
        self,
        access_token: str | None = None,
        *,
        timeout: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        token = access_token or os.getenv("THREADS_ACCESS_TOKEN")
        if not token:
            raise ThreadsError(
                "token ausente: passe access_token= ou defina THREADS_ACCESS_TOKEN no .env"
            )
        self.access_token = token

        self._client = client or httpx.Client(
            base_url=BASE_URL,
            timeout=timeout,
            transport=httpx.HTTPTransport(retries=2),  # cobre falhas de conexao
            headers={"Accept": "application/json", "User-Agent": "modelo-quant/0.1.0"},
        )
        self._client_proprio = client is None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._client_proprio:
            self._client.close()

    # ------------------------------------------------------------------ interno

    def _get(self, path: str, params: dict[str, Any], endpoint: str) -> dict[str, Any]:
        params = {k: v for k, v in params.items() if v is not None}
        params["access_token"] = self.access_token

        for tentativa in range(3):
            resposta = self._client.get(path, params=params)
            if resposta.status_code not in _RETRY_STATUS or tentativa == 2:
                break
            espera = _retry_after(resposta) or 0.5 * 2**tentativa
            log.warning(
                "HTTP %s em %s, tentando de novo em %.1fs", resposta.status_code, path, espera
            )
            time.sleep(espera)

        _raise_for_error(resposta, endpoint)
        return resposta.json()

    def _paginate(
        self,
        path: str,
        params: dict[str, Any],
        model: type[_M],
        endpoint: str,
        max_items: int | None,
    ) -> Iterator[_M]:
        """Percorre as paginas avancando pelo cursor `after`.

        Nao seguimos `paging.next` cru porque aquela URL embute o access token e
        vazaria em logs; alem disso ela ignoraria nosso controle de fields/limit.
        """
        emitidos = 0
        cursor: str | None = None

        while True:
            if cursor:
                params = {**params, "after": cursor}
            payload = self._get(path, params, endpoint)

            data = payload.get("data") or []
            for item in data:
                yield model.model_validate(item)
                emitidos += 1
                if max_items is not None and emitidos >= max_items:
                    return

            proximo = ((payload.get("paging") or {}).get("cursors") or {}).get("after")
            # A Meta as vezes ecoa o mesmo cursor indefinidamente -- para nesse caso.
            if not data or not proximo or proximo == cursor:
                return
            cursor = proximo

    # -------------------------------------------------------------- busca global

    def search(
        self,
        q: str,
        *,
        search_mode: str = "KEYWORD",
        search_type: str = "TOP",
        media_type: str | None = None,
        since: datetime | date | int | None = None,
        until: datetime | date | int | None = None,
        author_username: str | None = None,
        fields: Sequence[str] | str | None = None,
        limit: int | None = None,
        max_items: int | None = None,
    ) -> Iterator[Media]:
        """Busca posts publicos em toda a rede.

        search_mode: "KEYWORD" (default) ou "TAG" (busca por topico).
        search_type: "TOP" (default) ou "RECENT".
        media_type:  "TEXT", "IMAGE" ou "VIDEO"; None traz todos.

        Exige a permissao `threads_keyword_search`. Sem ela a API NAO da erro --
        ela restringe silenciosamente a busca aos posts do dono do token.
        """
        params = {
            "q": q,
            "search_mode": search_mode,
            "search_type": search_type,
            "media_type": media_type,
            "author_username": author_username,
            "since": _to_unix(since),
            "until": _to_unix(until),
            "fields": _join_fields(fields, MEDIA_FIELDS),
            "limit": _clamp_limit(limit),
        }
        return self._paginate("/keyword_search", params, Media, "keyword_search", max_items)

    # ------------------------------------------------------------ posts de perfil

    def profile_posts(
        self,
        username: str,
        *,
        query: str | Sequence[str] | None = None,
        unofficial: bool = False,
        search_type: str = "RECENT",
        since: datetime | date | int | None = None,
        until: datetime | date | int | None = None,
        fields: Sequence[str] | str | None = None,
        limit: int | None = None,
        max_items: int | None = None,
    ) -> Iterator[Media]:
        """Posts de um perfil.

        Caminho oficial (default): `/keyword_search` com author_username, que exige
        `query` -- uma palavra ou uma lista de palavras (buscas sequenciais, mescladas
        e deduplicadas por id). O resultado e um SUBCONJUNTO filtrado por palavra, nao
        o timeline: a API oficial nao expoe o timeline de um perfil arbitrario.

        Com unofficial=True usa o endpoint GraphQL web (threads_web.py), que devolve o
        timeline completo e nao exige aprovacao de permissao -- mas e fragil e nao
        suportado pela Meta. Nesse modo `query` e ignorado.
        """
        # Validacao eager (nao dentro do generator) para o erro aparecer na chamada.
        username = username.lstrip("@").strip()
        if not username:
            raise ThreadsError("username vazio; para os proprios posts use my_posts()")

        if unofficial:
            from modelo_quant import threads_web  # import local: modulo opcional/fragil

            return threads_web.profile_posts(username, max_items=max_items)

        if not query:
            raise ThreadsError(
                "o backend oficial exige query=: /keyword_search precisa de um termo 'q'. "
                "Para o timeline completo do perfil use unofficial=True; "
                "para os posts do proprio token use my_posts()."
            )

        termos = [query] if isinstance(query, str) else list(query)
        return self._buscar_por_autor(
            username, termos, search_type, since, until, fields, limit, max_items
        )

    def _buscar_por_autor(
        self,
        username: str,
        termos: list[str],
        search_type: str,
        since: datetime | date | int | None,
        until: datetime | date | int | None,
        fields: Sequence[str] | str | None,
        limit: int | None,
        max_items: int | None,
    ) -> Iterator[Media]:
        """Roda uma busca por termo, mesclando e deduplicando por id."""
        vistos: set[str] = set()
        casou_username = False
        emitidos = 0

        for termo in termos:
            restante = None if max_items is None else max_items - emitidos
            if restante is not None and restante <= 0:
                break
            for post in self.search(
                termo,
                search_type=search_type,
                author_username=username,
                since=since,
                until=until,
                fields=fields,
                limit=limit,
                max_items=restante,
            ):
                if post.id in vistos:
                    continue
                vistos.add(post.id)
                if (post.username or "").lower() == username.lower():
                    casou_username = True
                yield post
                emitidos += 1

        if vistos and not casou_username:
            # Modo de falha silencioso: sem threads_keyword_search aprovado, a API
            # devolve os posts do dono do token em vez de erro.
            log.warning(
                "nenhum post retornado pertence a @%s -- o app provavelmente nao tem "
                "a permissao threads_keyword_search aprovada, e a busca foi restrita "
                "aos posts do dono do token",
                username,
            )

    def my_posts(
        self,
        *,
        since: datetime | date | int | None = None,
        until: datetime | date | int | None = None,
        fields: Sequence[str] | str | None = None,
        limit: int | None = None,
        max_items: int | None = None,
    ) -> Iterator[Media]:
        """Timeline completo do usuario autenticado. Unico timeline que a API oficial expoe."""
        params = {
            "since": _to_unix(since),
            "until": _to_unix(until),
            "fields": _join_fields(fields, MEDIA_FIELDS),
            "limit": _clamp_limit(limit),
        }
        return self._paginate("/me/threads", params, Media, "threads", max_items)

    # ------------------------------------------------------------ objetos unicos

    def get_media(self, media_id: str, *, fields: Sequence[str] | str | None = None) -> Media:
        """Um post por id."""
        payload = self._get(f"/{media_id}", {"fields": _join_fields(fields, MEDIA_FIELDS)}, "media")
        return Media.model_validate(payload)

    def get_profile(self, username: str, *, fields: Sequence[str] | str | None = None) -> Profile:
        """Metadados publicos de um perfil. Nao traz posts -- para isso use profile_posts()."""
        payload = self._get(
            "/profile_lookup",
            {"username": username.lstrip("@"), "fields": _join_fields(fields, PROFILE_FIELDS)},
            "profile_lookup",
        )
        return Profile.model_validate(payload)

    # ---------------------------------------------------------------- comentarios

    def replies(
        self,
        media_id: str,
        *,
        nested: bool = False,
        reverse: bool = True,
        fields: Sequence[str] | str | None = None,
        limit: int | None = None,
        max_items: int | None = None,
    ) -> Iterator[Reply]:
        """Comentarios de um post.

        nested=False -> /replies, so o primeiro nivel.
        nested=True  -> /conversation, a arvore completa da thread.
        reverse=True -> mais recentes primeiro.

        Exige a permissao `threads_manage_replies`.
        """
        endpoint = "conversation" if nested else "replies"
        params = {
            # A API rejeita o True do Python; precisa ser a string.
            "reverse": "true" if reverse else "false",
            "fields": _join_fields(fields, REPLY_FIELDS),
            "limit": _clamp_limit(limit),
        }
        return self._paginate(f"/{media_id}/{endpoint}", params, Reply, endpoint, max_items)


def _retry_after(resposta: httpx.Response) -> float | None:
    valor = resposta.headers.get("Retry-After")
    if not valor:
        return None
    try:
        return float(valor)
    except ValueError:
        return None  # formato HTTP-date: cai no backoff exponencial
