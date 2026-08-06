"""modelo-quant: client da Threads API e utilitario de analise de sentimento.

O modulo de sentimento NAO e importado aqui de proposito -- ele arrasta torch
(~2,5 GB). Importe-o explicitamente quando precisar:

    from modelo_quant.sentiment import analyze
"""

from modelo_quant.threads import (
    Media,
    Profile,
    Reply,
    ThreadsAPIError,
    ThreadsClient,
    ThreadsError,
    ThreadsPermissionError,
)

__all__ = [
    "Media",
    "Profile",
    "Reply",
    "ThreadsAPIError",
    "ThreadsClient",
    "ThreadsError",
    "ThreadsPermissionError",
]
