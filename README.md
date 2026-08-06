# modelo-quant

Client da **Threads API** + utilitário de **análise de sentimento** sobre `pysentimiento`.

## Instalação

```bash
uv sync                      # client da Threads (httpx + pydantic) — venv de ~40 MB
uv sync --extra sentiment    # + pysentimiento/torch/spacy — venv sobe para ~890 MB
```

O extra é separado de propósito: `import modelo_quant` não carrega torch. Só quem
importa `modelo_quant.sentiment` paga por isso.

Copie `.env.example` para `.env` e preencha `THREADS_ACCESS_TOKEN`.

```bash
uv run main.py     # demo end-to-end
```

## Uso

```python
from modelo_quant import ThreadsClient

with ThreadsClient() as c:                       # token do .env
    # busca em toda a rede
    for post in c.search("carne bovina", search_type="RECENT", max_items=20):
        print(post.username, post.text)

    # posts de um perfil (ver limitação abaixo)
    posts = list(c.profile_posts("jbs_brasil", query=["carne", "exportacao"]))

    # comentários de um post
    for reply in c.replies(posts[0].id, max_items=50):
        print(reply.username, reply.text)

    # árvore completa da thread
    list(c.replies(posts[0].id, nested=True))

    # metadados do perfil (não traz posts)
    print(c.get_profile("jbs_brasil").follower_count)

    # timeline completo do próprio token
    list(c.my_posts())
```

Todos os métodos de listagem devolvem **generators** que paginam sozinhos. Use
`max_items=` para limitar — a paginação para de fazer requests assim que o limite é
atingido.

### Sentimento

```python
from modelo_quant.sentiment import analyze, analyze_many, analyze_posts, summarize

analyze("adorei esse produto")
# Sentiment(label='POS', score=0.98, probabilities={...}, task='sentiment', lang='pt')

analyze_many(["ótimo", "péssimo"])          # em lote, alinhado por índice
analyze("great product", lang="en")         # ou "es"

# tasks multi-label marcam vários rótulos de uma vez — vêm em .labels,
# e .label guarda o de maior probabilidade ("NONE" se nada foi marcado)
analyze("estou muito feliz com isso", task="emotion").labels   # ('joy',)
analyze("bom dia", task="hate_speech").label                   # 'NONE'

# compõe direto com o client, em lotes e de forma lazy
resultados = []
for post, s in analyze_posts(c.search("carne", max_items=100)):
    resultados.append(s)
print(summarize(resultados))
# {'total': 100, 'analisados': 97, 'ignorados': 3, 'contagem': {'NEU': 60, ...},
#  'share': {...}, 'polaridade_media': -0.12}
```

A primeira chamada de cada `(task, lang)` baixa o modelo do HuggingFace (~500 MB por
modelo, em `~/.cache/huggingface`) e o mantém em cache no processo. Textos vazios
retornam `None` e não chegam ao modelo.

## Limitações da Threads API (leia antes de debugar)

**Não existe endpoint oficial que liste o timeline de um perfil arbitrário.** O único
caminho oficial para posts de terceiros é `/keyword_search` com `author_username`, que
exige um termo `q`. Por isso `profile_posts()` pede `query=` e devolve um **subconjunto
filtrado por palavra**, não o timeline. O timeline completo só existe para o dono do
token, via `my_posts()`.

Como alternativa, `profile_posts(username, unofficial=True)` usa o endpoint GraphQL do
site (`threads_web.py`) e traz o timeline completo de qualquer perfil público, sem
precisar de aprovação de permissão. Em troca: é frágil (o `doc_id` rotaciona — ajuste
`THREADS_WEB_DOC_ID`), pode levar a bloqueio por IP, e não é suportado pela Meta.
Prefira o caminho oficial.

**Sem a permissão `threads_keyword_search` aprovada, a busca não dá erro** — ela
restringe silenciosamente os resultados aos posts do dono do token. O client emite um
`logging.warning` quando nenhum post retornado pertence ao perfil pedido; ative logs
para vê-lo:

```python
import logging; logging.basicConfig(level=logging.WARNING)
```

### Permissões por endpoint

| Método | Endpoint | Permissões |
|---|---|---|
| `search`, `profile_posts` | `/keyword_search` | `threads_basic` + `threads_keyword_search` |
| `my_posts` | `/me/threads` | `threads_basic` |
| `get_media` | `/{id}` | `threads_basic` |
| `get_profile` | `/profile_lookup` | `threads_basic` + `threads_profile_discovery` |
| `replies` | `/{id}/replies`, `/{id}/conversation` | `threads_basic` + `threads_manage_replies` |

`ThreadsPermissionError` já nomeia as permissões que faltam na própria mensagem.

## Estrutura

| Arquivo | Conteúdo |
|---|---|
| `src/modelo_quant/threads.py` | client, models, erros, paginação, retry |
| `src/modelo_quant/threads_web.py` | backend não-oficial (GraphQL web), opt-in |
| `src/modelo_quant/sentiment.py` | wrapper do pysentimiento |

Os models usam `extra="allow"`: campos novos da Meta não quebram o parsing e ficam
acessíveis como atributos.

## Desenvolvimento

```bash
uv run pytest -q        # offline: sem rede e sem download de modelo
uv run ruff check
uv run ruff format src tests
```
