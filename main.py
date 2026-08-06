"""Demo end-to-end: busca na rede, posts de um perfil, comentarios e sentimento.

    uv run main.py

Precisa de THREADS_ACCESS_TOKEN no .env. Para o passo de sentimento:
    uv sync --extra sentiment
"""

from modelo_quant import ThreadsClient, ThreadsError, ThreadsPermissionError

TERMO = "carne bovina"
PERFIL = "jbs_brasil"


def titulo(texto: str) -> None:
    print(f"\n{'=' * 70}\n{texto}\n{'=' * 70}")


def linha(post) -> None:
    quando = post.timestamp.date().isoformat() if post.timestamp else "?"
    texto = (post.text or "").replace("\n", " ")[:80]
    print(f"  [{quando}] @{post.username or '?':<20} {texto}")


def buscar_na_rede(client: ThreadsClient) -> list:
    titulo(f"1. Busca em toda a rede: {TERMO!r}")
    posts = list(client.search(TERMO, search_type="RECENT", max_items=10))
    for p in posts:
        linha(p)
    print(f"\n  {len(posts)} posts.")
    return posts


def posts_do_perfil(client: ThreadsClient) -> None:
    titulo(f"2. Posts do perfil @{PERFIL}")
    # O caminho oficial exige um termo: nao existe endpoint de timeline de terceiros.
    posts = list(client.profile_posts(PERFIL, query=["carne", "exportacao"], max_items=10))
    for p in posts:
        linha(p)
    print(f"\n  {len(posts)} posts (subconjunto filtrado pelos termos, nao o timeline).")
    print("  Para o timeline completo: profile_posts(PERFIL, unofficial=True) -- backend web.")


def comentarios(client: ThreadsClient, posts: list) -> None:
    titulo("3. Comentarios do primeiro post com replies")
    alvo = next((p for p in posts if p.has_replies), None)
    if alvo is None:
        print("  Nenhum dos posts da busca tem replies.")
        return

    print(f"  post {alvo.id}: {(alvo.text or '')[:70]}\n")
    for c in client.replies(alvo.id, max_items=10):
        linha(c)


def sentimento(posts: list) -> None:
    titulo("4. Sentimento dos posts da busca")
    # Import local: uma rodada sem este passo nao carrega torch.
    from modelo_quant.sentiment import analyze_posts, summarize

    resultados = []
    for post, s in analyze_posts(posts):
        resultados.append(s)
        rotulo = f"{s.label} {s.score:.0%}" if s else "sem texto"
        print(f"  {rotulo:<12} {(post.text or '')[:60]}")

    resumo = summarize(resultados)
    print(
        f"\n  {resumo['analisados']} analisados | {resumo['contagem']} | "
        f"polaridade media {resumo['polaridade_media']:+.2f}"
    )


def main() -> None:
    try:
        with ThreadsClient() as client:
            posts = buscar_na_rede(client)

            try:
                posts_do_perfil(client)
            except ThreadsPermissionError as e:
                print(f"\n  Sem permissao: {e}")

            try:
                comentarios(client, posts)
            except ThreadsPermissionError as e:
                print(f"\n  Sem permissao: {e}")

            if posts:
                sentimento(posts)
    except ThreadsError as e:
        print(f"\nFalhou: {e}")
    except ImportError as e:
        print(f"\n{e}")


if __name__ == "__main__":
    main()
