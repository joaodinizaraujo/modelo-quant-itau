from datetime import UTC, date, datetime

import httpx
import pytest
import respx

from modelo_quant.threads import (
    BASE_URL,
    MIN_SEARCH_TIMESTAMP,
    Media,
    ThreadsAPIError,
    ThreadsClient,
    ThreadsError,
    ThreadsPermissionError,
)

# --------------------------------------------------------------------------- setup


def test_token_ausente_erra_com_mensagem_util():
    with pytest.raises(ThreadsError, match="THREADS_ACCESS_TOKEN"):
        ThreadsClient()


def test_token_vem_do_ambiente(monkeypatch):
    monkeypatch.setenv("THREADS_ACCESS_TOKEN", "do-env")
    assert ThreadsClient().access_token == "do-env"


# ---------------------------------------------------------------------- search


@respx.mock(base_url=BASE_URL)
def test_search_monta_os_params(respx_mock, client, fx):
    rota = respx_mock.get("/keyword_search").mock(
        return_value=httpx.Response(200, json=fx("search_page2"))
    )

    list(
        client.search(
            "carne bovina",
            search_type="RECENT",
            media_type="TEXT",
            since=date(2026, 7, 1),
            limit=50,
        )
    )

    params = rota.calls.last.request.url.params
    assert params["q"] == "carne bovina"
    assert params["search_mode"] == "KEYWORD"
    assert params["search_type"] == "RECENT"
    assert params["media_type"] == "TEXT"
    assert params["limit"] == "50"
    assert params["access_token"] == "test-token"
    assert params["since"] == str(int(datetime(2026, 7, 1, tzinfo=UTC).timestamp()))
    assert "id" in params["fields"] and "text" in params["fields"]
    assert "until" not in params  # params None sao descartados


@respx.mock(base_url=BASE_URL)
def test_search_limita_o_limit_ao_maximo_da_api(respx_mock, client, fx):
    rota = respx_mock.get("/keyword_search").mock(
        return_value=httpx.Response(200, json=fx("search_page2"))
    )
    list(client.search("x", limit=500))
    assert rota.calls.last.request.url.params["limit"] == "100"


def test_since_anterior_ao_lancamento_da_rede_erra(client):
    with pytest.raises(ThreadsError, match=str(MIN_SEARCH_TIMESTAMP)):
        list(client.search("x", since=1000))


def test_fields_customizado_e_respeitado(client):
    with respx.mock(base_url=BASE_URL) as mock:
        rota = mock.get("/keyword_search").mock(return_value=httpx.Response(200, json={"data": []}))
        list(client.search("x", fields=["id", "text"]))
        assert rota.calls.last.request.url.params["fields"] == "id,text"


# ------------------------------------------------------------------ paginacao


@respx.mock(base_url=BASE_URL)
def test_paginacao_segue_o_cursor_after(respx_mock, client, fx):
    rota = respx_mock.get("/keyword_search").mock(
        side_effect=[
            httpx.Response(200, json=fx("search_page1")),
            httpx.Response(200, json=fx("search_page2")),
        ]
    )

    posts = list(client.search("carne"))

    assert [p.id for p in posts] == ["1001", "1002", "1003"]
    assert rota.call_count == 2
    assert "after" not in rota.calls[0].request.url.params
    assert rota.calls[1].request.url.params["after"] == "CURSOR_PAGINA2"


@respx.mock(base_url=BASE_URL)
def test_max_items_corta_antes_de_pedir_a_proxima_pagina(respx_mock, client, fx):
    rota = respx_mock.get("/keyword_search").mock(
        return_value=httpx.Response(200, json=fx("search_page1"))
    )

    posts = list(client.search("carne", max_items=2))

    assert len(posts) == 2
    assert rota.call_count == 1  # nao buscou a pagina 2


@respx.mock(base_url=BASE_URL)
def test_cursor_repetido_nao_gera_loop_infinito(respx_mock, client, fx):
    pagina = fx("search_page1")  # after sempre igual a CURSOR_PAGINA2
    pagina["paging"]["cursors"]["after"] = "MESMO"
    respx_mock.get("/keyword_search").mock(return_value=httpx.Response(200, json=pagina))

    posts = list(client.search("carne"))

    assert len(posts) == 4  # duas paginas identicas, depois para


@respx.mock(base_url=BASE_URL)
def test_data_vazia_encerra(respx_mock, client):
    rota = respx_mock.get("/keyword_search").mock(
        return_value=httpx.Response(200, json={"data": [], "paging": {"cursors": {"after": "Z"}}})
    )
    assert list(client.search("nada")) == []
    assert rota.call_count == 1


# --------------------------------------------------------------------- parsing


@respx.mock(base_url=BASE_URL)
def test_parsing_do_media(respx_mock, client, fx):
    respx_mock.get("/keyword_search").mock(
        return_value=httpx.Response(200, json=fx("search_page1"))
    )

    primeiro = next(client.search("carne"))

    assert primeiro.timestamp == datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    assert primeiro.username == "joao"
    assert primeiro.has_replies is True
    # extra="allow": campo novo da Meta nao quebra e continua acessivel
    assert primeiro.campo_novo_da_meta == "preservado por extra=allow"


@respx.mock(base_url=BASE_URL)
def test_get_media_e_get_profile(respx_mock, client, fx):
    respx_mock.get("/1001").mock(return_value=httpx.Response(200, json=fx("media")))
    rota_perfil = respx_mock.get("/profile_lookup").mock(
        return_value=httpx.Response(200, json=fx("profile"))
    )

    assert client.get_media("1001").text.startswith("a carne bovina")

    perfil = client.get_profile("@joao")
    assert perfil.follower_count == 1234
    assert perfil.is_verified is False
    assert rota_perfil.calls.last.request.url.params["username"] == "joao"  # @ removido


# ------------------------------------------------------------------ comentarios


@respx.mock(base_url=BASE_URL)
def test_replies_usa_endpoint_replies(respx_mock, client, fx):
    rota = respx_mock.get("/1001/replies").mock(
        return_value=httpx.Response(200, json=fx("replies"))
    )

    comentarios = list(client.replies("1001"))

    assert [c.id for c in comentarios] == ["2001", "2002"]
    assert comentarios[0].root_post == {"id": "1001"}
    assert rota.calls.last.request.url.params["reverse"] == "true"


@respx.mock(base_url=BASE_URL)
def test_replies_nested_usa_conversation(respx_mock, client, fx):
    rota = respx_mock.get("/1001/conversation").mock(
        return_value=httpx.Response(200, json=fx("replies"))
    )

    list(client.replies("1001", nested=True, reverse=False))

    assert rota.called
    assert rota.calls.last.request.url.params["reverse"] == "false"


# --------------------------------------------------------------- profile_posts


def test_profile_posts_oficial_exige_query(client):
    with pytest.raises(ThreadsError, match="query"):
        client.profile_posts("joao")


def test_profile_posts_username_vazio_aponta_para_my_posts(client):
    with pytest.raises(ThreadsError, match="my_posts"):
        client.profile_posts("@", query="carne")


@respx.mock(base_url=BASE_URL)
def test_profile_posts_manda_author_username(respx_mock, client, fx):
    rota = respx_mock.get("/keyword_search").mock(
        return_value=httpx.Response(200, json=fx("search_page2"))
    )

    list(client.profile_posts("@joao", query="carne"))

    params = rota.calls.last.request.url.params
    assert params["author_username"] == "joao"
    assert params["search_type"] == "RECENT"


@respx.mock(base_url=BASE_URL)
def test_profile_posts_multiplos_termos_deduplica(respx_mock, client, fx):
    # As duas buscas devolvem a mesma pagina: os ids repetidos saem uma vez so.
    rota = respx_mock.get("/keyword_search").mock(
        return_value=httpx.Response(200, json=fx("search_page2"))
    )

    posts = list(client.profile_posts("carlos", query=["boi", "gordo"]))

    assert rota.call_count == 2
    assert [p.id for p in posts] == ["1003"]


@respx.mock(base_url=BASE_URL)
def test_avisa_quando_nenhum_post_e_do_perfil_pedido(respx_mock, client, fx, caplog):
    respx_mock.get("/keyword_search").mock(
        return_value=httpx.Response(200, json=fx("search_page2"))  # posts do carlos
    )

    list(client.profile_posts("joao", query="carne"))

    assert "threads_keyword_search" in caplog.text


@respx.mock(base_url=BASE_URL)
def test_my_posts_usa_me_threads(respx_mock, client, fx):
    rota = respx_mock.get("/me/threads").mock(
        return_value=httpx.Response(200, json=fx("search_page2"))
    )
    assert len(list(client.my_posts())) == 1
    assert rota.called


# ----------------------------------------------------------------------- erros


@respx.mock(base_url=BASE_URL)
def test_erro_de_permissao_nomeia_os_scopes(respx_mock, client, fx):
    respx_mock.get("/keyword_search").mock(
        return_value=httpx.Response(403, json=fx("error_permission"))
    )

    with pytest.raises(ThreadsPermissionError) as exc:
        list(client.search("carne"))

    msg = str(exc.value)
    assert "threads_keyword_search" in msg
    assert "threads_basic" in msg
    assert exc.value.code == 200


@respx.mock(base_url=BASE_URL)
def test_erro_generico_preserva_code_e_subcode(respx_mock, client, fx):
    respx_mock.get("/1001").mock(return_value=httpx.Response(400, json=fx("error_generic")))

    with pytest.raises(ThreadsAPIError) as exc:
        client.get_media("1001")

    assert not isinstance(exc.value, ThreadsPermissionError)
    assert (exc.value.code, exc.value.subcode) == (100, 33)
    assert "HTTP 400" in str(exc.value)


@respx.mock(base_url=BASE_URL)
def test_corpo_nao_json_nao_quebra_o_mapeamento(respx_mock, client):
    respx_mock.get("/1001").mock(return_value=httpx.Response(502, text="<html>bad gateway</html>"))

    with pytest.raises(ThreadsAPIError, match="bad gateway"):
        client.get_media("1001")


# ----------------------------------------------------------------------- retry


@respx.mock(base_url=BASE_URL)
def test_retry_em_429_e_depois_sucesso(respx_mock, client, fx):
    rota = respx_mock.get("/keyword_search").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "1"}, json={"error": {"message": "slow"}}),
            httpx.Response(200, json=fx("search_page2")),
        ]
    )

    assert len(list(client.search("carne"))) == 1
    assert rota.call_count == 2


@respx.mock(base_url=BASE_URL)
def test_retry_desiste_depois_de_tres_tentativas(respx_mock, client):
    rota = respx_mock.get("/keyword_search").mock(
        return_value=httpx.Response(503, json={"error": {"message": "indisponivel", "code": 2}})
    )

    with pytest.raises(ThreadsAPIError, match="indisponivel"):
        list(client.search("carne"))

    assert rota.call_count == 3


# ------------------------------------------------------- backend nao-oficial


@respx.mock
def test_backend_web_normaliza_para_media(respx_mock, client, fx):
    respx_mock.get("https://www.threads.com/@joao").mock(
        return_value=httpx.Response(
            200,
            text='...{"LSD",[],{"token":"tok123"},1}... "props":{"user_id":"9001"} ...',
        )
    )
    rota_gql = respx_mock.post("https://www.threads.com/api/graphql").mock(
        return_value=httpx.Response(200, json=fx("web_graphql"))
    )

    posts = list(client.profile_posts("joao", unofficial=True))

    assert [p.id for p in posts] == ["3001", "3002"]
    assert posts[0].text == "timeline completo via backend web"
    assert posts[0].permalink == "https://www.threads.com/@joao/post/CCC"
    assert posts[0].timestamp is not None
    assert posts[0].like_count == 42  # campo extra que so o web expoe
    assert posts[1].media_type == "VIDEO"

    corpo = rota_gql.calls.last.request.content.decode()
    assert "lsd=tok123" in corpo
    assert rota_gql.calls.last.request.headers["X-FB-LSD"] == "tok123"


@respx.mock
def test_backend_web_max_items(respx_mock, client, fx):
    respx_mock.get("https://www.threads.com/@joao").mock(
        return_value=httpx.Response(
            200, text='{"LSD",[],{"token":"t"},1} "props":{"user_id":"9001"}'
        )
    )
    respx_mock.post("https://www.threads.com/api/graphql").mock(
        return_value=httpx.Response(200, json=fx("web_graphql"))
    )

    assert len(list(client.profile_posts("joao", unofficial=True, max_items=1))) == 1


@respx.mock
def test_backend_web_avisa_quando_o_html_muda(respx_mock, client):
    respx_mock.get("https://www.threads.com/@joao").mock(
        return_value=httpx.Response(200, text="<html>nada de util aqui</html>")
    )

    with pytest.raises(ThreadsError, match="THREADS_WEB_DOC_ID"):
        list(client.profile_posts("joao", unofficial=True))


@respx.mock
def test_backend_web_perfil_inexistente(respx_mock, client):
    respx_mock.get("https://www.threads.com/@fantasma").mock(
        return_value=httpx.Response(404, text="not found")
    )

    with pytest.raises(ThreadsError, match="nao encontrado"):
        list(client.profile_posts("fantasma", unofficial=True))


def test_media_aceita_campos_extras_sem_quebrar():
    m = Media.model_validate({"id": "1", "coisa_nova": 42})
    assert m.coisa_nova == 42
    assert m.text is None
