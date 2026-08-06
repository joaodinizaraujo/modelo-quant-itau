import subprocess
import sys
from types import SimpleNamespace

import pytest

from modelo_quant import sentiment
from modelo_quant.threads import Media


class AnalyzerFake:
    """Imita o analyzer do pysentimiento: predict aceita str ou lista."""

    def __init__(self):
        self.chamadas: list[object] = []

    def _um(self, texto: str):
        positivo = "otimo" in texto or "adorei" in texto
        probas = (
            {"POS": 0.9, "NEG": 0.05, "NEU": 0.05}
            if positivo
            else {
                "POS": 0.1,
                "NEG": 0.8,
                "NEU": 0.1,
            }
        )
        return SimpleNamespace(output="POS" if positivo else "NEG", probas=probas)

    def predict(self, entrada):
        self.chamadas.append(entrada)
        if isinstance(entrada, str):
            return self._um(entrada)
        return [self._um(t) for t in entrada]


@pytest.fixture
def analyzer(monkeypatch):
    fake = AnalyzerFake()
    monkeypatch.setattr(sentiment, "_load", lambda task, lang: fake)
    return fake


# ------------------------------------------------------------------- analyze


def test_analyze_retorna_label_e_score(analyzer):
    s = sentiment.analyze("adorei esse produto")
    assert s.label == "POS"
    assert s.score == pytest.approx(0.9)
    assert s.polarity == 1
    assert s.lang == "pt"


def test_analyze_negativo(analyzer):
    assert sentiment.analyze("odiei, veio quebrado").polarity == -1


@pytest.mark.parametrize("texto", ["", "   ", "\n"])
def test_texto_vazio_nao_chama_o_modelo(analyzer, texto):
    assert sentiment.analyze(texto) is None
    assert analyzer.chamadas == []


def test_task_multilabel_devolve_lista_de_rotulos(monkeypatch):
    """`emotion` e `hate_speech` retornam output como lista, nao string."""

    class MultiLabel:
        def predict(self, _entrada):
            return SimpleNamespace(
                output=["joy", "surprise"],
                probas={"joy": 0.7, "surprise": 0.9, "anger": 0.02},
            )

    monkeypatch.setattr(sentiment, "_load", lambda task, lang: MultiLabel())

    s = sentiment.analyze("estou muito feliz", task="emotion")

    assert s.labels == ("joy", "surprise")
    assert s.label == "surprise"  # o de maior probabilidade entre os marcados
    assert s.score == pytest.approx(0.9)
    assert s.polarity == 0  # rotulo que nao e POS/NEG nao entra na polaridade


def test_multilabel_vazio_vira_none(monkeypatch):
    """hate_speech em texto inofensivo marca zero rotulos."""

    class Nenhum:
        def predict(self, _entrada):
            return SimpleNamespace(output=[], probas={"hateful": 0.01})

    monkeypatch.setattr(sentiment, "_load", lambda task, lang: Nenhum())

    s = sentiment.analyze("bom dia", task="hate_speech")

    assert s.label == "NONE"
    assert s.labels == ()
    assert s.score == 0.0


def test_sem_output_cai_na_maior_probabilidade(monkeypatch):
    class SemOutput:
        def predict(self, _entrada):
            return SimpleNamespace(probas={"POS": 0.2, "NEG": 0.75, "NEU": 0.05})

    monkeypatch.setattr(sentiment, "_load", lambda task, lang: SemOutput())

    assert sentiment.analyze("qualquer coisa").label == "NEG"


# -------------------------------------------------------------- analyze_many


def test_analyze_many_alinha_por_indice(analyzer):
    resultados = sentiment.analyze_many(["adorei", "", "odiei"])

    assert len(resultados) == 3
    assert resultados[0].label == "POS"
    assert resultados[1] is None  # vazio mantem a posicao
    assert resultados[2].label == "NEG"


def test_analyze_many_batcha(analyzer):
    sentiment.analyze_many([f"texto {i}" for i in range(10)], batch_size=4)

    # 10 textos em lotes de 4 -> 3 chamadas de 4, 4 e 2
    assert [len(c) for c in analyzer.chamadas] == [4, 4, 2]


def test_analyze_many_tudo_vazio_nao_chama_o_modelo(analyzer):
    assert sentiment.analyze_many(["", "  "]) == [None, None]
    assert analyzer.chamadas == []


# ------------------------------------------------------------- analyze_posts


def test_analyze_posts_anota_cada_post(analyzer):
    posts = [
        Media(id="1", text="otimo resultado"),
        Media(id="2", text="pessimo trimestre"),
        Media(id="3", text=None),
    ]

    pares = list(sentiment.analyze_posts(posts))

    assert [p.id for p, _ in pares] == ["1", "2", "3"]
    assert [s.label if s else None for _, s in pares] == ["POS", "NEG", None]


def test_analyze_posts_consome_lazy(analyzer):
    """Recebe um generator infinito: so o primeiro lote deve ser analisado."""

    def infinito():
        i = 0
        while True:
            i += 1
            yield Media(id=str(i), text="otimo")

    pares = sentiment.analyze_posts(infinito(), batch_size=2)
    primeiro = next(pares)

    assert primeiro[0].id == "1"
    assert len(analyzer.chamadas) == 1  # nao consumiu o generator inteiro


# ----------------------------------------------------------------- summarize


def test_summarize_agrega(analyzer):
    resultados = sentiment.analyze_many(["adorei", "otimo", "odiei", ""])

    resumo = sentiment.summarize(resultados)

    assert resumo["total"] == 4
    assert resumo["analisados"] == 3
    assert resumo["ignorados"] == 1
    assert resumo["contagem"] == {"POS": 2, "NEG": 1}
    assert resumo["share"]["POS"] == pytest.approx(2 / 3)
    assert resumo["polaridade_media"] == pytest.approx(1 / 3)


def test_summarize_vazio_nao_divide_por_zero():
    resumo = sentiment.summarize([])
    assert resumo == {
        "total": 0,
        "analisados": 0,
        "ignorados": 0,
        "contagem": {},
        "share": {},
        "polaridade_media": 0.0,
    }


# ------------------------------------------------------------ isolamento do torch


def test_importar_o_client_nao_arrasta_torch():
    """A garantia central: `import modelo_quant` nao pode custar 2,5 GB de ML."""
    codigo = (
        "import modelo_quant, sys; "
        "pesados = {m.split('.')[0] for m in sys.modules} & "
        "{'torch', 'transformers', 'pysentimiento', 'datasets', 'spacy'}; "
        "assert not pesados, pesados"
    )
    r = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr


def test_importar_o_modulo_de_sentimento_tambem_e_barato():
    """Importar sentiment.py nao carrega o modelo; so `analyze()` carrega."""
    codigo = (
        "from modelo_quant.sentiment import analyze; import sys; assert 'torch' not in sys.modules"
    )
    r = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr


def test_sem_o_extra_instalado_o_erro_explica_como_instalar(monkeypatch):
    import builtins

    real = builtins.__import__

    def falha(nome, *args, **kwargs):
        if nome == "pysentimiento":
            raise ImportError("no module named pysentimiento")
        return real(nome, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", falha)
    sentiment._load.cache_clear()

    with pytest.raises(ImportError, match="uv sync --extra sentiment"):
        sentiment.analyze("texto qualquer")

    sentiment._load.cache_clear()
