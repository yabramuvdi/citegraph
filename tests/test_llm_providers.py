"""Provider contracts exercised through real SDKs and offline HTTP transports."""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from tenacity import wait_none

from citegraph.config import Settings
from citegraph.extract_metadata import extract_metadata_from_markdown
from citegraph.extract_references import extract_references_from_markdown
from citegraph.llm import (
    AnthropicClient,
    GeminiClient,
    LLMError,
    LLMResponse,
    OpenAIClient,
    create_llm_client,
    parse_structured_response,
)
from citegraph.pipeline import Pipeline
from citegraph.schemas import PaperMetadata, Reference

METADATA = {"Title": "A study", "Authors_List": ["Jane Smith"], "Journal": "J", "Year": 2020}


def response(provider, payload=None, *, truncated=False, refusal=False, text=None):
    text = text if text is not None else json.dumps({"result": payload})
    if provider == "openai":
        content = (
            [{"type": "refusal", "refusal": "Declined"}]
            if refusal else [{"type": "output_text", "text": text, "annotations": []}]
        )
        return {
            "id": "resp_test", "object": "response", "created_at": 0,
            "model": "test-model", "status": "incomplete" if truncated else "completed",
            "incomplete_details": {"reason": "max_output_tokens"} if truncated else None,
            "output": [{"type": "message", "id": "msg_test", "role": "assistant",
                        "status": "completed", "content": content}],
            "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
        }
    return {
        "id": "msg_test", "type": "message", "role": "assistant", "model": "test-model",
        "content": [{"type": "text", "text": text}], "stop_sequence": None,
        "stop_reason": "refusal" if refusal else "max_tokens" if truncated else "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 10},
    }


def client_for(provider, monkeypatch, replies):
    requests = []
    replies = iter(replies)

    def serve(request):
        requests.append(json.loads(request.content))
        status, body = next(replies)
        return http.Response(status, json=body)

    sdk = pytest.importorskip(provider)
    # The SDKs support HTTPX or HTTPX2 depending on their major version.
    cls = sdk.OpenAI if provider == "openai" else sdk.Anthropic
    annotation = str(inspect.signature(cls).parameters["http_client"].annotation)
    http = importlib.import_module("httpx2" if "httpx2" in annotation else "httpx")
    transport = http.Client(transport=http.MockTransport(serve))
    settings = Settings(_env_file=None)
    if provider == "openai":
        client = OpenAIClient(api_key="test", model="test-model", settings=settings)
        client._client = sdk.OpenAI(api_key="test", http_client=transport, max_retries=0)
    else:
        client = AnthropicClient(api_key="test", model="test-model", settings=settings)
        client._client = sdk.Anthropic(api_key="test", http_client=transport, max_retries=0)
    monkeypatch.setattr(client._request.retry, "wait", wait_none())
    return client, requests


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_structured_metadata_request_and_validation(provider, monkeypatch):
    client, requests = client_for(provider, monkeypatch, [(200, response(provider, METADATA))])
    result = client.generate_structured(
        prompt="Extract this paper", system_instruction="Bibliographic extraction",
        response_schema=PaperMetadata, max_output_tokens=1000,
    )
    assert parse_structured_response(result, PaperMetadata).Title == "A study"
    request = requests[0]
    assert request["model"] == "test-model"
    if provider == "openai":
        assert request["input"] == "Extract this paper"
        assert request["instructions"] == "Bibliographic extraction"
        assert request["max_output_tokens"] == 1000
        assert request["store"] is False
        fmt = request["text"]["format"]
        assert fmt["strict"] is True
    else:
        assert request["messages"] == [{"role": "user", "content": "Extract this paper"}]
        assert request["system"] == "Bibliographic extraction"
        assert request["max_tokens"] == 1000
        fmt = request["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["additionalProperties"] is False
    assert fmt["schema"]["required"] == ["result"]
    assert set(fmt["schema"]["properties"]["result"]["required"]) == set(METADATA)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("records", [[], [METADATA]])
def test_reference_arrays_remain_arrays(provider, records, monkeypatch):
    client, _ = client_for(provider, monkeypatch, [(200, response(provider, records))])
    result = client.generate_structured(prompt="References", response_schema=list[Reference])
    refs = parse_structured_response(result, Reference)
    assert len(refs) == len(records)
    if records:
        assert isinstance(refs[0], Reference)
        assert refs[0].Title == "A study"
        assert refs[0].Authors_List == ["Jane Smith"]
        assert refs[0].Year == 2020


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_refusal_is_a_failure_not_an_empty_bibliography(provider, monkeypatch):
    client, requests = client_for(provider, monkeypatch, [(200, response(provider, refusal=True))])
    with pytest.raises(LLMError, match="refus"):
        client.generate_structured(prompt="References", response_schema=list[Reference])
    assert len(requests) == 1


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_provider_data_still_passes_local_schema_validation(provider, monkeypatch):
    bad = {**METADATA, "Year": "not a year"}
    client, _ = client_for(provider, monkeypatch, [(200, response(provider, bad))])
    with pytest.raises(ValidationError):
        result = client.generate_structured(prompt="Metadata", response_schema=PaperMetadata)
        parse_structured_response(result, PaperMetadata)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_reference_result_must_be_an_array(provider, monkeypatch):
    client, _ = client_for(provider, monkeypatch, [(200, response(provider, METADATA))])
    with pytest.raises(ValidationError):
        client.generate_structured(prompt="References", response_schema=list[Reference])


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("status, attempts", [(400, 1), (429, 3), (503, 3)])
def test_only_transient_http_failures_retry(provider, status, attempts, monkeypatch):
    error = {"error": {"type": "api_error", "message": "Test failure"}}
    client, requests = client_for(provider, monkeypatch, [(status, error)] * attempts)
    with pytest.raises(LLMError):
        client.generate_structured(prompt="Metadata", response_schema=PaperMetadata)
    assert len(requests) == attempts


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_truncation_retries_with_a_bounded_budget(provider, monkeypatch, tmp_path):
    client, requests = client_for(provider, monkeypatch, [
        (200, response(provider, text='{"result":[', truncated=True)),
        (200, response(provider, [METADATA])),
    ])
    md = tmp_path / "paper.md"
    md.write_text("## References\nA study", encoding="utf-8")
    refs = extract_references_from_markdown(md, client=client)
    assert [r.Title for r in refs] == ["A study"]
    key = "max_output_tokens" if provider == "openai" else "max_tokens"
    assert [r[key] for r in requests] == [8000, 16000]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_partial_envelope_can_be_recovered_after_the_last_retry(provider, monkeypatch, tmp_path):
    partial = '{"result":[' + json.dumps(METADATA) + ', {"Title":"incomplete'
    client, _ = client_for(provider, monkeypatch, [
        (200, response(provider, text=partial, truncated=True)),
        (200, response(provider, text=partial, truncated=True)),
    ])
    md = tmp_path / "paper.md"
    md.write_text("## References\nA study", encoding="utf-8")
    assert [r.Title for r in extract_references_from_markdown(md, client=client)] == ["A study"]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_truncated_metadata_is_not_accepted(provider, monkeypatch, tmp_path):
    client, _ = client_for(provider, monkeypatch, [(200, response(provider, METADATA, truncated=True))])
    md = tmp_path / "paper.md"
    md.write_text("A study", encoding="utf-8")
    with pytest.raises(LLMError, match="truncat"):
        extract_metadata_from_markdown(md, client=client)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_missing_optional_sdk_has_installation_hint(provider, monkeypatch):
    monkeypatch.setitem(sys.modules, provider, None)
    cls = OpenAIClient if provider == "openai" else AnthropicClient
    with pytest.raises(ImportError, match=rf"citegraph\[{provider}\]"):
        _ = cls(api_key="test", model="test-model").client


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_new_providers_require_an_explicit_model_for_extraction(provider, tmp_path):
    pipeline = Pipeline(pdf_dir=None, out_dir=tmp_path, provider=provider)
    with pytest.raises(ValueError, match="model"):
        _ = pipeline.model


def test_environment_selects_provider_and_its_key(monkeypatch):
    monkeypatch.setenv("CITEGRAPH_PROVIDER", "anthropic")
    monkeypatch.setenv("CITEGRAPH_MODEL", "selected-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    client = create_llm_client(settings=Settings(_env_file=None))
    assert isinstance(client, AnthropicClient)
    assert client.model == "selected-model"
    assert client._api_key == "anthropic-test"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_switching_provider_invalidates_cache_and_estimate_agrees(provider, tmp_path):
    from tests.test_pipeline import _FakeClient

    class CountingClient(_FakeClient):
        calls = 0

        def generate_structured(self, **kwargs):
            self.calls += 1
            return super().generate_structured(**kwargs)

    client = CountingClient()
    pipeline = Pipeline(pdf_dir=None, out_dir=tmp_path, client=client, show_progress=False)
    (pipeline.layout.markdown_dir / "paper.md").write_text("A study", encoding="utf-8")
    pipeline.extract_paper_metadata()
    assert client.calls == 1
    client.provider = provider
    estimate = pipeline.estimate_extraction_cost()
    assert estimate.n_metadata_to_process == 1
    assert estimate.provider == provider
    assert estimate.cost_usd() is None
    pipeline.extract_paper_metadata()
    assert client.calls == 2


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_non_gemini_does_not_claim_legacy_cache(provider, tmp_path):
    pipeline = Pipeline(pdf_dir=None, out_dir=tmp_path, provider=provider, model="test-model")
    (pipeline.layout.markdown_dir / "paper.md").write_text("A study", encoding="utf-8")
    (pipeline.layout.metadata_dir / "paper.json").write_text(json.dumps(METADATA), encoding="utf-8")
    assert pipeline.estimate_extraction_cost().n_metadata_to_process == 1


def test_custom_client_does_not_need_a_gemini_base_class(tmp_path):
    client = SimpleNamespace(provider="openai", model="custom", default_max_output_tokens=1000)
    pipeline = Pipeline(pdf_dir=None, out_dir=tmp_path, client=client)
    assert pipeline.client is client
    assert pipeline.provider == "openai"
    assert pipeline.model == "custom"


@pytest.mark.parametrize("provider, cls, key", [
    ("gemini", GeminiClient, "GOOGLE_API_KEY"),
    ("openai", OpenAIClient, "OPENAI_API_KEY"),
    ("anthropic", AnthropicClient, "ANTHROPIC_API_KEY"),
])
def test_client_uses_only_its_provider_key_and_applies_timeout(provider, cls, key):
    settings = Settings(
        _env_file=None, GOOGLE_API_KEY="google-test", OPENAI_API_KEY="openai-test",
        ANTHROPIC_API_KEY="anthropic-test", CITEGRAPH_REQUEST_TIMEOUT_S=7,
    )
    client = cls(model="test-model", settings=settings)
    assert client._api_key == {"GOOGLE_API_KEY": "google-test", "OPENAI_API_KEY": "openai-test",
                               "ANTHROPIC_API_KEY": "anthropic-test"}[key]
    if provider == "gemini":
        assert client.client._api_client._http_options.timeout == 7000
    else:
        assert client.client.timeout == 7
        assert client.client.max_retries == 0
    client.client.close()


def test_existing_gemini_recipe_remains_reusable(tmp_path):
    from citegraph.io import write_cache_fingerprint, write_json
    from citegraph.pipeline import extraction_recipe

    pipeline = Pipeline(pdf_dir=None, out_dir=tmp_path, model="test-model")
    md = pipeline.layout.markdown_dir / "paper.md"
    md.write_text("A study", encoding="utf-8")
    cached = pipeline.layout.metadata_dir / "paper.json"
    write_json(cached, METADATA)
    recipe = extraction_recipe("test-model")
    del recipe["provider"]  # The format written before provider selection existed.
    write_cache_fingerprint(cached, md, recipe=recipe)
    assert pipeline.estimate_extraction_cost().n_metadata_to_process == 0


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_same_content_under_a_new_name_does_not_cross_providers(provider, tmp_path):
    from citegraph.io import write_cache_fingerprint, write_json
    from citegraph.pipeline import extraction_recipe

    pipeline = Pipeline(pdf_dir=None, out_dir=tmp_path, provider=provider, model="test-model")
    md = pipeline.layout.markdown_dir / "renamed.md"
    md.write_text("A study", encoding="utf-8")
    cached = pipeline.layout.metadata_dir / "old.json"
    write_json(cached, METADATA)
    write_cache_fingerprint(cached, md, recipe=extraction_recipe("test-model"))
    assert pipeline.estimate_extraction_cost().n_metadata_to_process == 1


def test_cli_estimate_selects_provider_without_sdk_calls(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from citegraph.cli import app

    md = tmp_path / "markdown"
    md.mkdir()
    (md / "paper.md").write_text("A study", encoding="utf-8")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = CliRunner().invoke(app, [
        "estimate", "--out", str(tmp_path), "--provider", "openai", "--model", "selected-model",
    ])
    assert result.exit_code == 0, result.output
    assert "openai" in result.output
    assert "selected-model" in result.output
    assert "unknown" in result.output


def test_local_commands_do_not_need_an_llm_model(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from citegraph.cli import app

    monkeypatch.setenv("CITEGRAPH_PROVIDER", "openai")
    monkeypatch.delenv("CITEGRAPH_MODEL", raising=False)
    result = CliRunner().invoke(app, ["authors", "--out", str(tmp_path)])
    assert "model" not in str(result.exception)
    assert "Run" in result.output  # Normal missing-upstream hint, not an LLM configuration error.


def test_cli_missing_provider_model_is_an_actionable_usage_error(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from citegraph.cli import app

    monkeypatch.delenv("CITEGRAPH_MODEL", raising=False)
    result = CliRunner().invoke(app, ["estimate", "--out", str(tmp_path), "--provider", "openai"])
    assert result.exit_code == 2
    assert "--model" in result.output


def test_changed_output_budget_refreshes_metadata_cache(tmp_path):
    from tests.test_pipeline import _FakeClient

    client = _FakeClient()
    pipeline = Pipeline(pdf_dir=None, out_dir=tmp_path, client=client, show_progress=False)
    (pipeline.layout.markdown_dir / "paper.md").write_text("A study", encoding="utf-8")
    pipeline.extract_paper_metadata()
    assert pipeline.estimate_extraction_cost().n_metadata_to_process == 0
    client._settings = Settings(_env_file=None, CITEGRAPH_METADATA_MAX_OUTPUT_TOKENS=1500)
    assert pipeline.estimate_extraction_cost().n_metadata_to_process == 1


def test_minimal_injected_client_records_the_budgets_actually_used(tmp_path):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs["max_output_tokens"])
        if kwargs["response_schema"] is PaperMetadata:
            return LLMResponse(parsed=METADATA)
        return LLMResponse(truncated=True, text="[") if len(calls) == 2 else LLMResponse(parsed=[])

    client = SimpleNamespace(
        provider="openai", model="custom", default_max_output_tokens=20_000,
        generate_structured=generate,
    )
    pipeline = Pipeline(pdf_dir=None, out_dir=tmp_path, client=client, show_progress=False)
    (pipeline.layout.markdown_dir / "paper.md").write_text("A study", encoding="utf-8")
    pipeline.extract_paper_metadata()
    pipeline.extract_paper_references()
    meta = json.loads((pipeline.layout.metadata_dir / "paper.json.sha256").read_text())
    refs = json.loads((pipeline.layout.references_dir / "paper.json.sha256").read_text())
    assert calls == [500, 20_000, 40_000]
    assert meta["recipe"]["generation"] == {"max_output_tokens": 500}
    assert refs["recipe"]["generation"] == {"max_output_tokens": 20_000, "max_retry_output_tokens": 80_000}
    assert pipeline.estimate_extraction_cost().n_metadata_to_process == 0
