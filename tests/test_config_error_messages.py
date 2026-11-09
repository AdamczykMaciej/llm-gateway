"""Every GatewayConfig validation error names the field, shows what was given
(redacted when secret-like) and says how to fix it."""

import json

import pytest
from pydantic import ValidationError

from llm_gateway import GatewayConfig, ProviderMetadata


def _msg(**kwargs) -> str:
    with pytest.raises(ValidationError) as exc:
        GatewayConfig(**kwargs)
    return str(exc.value)


def _cred_file(tmp_path, body: dict) -> str:
    path = tmp_path / "cred.json"
    path.write_text(json.dumps(body))
    return str(path)


CASES = [
    ("anthropic_thinking", {"anthropic_thinking": "sometimes"}, "'sometimes'"),
    ("anthropic_effort", {"anthropic_effort": "extreme"}, "'extreme'"),
    ("azure_auth", {"azure_auth": "magic"}, "'magic'"),
    ("azure_reasoning_effort", {"azure_reasoning_effort": "ultra"}, "'ultra'"),
    ("azure_region", {"azure_region": "West Europe!"}, "'West Europe!'"),
    ("azure_data_zone", {"azure_data_zone": "EU Zone"}, "'EU Zone'"),
    ("azure_max_tokens_param", {"azure_max_tokens_param": "limit"}, "'limit'"),
    ("require_residency", {"require_residency": "Not A Region!"}, "'Not A Region!'"),
    ("vertex_project_id", {"vertex_project_id": "Bad_Project"}, "'Bad_Project'"),
    ("gemini_project_id", {"gemini_project_id": "Bad_Project"}, "'Bad_Project'"),
    ("vertex_location", {"vertex_location": "mars-1"}, "'mars-1'"),
    ("gemini_location", {"gemini_location": "mars-1"}, "'mars-1'"),
    (
        "vertex_impersonate_service_account",
        {"vertex_impersonate_service_account": "not-an-email"},
        "'not-an-email'",
    ),
    ("openai_compat_base_url", {"openai_compat_base_url": "ftp://x.example"}, "'ftp://x.example'"),
    ("azure_endpoint", {"azure_endpoint": "http://x.example"}, "'http://x.example'"),
    ("huggingface_base_url", {"huggingface_base_url": "http://x.example"}, "'http://x.example'"),
    (
        "azure_endpoint",
        {"azure_endpoint": "https://169.254.169.254/"},
        "'https://169.254.169.254/'",
    ),
    (
        "model_prices",
        {"model_prices": {"gpt-4o": {"input_per_mtok": 1, "output_per_mtok": 2}}},
        "'gpt-4o'",
    ),
    ("provider_metadata", {"provider_metadata": {"opnai": ProviderMetadata()}}, "'opnai'"),
    ("policy_only", {"policy_only": "opnai"}, "'opnai'"),
    ("policy_residency", {"policy_residency": "Nowhere Land"}, "'Nowhere Land'"),
    ("policy_sort", {"policy_sort": "cheapest"}, "'cheapest'"),
]


@pytest.mark.parametrize(
    ("field", "kwargs", "given"), CASES, ids=[c[1].__repr__()[:40] for c in CASES]
)
def test_error_names_field_value_and_fix(field, kwargs, given):
    msg = _msg(**kwargs)
    assert field in msg, msg
    assert f"Got {given}" in msg, msg
    assert "Fix:" in msg, msg


def test_url_credentials_are_never_echoed():
    msg = _msg(openai_compat_base_url="https://user:hunter2secret@host.example/v1")
    assert "openai_compat_base_url" in msg
    assert "hunter2secret" not in msg
    assert "Fix:" in msg


def test_unparseable_url_with_secret_is_not_echoed():
    msg = _msg(azure_endpoint="https://user:hunter2secret@[bad/")
    assert "hunter2secret" not in msg
    assert "azure_endpoint" in msg


def test_secret_like_field_values_are_redacted():
    from llm_gateway._validation import describe_given

    shown = describe_given("openai_api_key", "sk-supersecretvalue123")
    assert "supersecret" not in shown
    assert "redacted" in shown
    assert describe_given("x_token", "abcdef") != "'abcdef'"
    assert describe_given("region", "eu") == "'eu'"
    assert len(describe_given("region", "x" * 500)) < 220


def test_vertex_credentials_file_errors(tmp_path):
    missing = str(tmp_path / "missing.json")
    msg = _msg(vertex_credentials_file=missing)
    assert "vertex_credentials_file" in msg and missing in msg and "Fix:" in msg

    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    msg = _msg(vertex_credentials_file=str(bad))
    assert "not valid JSON" in msg and "Fix:" in msg

    arr = tmp_path / "arr.json"
    arr.write_text("[]")
    msg = _msg(vertex_credentials_file=str(arr))
    assert "JSON object" in msg and "Fix:" in msg

    path = _cred_file(tmp_path, {"type": "mystery"})
    msg = _msg(vertex_credentials_file=path)
    assert "Got 'mystery'" in msg and "Fix:" in msg

    path = _cred_file(tmp_path, {"type": "service_account", "private_key": "TOPSECRETKEY"})
    msg = _msg(vertex_credentials_file=path)
    assert "TOPSECRETKEY" not in msg and "Fix:" in msg


def test_vertex_cross_field_errors(tmp_path):
    ext = _cred_file(tmp_path, {"type": "external_account"})
    msg = _msg(vertex_azure_app_id_uri="api://x")
    assert "vertex_azure_app_id_uri" in msg and "Got 'api://x'" in msg and "Fix:" in msg
    msg = _msg(vertex_credentials_file=ext, vertex_azure_managed_identity_client_id="abc-123")
    assert "vertex_azure_managed_identity_client_id" in msg and "Fix:" in msg
    aws = _cred_file(
        tmp_path, {"type": "external_account", "credential_source": {"environment_id": "aws1"}}
    )
    msg = _msg(vertex_credentials_file=aws, vertex_azure_app_id_uri="api://x")
    assert "AWS" in msg and "Fix:" in msg
    imp = _cred_file(
        tmp_path,
        {
            "type": "impersonated_service_account",
            "source_credentials": {"type": "authorized_user"},
        },
    )
    msg = _msg(
        vertex_credentials_file=imp,
        vertex_impersonate_service_account="a@p.iam.gserviceaccount.com",
    )
    assert "vertex_impersonate_service_account" in msg and "Fix:" in msg
    ext2 = _cred_file(
        tmp_path,
        {
            "type": "external_account",
            "service_account_impersonation_url": "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/other@p.iam.gserviceaccount.com:generateAccessToken",
        },
    )
    msg = _msg(
        vertex_credentials_file=ext2,
        vertex_impersonate_service_account="a@p.iam.gserviceaccount.com",
    )
    assert "differs" in msg and "Got 'a@p.iam.gserviceaccount.com'" in msg and "Fix:" in msg


def test_secret_field_detection_does_not_swallow_ordinary_fields():
    from llm_gateway._validation import describe_given

    assert describe_given("azure_max_tokens_param", "limit") == "'limit'"
    for name in ("azure_api_key", "hf_token", "gateway_api_keys", "client_secret"):
        assert "redacted" in describe_given(name, "value1234")
