# Security policy

## Reporting a vulnerability

Please do not open a public issue. Report it privately through GitHub's
"Report a vulnerability" button (Security tab) on
[AdamczykMaciej/llm-gateway](https://github.com/AdamczykMaciej/llm-gateway/security/advisories/new).
Include the version, a minimal reproduction and the impact. Expect an
acknowledgement within a few days. Fixes ship as a patch release with a
CHANGELOG entry.

Supported version: the latest release only.

## How secrets are handled

- Credentials come from the environment (or `GatewayConfig(...)` arguments),
  never from files in the package. Do not commit `.env`; only SOPS ciphertext
  belongs in a repository.
- `GatewayConfig.__repr__`/`str()` omit provider API keys and `GATEWAY_API_KEYS`.
- Provider error text is scrubbed (`errors.redact_secrets`) before it enters
  an exception message, a span status or the service's 503 body: configured
  keys, bearer tokens, `api-key: ...` pairs and common key shapes.
- Traces carry token counts, never prompt text, unless `TRACE_INCLUDE_PROMPTS=true`,
  and then only after `mask_pii()` (best-effort regex, not a certified detector).
- Error messages never include prompt content or model output.
- Vertex key files are rejected: use Workload Identity Federation or ADC.
- The HTTP service is open when `GATEWAY_API_KEYS` is unset. Set it before
  exposing the service, and keep `SSL_VERIFY` at its default `true`.
