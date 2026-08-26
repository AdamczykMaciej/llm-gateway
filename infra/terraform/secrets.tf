# Secret Manager holds provider keys and the gateway's own client API keys.
# Real values come from infra/terraform/secrets.enc.yaml — SOPS-encrypted
# against the Cloud KMS key in kms.tf, safe to commit as ciphertext. Never
# put a real value directly in a .tf file. See secrets.yaml.example for the
# edit workflow.

data "sops_file" "secrets" {
  source_file = "${path.module}/secrets.enc.yaml"
}

locals {
  gateway_secret_values = {
    anthropic-api-key = data.sops_file.secrets.data["anthropic_api_key"]
    groq-api-key      = data.sops_file.secrets.data["groq_api_key"]
    openai-api-key    = data.sops_file.secrets.data["openai_api_key"]
    gateway-api-keys  = data.sops_file.secrets.data["gateway_api_keys"]
  }

  # Secret Manager rejects an empty payload, and every one of these is
  # genuinely optional at first (a fresh deploy might only have one provider
  # key set). Presence is derived only from the plain sops-decrypted values
  # (known at plan time), wrapped in nonsensitive() the same way the
  # interviewer app's secrets.tf does — presence isn't secret, only the
  # values are.
  gateway_secret_present = {
    for k, v in local.gateway_secret_values : k => nonsensitive(v != "")
  }
  gateway_secrets = {
    for k, v in local.gateway_secret_values : k => v if local.gateway_secret_present[k]
  }

  # ENV var name -> secret_id suffix, filtered the same way. Consumed by
  # cloud_run.tf so the two stay in sync automatically.
  gateway_env_secret_refs = {
    for env_name, secret_key in {
      ANTHROPIC_API_KEY = "anthropic-api-key"
      GROQ_API_KEY      = "groq-api-key"
      OPENAI_API_KEY    = "openai-api-key"
      GATEWAY_API_KEYS  = "gateway-api-keys"
    } : env_name => secret_key if local.gateway_secret_present[secret_key]
  }
}

resource "google_secret_manager_secret" "gateway" {
  for_each  = local.gateway_secrets
  project   = var.project_id
  secret_id = "${var.app_name}-${each.key}"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "gateway" {
  for_each    = local.gateway_secrets
  secret      = google_secret_manager_secret.gateway[each.key].id
  secret_data = each.value
}

resource "google_secret_manager_secret_iam_member" "gateway_access" {
  for_each  = local.gateway_secrets
  project   = var.project_id
  secret_id = google_secret_manager_secret.gateway[each.key].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}
