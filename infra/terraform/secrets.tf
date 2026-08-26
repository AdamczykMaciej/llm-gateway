# Secret Manager holds provider keys and the gateway's own client API keys.
# Unlike the interviewer app, this repo does NOT use SOPS — it's a small
# greenfield service, so Terraform only creates placeholder secret *shells*
# with a dummy initial version. Populate real values after the first apply:
#
#   echo -n "sk-ant-real-key" | gcloud secrets versions add \
#     llm-gateway-anthropic-api-key --project=<project_id> --data-file=-
#
# Terraform never sees or manages the real values, so `terraform plan` will
# never show a diff for them.

locals {
  gateway_secret_ids = [
    "anthropic-api-key",
    "groq-api-key",
    "openai-api-key",
    "gateway-api-keys",
  ]
}

resource "google_secret_manager_secret" "gateway" {
  for_each  = toset(local.gateway_secret_ids)
  project   = var.project_id
  secret_id = "${var.app_name}-${each.value}"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "gateway_placeholder" {
  for_each = toset(local.gateway_secret_ids)
  secret   = google_secret_manager_secret.gateway[each.value].id

  # Placeholder only — real values are added out-of-band (see comment above)
  # and Terraform's `ignore_changes` keeps it from ever reverting them.
  secret_data = "REPLACE_ME"

  lifecycle {
    ignore_changes = [secret_data]
  }
}

resource "google_secret_manager_secret_iam_member" "gateway_access" {
  for_each  = toset(local.gateway_secret_ids)
  project   = var.project_id
  secret_id = google_secret_manager_secret.gateway[each.value].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}
