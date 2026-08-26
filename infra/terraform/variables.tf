variable "project_id" {
  description = "GCP project id. No default — must be passed explicitly."
  type        = string
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "app_name" {
  type    = string
  default = "llm-gateway"
}

variable "github_repo" {
  description = "GitHub repo allowed to assume the CI service account, as owner/name."
  type        = string
  default     = "AdamczykMaciej/llm-gateway"
}

variable "gateway_image" {
  description = "Full Artifact Registry image ref for the Cloud Run service."
  type        = string
  default     = "" # set via -var at deploy time once an image has been pushed
}

variable "min_instances" {
  type    = number
  default = 0
}

variable "max_instances" {
  type    = number
  default = 3
}

# ── Routing config — the fast lane for changing these is
# `gcloud run services update <app_name> --update-env-vars ...` (seconds,
# no rebuild/redeploy). These Terraform defaults are what a fresh
# `terraform apply` resets to, so a manual gcloud tweak is temporary unless
# also reflected here — that's expected IaC behavior, not a bug. ──────────

variable "provider_order" {
  description = "Comma-separated provider fallback order."
  type        = string
  default     = "anthropic,groq,openai"
}

variable "claude_model" {
  type    = string
  default = "claude-haiku-4-5-20251001"
}

variable "groq_model" {
  type    = string
  default = "llama-3.3-70b-versatile"
}

variable "openai_model" {
  type    = string
  default = "gpt-4o-mini"
}
