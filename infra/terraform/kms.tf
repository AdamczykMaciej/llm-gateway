# KMS key SOPS uses to encrypt/decrypt infra/terraform/secrets.enc.yaml.
# The encrypted file is safe to commit; only holders of decrypt permission
# on this key (granted below) can read the real values.

resource "google_kms_key_ring" "sops" {
  project  = var.project_id
  name     = "${var.app_name}-sops"
  location = "global" # SOPS's gcp_kms backend expects a global-style resource name

  depends_on = [google_project_service.apis]
}

resource "google_kms_crypto_key" "sops" {
  name     = "secrets"
  key_ring = google_kms_key_ring.sops.id

  lifecycle {
    prevent_destroy = true # losing this key makes secrets.enc.yaml unrecoverable
  }
}

variable "sops_admin_member" {
  description = "IAM member (e.g. user:you@example.com) allowed to encrypt/decrypt locally with sops."
  type        = string
}

resource "google_kms_crypto_key_iam_member" "admin" {
  crypto_key_id = google_kms_crypto_key.sops.id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = var.sops_admin_member
}

resource "google_kms_crypto_key_iam_member" "ci" {
  crypto_key_id = google_kms_crypto_key.sops.id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = "serviceAccount:${google_service_account.ci.email}"
}

output "sops_kms_key" {
  value = google_kms_crypto_key.sops.id
}
