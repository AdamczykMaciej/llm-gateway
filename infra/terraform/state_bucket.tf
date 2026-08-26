# The state bucket itself, managed by Terraform for documentation/consistency
# even though it had to be created out-of-band first (chicken-and-egg: the
# backend block in providers.tf needs a bucket to already exist before
# `terraform init` can even run). See ../../README.md bootstrap order.
#
# Protections applied here, deliberately beyond the interviewer app's own
# state bucket (worth back-porting there too):
# - prevent_destroy: this resource manages the very bucket its own state
#   lives in — an accidental `terraform destroy` or resource removal here
#   would be unrecoverable without this.
# - public_access_prevention = "enforced": belt-and-suspenders against any
#   future IAM binding accidentally granting allUsers/allAuthenticatedUsers.
# - lifecycle_rule: versioning is on (recovery from a bad/corrupted state
#   write) but without an expiry, noncurrent versions accumulate forever.
#   Keep 90 days of history, then let them age out.
#
# Locking: the GCS backend (see providers.tf) uses the bucket's own object
# generation preconditions for state locking — no separate lock table needed.
#
# Secrets never enter this state: every Secret Manager *value* in this stack
# (see secrets.tf) is written out-of-band via `gcloud secrets versions add`
# and excluded from Terraform's view with `lifecycle { ignore_changes }` —
# only the placeholder string "REPLACE_ME" is ever recorded here.

resource "google_storage_bucket" "tfstate" {
  project                     = var.project_id
  name                        = "${var.project_id}-tfstate"
  location                    = upper(var.region)
  force_destroy               = false
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      num_newer_versions         = 20
      days_since_noncurrent_time = 90
    }
    action {
      type = "Delete"
    }
  }

  lifecycle {
    prevent_destroy = true
  }
}
