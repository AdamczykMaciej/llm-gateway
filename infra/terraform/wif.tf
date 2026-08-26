# Lets GitHub Actions authenticate to GCP without any stored long-lived
# key — the workflow exchanges its OIDC token for short-lived GCP
# credentials, scoped to this exact repo. Own pool, own project — entirely
# separate from any other app's WIF trust boundary.

resource "google_iam_workload_identity_pool" "github" {
  project                   = var.project_id
  workload_identity_pool_id = "github-actions"
  display_name              = "GitHub Actions"

  depends_on = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  project                            = var.project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github"
  display_name                       = "GitHub"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
  }

  # Only this repo's workflows can assume the CI service account below.
  attribute_condition = "assertion.repository == \"${var.github_repo}\""

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account" "ci" {
  project      = var.project_id
  account_id   = "${var.app_name}-ci"
  display_name = "${var.app_name} CI/CD (GitHub Actions)"

  depends_on = [google_project_service.apis]
}

resource "google_service_account_iam_member" "ci_wif_binding" {
  service_account_id = google_service_account.ci.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repo}"
}

# Permissions the CI service account needs to build+push images and run
# `terraform apply` end to end for this stack's resource set.
locals {
  ci_project_roles = [
    "roles/run.admin",
    "roles/artifactregistry.admin",
    "roles/secretmanager.admin",
    "roles/iam.serviceAccountUser",
    "roles/iam.serviceAccountAdmin",
    "roles/serviceusage.serviceUsageAdmin",
    "roles/iam.workloadIdentityPoolAdmin",
    # Needed because google_project_iam_member.ci (this very list) is itself
    # a Terraform-managed resource that CI refreshes on every untargeted
    # apply — reading/updating project IAM policy requires this.
    "roles/resourcemanager.projectIamAdmin",
  ]
}

resource "google_project_iam_member" "ci" {
  for_each = toset(local.ci_project_roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.ci.email}"
}

resource "google_storage_bucket_iam_member" "ci_tfstate" {
  bucket = google_storage_bucket.tfstate.name
  # storage.admin (not just objectAdmin) because google_storage_bucket.tfstate
  # is itself a Terraform-managed resource — refreshing it needs
  # storage.buckets.get, which objectAdmin doesn't grant. Scoped to just this
  # one bucket via the IAM member binding, not project-wide.
  role   = "roles/storage.admin"
  member = "serviceAccount:${google_service_account.ci.email}"
}
