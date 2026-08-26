resource "google_service_account" "runtime" {
  project      = var.project_id
  account_id   = "${var.app_name}-runtime"
  display_name = "${var.app_name} Cloud Run runtime"

  depends_on = [google_project_service.apis]
}

# Auth is enforced at the application layer (GATEWAY_API_KEYS bearer check),
# not at the Cloud Run/IAM layer — same posture as a public API gateway.
resource "google_cloud_run_v2_service_iam_member" "public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.gateway.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
