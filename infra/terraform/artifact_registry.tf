resource "google_artifact_registry_repository" "images" {
  project       = var.project_id
  location      = var.region
  repository_id = var.app_name
  format        = "DOCKER"
  description   = "Container images for ${var.app_name}"

  depends_on = [google_project_service.apis]
}
