resource "google_cloud_run_v2_service" "gateway" {
  project             = var.project_id
  name                = var.app_name
  location            = var.region
  deletion_protection = false
  ingress             = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.runtime.email

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    containers {
      image = var.gateway_image

      ports {
        container_port = 8080
      }

      env {
        name  = "PROVIDER_ORDER"
        value = "anthropic,groq,openai"
      }

      dynamic "env" {
        for_each = {
          ANTHROPIC_API_KEY = "anthropic-api-key"
          GROQ_API_KEY      = "groq-api-key"
          OPENAI_API_KEY    = "openai-api-key"
          GATEWAY_API_KEYS  = "gateway-api-keys"
        }
        content {
          name = env.key
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.gateway[env.value].secret_id
              version = "latest"
            }
          }
        }
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }

      startup_probe {
        http_get {
          path = "/healthz"
          port = 8080
        }
        initial_delay_seconds = 5
        period_seconds        = 5
        failure_threshold     = 6
      }
    }
  }

  depends_on = [
    google_project_service.apis,
    google_secret_manager_secret_version.gateway_placeholder,
  ]
}
