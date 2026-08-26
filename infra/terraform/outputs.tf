output "gateway_url" {
  value = google_cloud_run_v2_service.gateway.uri
}

output "ci_workload_identity_provider" {
  value = google_iam_workload_identity_pool_provider.github.name
}

output "ci_service_account_email" {
  value = google_service_account.ci.email
}
