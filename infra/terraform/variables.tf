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
