terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }

  # Remote state — required so CI (ephemeral runners) and local runs share
  # the same state. Backend blocks can't reference variables, so this is a
  # literal bucket name; it's created/managed by state_bucket.tf as an
  # ordinary resource, imported into itself (see ../../README.md bootstrap order).
  backend "gcs" {
    bucket = "llm-gateway-mca01-tfstate"
    prefix = "llm-gateway"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
