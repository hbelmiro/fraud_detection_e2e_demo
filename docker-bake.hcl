variable "TAG" {
  default = "latest"
}

group "default" {
  targets = [
    "pipeline",
    "data-preparation",
    "feature-engineering",
    "train",
    "rest-predictor"
  ]
}

target "pipeline" {
  context = "./pipeline"
  dockerfile = "Containerfile"
  tags = ["quay.io/hbelmiro/fraud-detection-e2e-demo-pipeline-rhoai:${TAG}"]
  platforms = ["linux/amd64"]
}

target "data-preparation" {
  context = "./data_preparation"
  dockerfile = "Containerfile"
  tags = ["quay.io/hbelmiro/fraud-detection-e2e-demo-data-preparation-rhoai:${TAG}"]
  platforms = ["linux/amd64"]
}

target "feature-engineering" {
  context = "./feature_engineering"
  dockerfile = "Containerfile"
  tags = ["quay.io/hbelmiro/fraud-detection-e2e-demo-feature-engineering-rhoai:${TAG}"]
  platforms = ["linux/amd64"]
}

target "train" {
  context = "./train"
  dockerfile = "Containerfile"
  tags = ["quay.io/hbelmiro/fraud-detection-e2e-demo-train-rhoai:${TAG}"]
  platforms = ["linux/amd64"]
}

target "rest-predictor" {
  context = "./rest_predictor"
  dockerfile = "Containerfile"
  tags = ["quay.io/hbelmiro/fraud-detection-e2e-demo-rest-predictor-rhoai:${TAG}"]
  platforms = ["linux/amd64"]
}
