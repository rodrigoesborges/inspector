group "default" {
  targets = ["backend", "ui"]
}

variable "TAG" {
  default = "latest"
}

target "backend" {
  context = "./backend"
  dockerfile = "Dockerfile"
  tags = ["myorg/backend:${TAG}"]

  cache-from = ["type=local,src=./.docker-cache"]
  cache-to   = ["type=local,dest=./.docker-cache,mode=max"]
}

target "ui" {
  context = "./ui"
  dockerfile = "Dockerfile"
  tags = ["myorg/ui:${TAG}"]

  cache-from = ["type=local,src=./.docker-cache"]
  cache-to   = ["type=local,dest=./.docker-cache,mode=max"]
}