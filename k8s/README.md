# BioForge on Kubernetes

Scaffold manifests, not a production-hardened deployment - review resource
requests/limits, add `NetworkPolicy`/`PodDisruptionBudget`/HPA, and point
`ingress.yaml` at a real host/TLS cert before using this outside a local
cluster (kind/minikube).

```bash
kubectl apply -k k8s/base
kubectl -n bioforge get pods
kubectl -n bioforge port-forward svc/gateway 8000:8000
```

Every service shares one image (`bioforge/bioforge:latest`, built from the
root `Dockerfile`); which service a pod runs is set by the `SERVICE` env var
on its `Deployment` (see `k8s/base/deployments.yaml`). Build and push it
first:

```bash
docker build -t bioforge/bioforge:latest .
kind load docker-image bioforge/bioforge:latest   # for a local kind cluster
```

`generate` (the diffusion/scoring service) is the one place a real GPU
checkpoint would need `nvidia.com/gpu` resource requests and a
GPU-scheduled node pool - see the commented line in its `Deployment`.
`orchestrator` talks to `redis` for the optional Celery/pub-sub path
(`bioforge.common.messaging`); every service still works over plain REST
without it.
