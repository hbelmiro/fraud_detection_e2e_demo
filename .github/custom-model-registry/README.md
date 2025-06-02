# Custom Model Registry Overlay

This directory contains a custom Kustomize overlay for the Kubeflow Model Registry that addresses the "mbind: Operation not permitted" error in the MySQL container.

## Problem

The MySQL container in the Model Registry deployment attempts to use the `mbind()` system call to control memory placement policy but doesn't have the necessary permissions in the container environment. This results in the following error message in the logs:

```
mbind: Operation not permitted
```

While this error is often just a warning and doesn't affect MySQL's functionality, it can cause the pod to be marked as unhealthy in some environments.

## Solution

The custom overlay adds the `SYS_NICE` capability to the MySQL container's security context, which grants the necessary permissions for memory binding operations:

```yaml
securityContext:
  capabilities:
    add:
    - SYS_NICE
```

## Implementation

The overlay consists of two files:

1. `overlays/mysql-fix/kustomization.yaml` - References the original model registry manifests and applies our patch
2. `overlays/mysql-fix/mysql-patch.yaml` - Contains the security context patch for the MySQL container

## Usage

In the GitHub workflow, instead of applying the model registry manifests directly from GitHub:

```yaml
kubectl apply -k "https://github.com/kubeflow/model-registry/manifests/kustomize/overlays/db?ref=v0.2.16"
```

We use our custom overlay:

```yaml
kubectl apply -k ".github/custom-model-registry/overlays/mysql-fix"
```

This approach allows us to fix the issue without modifying the original manifests, making it easier to upgrade to newer versions of the model registry in the future.