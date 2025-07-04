#!/usr/bin/env bash

set -euo pipefail

kubectl delete sparkapplication --all
kubectl delete inferenceservice --all

kubectl get pods -n fraud-detection --no-headers | awk -v dsp_ns="fraud-detection" '$3 ~ /Completed|Error|Failed|Init:OOMKilled/ {print "kubectl delete pod -n fraud-detection " $1}' | sh