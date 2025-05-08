## Install the Spark Operator

https://www.kubeflow.org/docs/components/spark-operator/getting-started/

```shell
helm install spark-operator spark-operator/spark-operator \
    --namespace spark-operator \
    --create-namespace
```

### Configure the Spark Operator to watch all namespaces

Make sure the Kubeflow Spark Operator is watching the my-spark-job namespace. Run this command to let it watch all
namespaces:

```shell
helm upgrade spark-operator spark-operator/spark-operator --set spark.jobNamespaces={} --namespace spark-operator
```
