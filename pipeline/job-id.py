from kfp import dsl


@dsl.component(base_image="python:3.10")
def print_op():
    print("Job ID:", dsl.PIPELINE_TASK_NAME_PLACEHOLDER)
    print("workflow uid:", "{{workflow.uid}}")


@dsl.pipeline
def my_pipeline():
    print_op_task = print_op()
    print_op_task.set_caching_options(False)
