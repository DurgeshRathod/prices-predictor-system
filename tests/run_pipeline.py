import click
from mlflow import get_tracking_uri

from pipelines.training_pipeline import ml_pipeline


@click.command()
def main():
    run = ml_pipeline()
    print(
        "Now run \n"
        f"  mlflow ui --backend-store '{get_tracking_uri()}'"
        "To inspect your experiments run within you mlflow UI.\n"
        "You can find your runs tracked within the experiments."
    )


if __name__ == "main":
    main()
