import json

from app.profiling.profiler import profile_csv


if __name__ == "__main__":
    result = profile_csv("data/sample/orders.csv")

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )