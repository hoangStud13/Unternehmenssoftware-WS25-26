import importlib.util
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent



def run_script(name: str, relative_path: Path):
    path = BASE_DIR / relative_path

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "main"):
        module.main()
    else:
        raise RuntimeError(f"{name} hat keine main()-Funktion")

def run_all_scripts():
    model = input("Which model should run?"+"\n"+"1. LSTM"+"\n"+"2. RNN"+"\n"+"3. Feed_Forward"+"\n")
    run_script(
        "fetch_nasdaq_index",
        Path("scripts/01_data_acquisition/fetch_nasdaq_index.py")
    )
    run_script(
        "fetch_news",
        Path("scripts/01_data_acquisition/fetch_news.py")
    )
    run_script(
        "process_news_sentiment",
        Path("scripts/02_data_understanding/process_news_sentiment.py")
    )
    run_script(
        "process_news_sentiment",
        Path("scripts/03_pre_split_prep/main.py")
    )
    run_script(
        "split_data",
        Path("scripts/04_split_data/split_data.py")
    )
    run_script(
        "post_split_prep",
        Path("scripts/05_post_split_prep/post_split_prep.py")
    )
    if model == "1":
        run_script(
            "model_training",
            Path("scripts/06_model_training/rnn/rnn.py")
        )
        print("Done")
    elif model == "2":
        run_script(
            "model_training",
            Path("scripts/06_model_training/lstm/lstm.py")
        )
        run_script(
            "model_training",
            Path("scripts/07_employment/lstm_deploy.py")
        )
    else:
        run_script(
            "model_training",
            Path("scripts/06_model_training/feed_forward/feed_forward.py")
        )
        run_script(
            "model_training",
            Path("scripts/07_employment/feed_forward_deploy.py")
        )

def run_deployment():
    model=input("Which model should run?"+"\n"+"1. RNN"+"\n"+"2. LSTM"+"\n"+"3. Feed_Forward"+"\n")
    if model == "1":
        run_script(
            "model_training",
            Path("scripts/07_employment/main.py")
        )
    elif model == "2":
        run_script(
            "model_training",
            Path("scripts/07_employment/main.py")
        )
    else:
        run_script(
            "model_training",
            Path("scripts/07_employment/main.py")
        )

def run_data_pipeline():
    run_script("fetch_nasdaq_index", Path("scripts/01_data_acquisition/fetch_nasdaq_index.py"))
    run_script("fetch_news", Path("scripts/01_data_acquisition/fetch_news.py"))
    run_script("process_news_sentiment", Path("scripts/02_data_understanding/process_news_sentiment.py"))
    run_script("pre_split_prep", Path("scripts/03_pre_split_prep/main.py"))
    run_script("split_data", Path("scripts/04_split_data/split_data.py"))
    run_script("post_split_prep", Path("scripts/05_post_split_prep/post_split_prep.py"))

def run_training():
    model = input("Which model should run?"+"\n"+"1. RNN"+"\n"+"2. LSTM"+"\n"+"3. Feed_Forward"+"\n")
    match model:
        case "1":
            run_script("rnn", Path("scripts/06_model_training/rnn/rnn.py"))
        case "2":
            run_script("lstm", Path("scripts/06_model_training/lstm/lstm.py"))
        case "3":
            run_script("feed_forward", Path("scripts/06_model_training/feed_forward/feed_forward.py"))
        case _:
            print("Ungültiges Modell")
            return

def run_backtesting():
    model = input("Which model should run(backtesting)?"+"\n"+"1. LSTM"+"\n"+"2. Feed_Forward"+"\n")
    match model:
        case "1":
            run_script("deployment", Path("scripts/08_backtesting/lstm_backtesting.py"))
        case "2":
            run_script("deployment", Path("scripts/08_backtesting/feed_forward_backtesting.py"))
        case _:
            print("Ungültiges Modell")
            return


def main():
    while True:
        choice = input(
            "\nWas möchtest du ausführen?\n"
            "1 - Gesamte Pipeline (01–07)\n"
            "2 - Nur Datenpipeline (01–05)\n"
            "3 - Training + Deployment (06–07)\n"
            "4 - Nur Deployment\n"
            "5 - Nur Backtesting\n"
            "0 - Beenden\n"
            "Auswahl: "
        ).strip()

        match choice:
            case "1":
                run_all_scripts()
            case "2":
                run_data_pipeline()
            case "3":
                run_training()
            case "4":
                run_deployment()
            case "5":
                run_backtesting()
            case "0":
                print("Programm beendet.")
                break
            case _:
                print("❌ Ungültige Auswahl, bitte erneut versuchen.")

if __name__ == "__main__":
    main()
