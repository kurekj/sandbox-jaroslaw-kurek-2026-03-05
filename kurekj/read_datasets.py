import os
import pickle
import pandas as pd

def load_pickle(file_path):
    """Wczytuje dane z pliku .pkl"""
    if not os.path.exists(file_path):
        print(f"❌ Plik nie istnieje: {file_path}")
        return None
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"✅ Wczytano dane z: {file_path}")
    return data

def main():
    # Budowanie ścieżki względem lokalizacji tego skryptu
    base_dir = os.path.join(
        os.path.dirname(__file__),
        "..", "src", "v2", "autoencoder", "logs"
    )
    base_dir = os.path.abspath(base_dir)

    train_path = os.path.join(base_dir, "train_data.pkl")
    val_path = os.path.join(base_dir, "val_data.pkl")

    print(f"📂 Szukam plików w: {base_dir}\n")

    # Wczytanie danych
    train_data = load_pickle(train_path)
    val_data = load_pickle(val_path)

    # Ścieżki wyjściowe
    output_dir = os.path.dirname(__file__)
    train_xlsx = os.path.join(output_dir, "train_data.xlsx")
    val_xlsx = os.path.join(output_dir, "val_data.xlsx")
    summary_xlsx = os.path.join(output_dir, "datasets_summary.xlsx")

    # --- ZAPIS TRAIN ---
    if isinstance(train_data, pd.DataFrame):
        train_data.to_excel(train_xlsx, index=False)
        print(f"📊 Zapisano TRAIN DATA do: {train_xlsx}")
    else:
        print(f"⚠️ train_data nie jest DataFrame (typ: {type(train_data)})")
        return

    # --- ZAPIS VAL ---
    if isinstance(val_data, pd.DataFrame):
        val_data.to_excel(val_xlsx, index=False)
        print(f"📊 Zapisano VAL DATA do: {val_xlsx}")
    else:
        print(f"⚠️ val_data nie jest DataFrame (typ: {type(val_data)})")
        return

    # --- TWORZENIE PODSUMOWANIA ---
    train_rows, train_cols = train_data.shape
    val_rows, val_cols = val_data.shape
    total_rows = train_rows + val_rows
    train_pct = (train_rows / total_rows) * 100 if total_rows > 0 else 0
    val_pct = (val_rows / total_rows) * 100 if total_rows > 0 else 0

    summary_data = {
        "Zbiór": ["train_data", "val_data", "SUMA"],
        "Liczba kolumn": [train_cols, val_cols, None],
        "Liczba wierszy": [train_rows, val_rows, total_rows],
        "Udział %": [f"{train_pct:.2f}%", f"{val_pct:.2f}%", "100.00%"]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(summary_xlsx, index=False)

    print(f"\n📈 Utworzono podsumowanie w: {summary_xlsx}")
    print(summary_df)

if __name__ == "__main__":
    main()
