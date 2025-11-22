import pandas as pd
import os

RAW_PATH = "data/raw/telco_churn.csv"
PROCESSED_PATH = "data/processed/data_clean.csv"

def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    # estandarizamos nombres de columnas
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # eliminar duplicados
    if "customer_id" in df.columns:
        df = df.drop_duplicates(subset="customer_id")

    # limpiar strings vacíos
    df = df.replace([" ", ""], pd.NA)

    # convertir columnas numéricas
    numeric_cols = ["monthly_charges", "total_charges", "tenure_months"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # imputación básica
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # normalizar categorías problemáticas
    if "multiple_lines" in df.columns:
        df["multiple_lines"] = df["multiple_lines"].replace(
            {"No phone service": "No"}
        )

    # crear nuevas features
    df["tenure_years"] = df["tenure_months"] / 12
    df["avg_monthly_spend"] = df["total_charges"] / df["tenure_months"].replace(0, 1)

    # eliminar columnas no útiles
    df = df.drop(columns=["customer_id"], errors="ignore")

    return df


def main():
    print("Descargando raw dataset...")
    df_raw = pd.read_csv(RAW_PATH)

    print(" Limpiando dataset...")
    df_clean = clean_telco_data(df_raw)

    # crear carpeta processed si no existe
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

    print(f" Guardando dataset procesado en: {PROCESSED_PATH}")
    df_clean.to_csv(PROCESSED_PATH, index=False)

    print(" Dataset limpio y guardado!")


if __name__ == "__main__":
    main()