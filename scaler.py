import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os

def load_data(filepath):
    """
    Carrega os dados do arquivo CSV
    
    Parameters:
    -----------
    filepath : str
        Caminho para o arquivo CSV
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame com os dados carregados
    """
    return pd.read_csv(filepath)

def scale_data(df):
    """
    Escala as features usando StandardScaler
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame com as colunas originais
        
    Returns:
    --------
    tuple
        DataFrame escalado e objeto scaler ajustado
    """
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit and transform the data
    columns_to_scale = ["cost", "productivity", "area", "value"]
    scaled_columns = ["cost", "productivity", "area", "value"]
    
    # Fit and transform the data
    scaled_data = scaler.fit_transform(df[columns_to_scale])
    
    # Add scaled columns to dataframe
    for i, col in enumerate(scaled_columns):
        df[col] = scaled_data[:, i]
    
    # Salva o scaler treinado
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/standard_scaler.joblib')
    
    return df, scaler 

if __name__ == "__main__":
    # Carrega os dados
    df = load_data("1_data_to_scale.csv")
    
    # Escala os dados e obtém o scaler
    scaled_df, scaler = scale_data(df)
    
    # Salva os dados escalados em CSV
    scaled_df.to_csv('2_scaled_data.csv', index=False)
    
    print("Dados escalados e salvos em '2_scaled_data.csv'")
    print("Scaler salvo em 'models/standard_scaler.joblib'") 