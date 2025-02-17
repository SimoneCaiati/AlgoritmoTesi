import os
import pandas as pd 
import numpy as np
class DirManager:
    def __init__(self, base_dir, file_index, specificPD, test):
        self.specificPD=specificPD
        if not test:
            self.mediaDir = os.path.join( "Media", str(file_index), specificPD)
        else:
            self.mediaDir = os.path.join( "MediaTest", str(file_index), specificPD)
        self.file_PD_Dir = os.path.join(base_dir, "PositionalDatas", str(file_index))
        self.fastFourierDir = os.path.join(self.mediaDir, "FastFourierPlots")

    def create_directories(self):
        os.makedirs(self.file_PD_Dir, exist_ok=True)  
        os.makedirs(self.mediaDir, exist_ok=True)
        if self.specificPD == "PD2":
            os.makedirs(self.fastFourierDir, exist_ok=True)

    def save_position_data(self, data, filename):
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.file_PD_Dir, filename + ".csv"), index=False)
 
def load_and_rename_csv(filepath, rename_dict, drop_cols=None, skiprows=None):
    """Carica un CSV, lo rinomina e rimuove colonne specificate. Se il file non esiste, crea un DataFrame vuoto con valori 0."""
    
    if not os.path.exists(filepath):
        print(f"File non trovato: {filepath}. Creazione di un DataFrame di default con zeri.")
        
        # Creazione di un DataFrame con solo le colonne richieste, tutte con valore 0
        empty_df = pd.DataFrame(columns=rename_dict.values())
        for col in rename_dict.values():
            empty_df[col] = 0  # Imposta tutte le colonne a 0
        return empty_df

    # Caricamento del file
    df = pd.read_csv(filepath, delimiter=',', na_values=[''], skiprows=skiprows)

    # Rimozione delle colonne specificate (se esistono)
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')

    # Rinomina delle colonne
    return df.rename(columns=rename_dict)
