import serial
import os
import pandas as pd
import numpy as np
from datetime import datetime

def read_arduino_data(serial_port='/dev/ttyUSB0', baud_rate=9600, timeout=2, output_dir='ArduinoData', file_name='sensor_data.csv'):
    """
    Legge i dati dalla porta seriale di Arduino e li salva in un file CSV.
    """
    
    # Creazione della cartella se non esiste
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_path = os.path.join(output_dir, file_name)
    
    # Apertura della connessione seriale
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=timeout)
        print(f"Connessione stabilita su {serial_port} a {baud_rate} baud")
    except serial.SerialException as e:
        print(f"Errore di connessione alla porta seriale: {e}")
        return
    
    columns = ["Timestamp", "Acc x (g)", "Acc y (g)", "Acc z (g)",
               "Gyro x (deg/s)", "Gyro y (deg/s)", "Gyro z (deg/s)",
               "Mag x (microTesla)", "Mag y (microTesla)", "Mag z (microTesla)",
               "Pressione (milliBar)", "Altitudine (m)"]
    
    data_list = []
    try:
        while True:
            line = ser.readline().decode('utf-8').strip()
            if line:
                try:
                    values = line.split(',')
                    if len(values) == len(columns) - 1:  # -1 perché Timestamp viene aggiunto
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                        row = [timestamp] + [float(v) for v in values]
                        data_list.append(row)
                        print(row)
                except ValueError as ve:
                    print(f"Errore nella conversione dei dati: {ve}")
    except KeyboardInterrupt:
        print("Interruzione manuale. Salvando i dati...")
    
    ser.close()
    df = pd.DataFrame(data_list, columns=columns)
    df.to_csv(file_path, index=False)
    print(f"Dati salvati in {file_path}")
    
    return df