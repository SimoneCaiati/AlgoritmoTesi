import sqlite3

# Connessione al database (crea il file se non esiste)
conn = sqlite3.connect("DBtesi.db")

# Creazione di un cursore per eseguire comandi SQL
cur = conn.cursor()

# Creazione Tabelle
cur.execute("""
    CREATE TABLE IF NOT EXISTS Utente (
        ID_Utente INTEGER PRIMARY KEY AUTOINCREMENT,
        Nome TEXT CHECK(length(Nome) <= 50),
        Cognome TEXT CHECK(length(Cognome) <= 50),    
        Sesso TEXT CHECK(Sesso IN('Maschio','Femmina'))
    )
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS AnalysisType (
        ID_Analisi INTEGER PRIMARY KEY AUTOINCREMENT,
        Nome TEXT CHECK(length(Nome) <= 50),
        Tipo TEXT CHECK(length(Tipo) <= 50),    
        Dispositivo TEXT CHECK(length(Dispositivo) <= 50)
    )
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS DigitalBiomarkerType (
        ID_DigitalBiomarkerType INTEGER PRIMARY KEY AUTOINCREMENT,
        Tipo TEXT CHECK(length(Tipo) <= 50),
        Unita_di_misura TEXT CHECK(length(Unita_di_misura) <= 50), 
        Descrizione TEXT
    )
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS Calendario (
        ID_Calendario INTEGER PRIMARY KEY AUTOINCREMENT,
        Giorno INTEGER,
        Mese INTEGER,
        Anno INTEGER
    )
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS Biomarker (
        ID_Biomarker INTEGER PRIMARY KEY AUTOINCREMENT,
        Valore INTEGER,
        ID_DigitalBiomarkerType INTEGER,
        ID_Analisi INTEGER,
        FOREIGN KEY (ID_DigitalBiomarkerType) REFERENCES DigitalBiomarkerType(ID_DigitalBiomarkerType),
        FOREIGN KEY (ID_Analisi) REFERENCES AnalysisType(ID_Analisi)
    )
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS File (
        ID_File INTEGER PRIMARY KEY AUTOINCREMENT,
        TipoAttivita TEXT CHECK(length(TipoAttivita) <= 50),
        NomeFile TEXT CHECK(length(NomeFile) <= 50),
        Orario TEXT CHECK(Orario LIKE '%%:%%:%%'),
        ID_Utente INTEGER,
        ID_Calendario INTEGER,
        FOREIGN KEY (ID_Utente) REFERENCES Utente(ID_Utente),   
        FOREIGN KEY (ID_Calendario) REFERENCES Calendario(ID_Calendario)
    )
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS Deriva (
        ID_File INTEGER,
        ID_Biomarker INTEGER,
        PRIMARY KEY (ID_File, ID_Biomarker),
        FOREIGN KEY (ID_File) REFERENCES File(ID_File),
        FOREIGN KEY (ID_Biomarker) REFERENCES Biomarker(ID_Biomarker)
    )
""")


# Confermare e chiudere la connessione
conn.commit()
conn.close()
print("Database SQLite Creato con successo!")
