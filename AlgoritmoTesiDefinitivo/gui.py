import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import pandas as pd
import numpy as np

# Funzione per creare un rettangolo con angoli arrotondati
def create_rounded_rectangle(canvas, x1, y1, x2, y2, radius=20, **kwargs):
    points = [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)

# Funzione per caricare un file
def upload_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        print(f"File selezionato: {file_path}")

def main_interface(file_name, activity, date_info):
    # Caricamento da "DataBase"
    AllFileData = pd.read_csv("FileData.csv", delimiter=';', na_values=['']).replace(" ", "")
    FileDataSaved = pd.read_csv("FileDataSaved.csv", delimiter=';', na_values=['']).replace(" ", "")
    
    # DataFrame per i preferiti in memoria
    favorites_df = FileDataSaved.copy()

    # Funzione per aggiornare il contatore nel bottone "Attività Salvate"
    def update_favorites_count():
        favorites_button_canvas.itemconfig(favorites_counter_text_id, text=str(favorites_df.shape[0]))

    # Funzione per aggiornare la view nel Tab3 (Salvate)
    def update_favorites_view():
        # Svuota il frame interno di Tab3
        for widget in favorites_inner_frame.winfo_children():
            widget.destroy()
        # Per ogni record salvato, crea un frame simile a Tab2
        for idx, row in favorites_df.iterrows():
            fav_frame = tk.Frame(favorites_inner_frame, bg="#F1F1F1", bd=2, relief="groove", padx=10, pady=10)
            fav_frame.pack(fill="x", padx=10, pady=5)
            
            # Indicatori (puoi personalizzare la visualizzazione)
            indicators = [
                ("Numero Passi", row["Numero_Passi"], "feet"), 
                ("Velocità Media", row["Velocita_media"], "m/s"), 
                ("Tempo Impiegato", row["Tempo_impiegato"], "s")
            ]
            circle_frame = tk.Frame(fav_frame, bg="#F1F1F1")
            circle_frame.pack(side="top", fill="x")
            for i, (label_text, value, unit) in enumerate(indicators):
                canvas = tk.Canvas(circle_frame, width=100, height=100, bg="#F1F1F1", highlightthickness=0)
                canvas.grid(row=0, column=i, padx=20, pady=10)
                canvas.create_oval(10, 10, 90, 90, outline="#1DAA8D", width=3, fill="#D9EFE3")
                canvas.create_text(50, 40, text=str(value), font=("Arial", 12, "bold"), fill="black")
                canvas.create_text(50, 60, text=unit, font=("Arial", 10), fill="black")
                ttk.Label(circle_frame, text=label_text, font=("Arial", 9), foreground="#B5B5B5", background="#F1F1F1").grid(row=1, column=i, pady=5)
            
            # Dettagli File
            details_frame = tk.Frame(fav_frame, bg="#F1F1F1")
            details_frame.pack(fill="x", pady=10)
            tk.Label(details_frame, text="Nome del file:", font=("Arial", 10, "bold"), fg="black", bg="#F1F1F1").grid(row=0, column=0, sticky="w", padx=5)
            tk.Label(details_frame, text=row["Nome_File"], font=("Arial", 10), fg="black", bg="#F1F1F1").grid(row=0, column=1, sticky="w", padx=5)
            tk.Label(details_frame, text="Tipo attività:", font=("Arial", 10, "bold"), fg="black", bg="#F1F1F1").grid(row=1, column=0, sticky="w", padx=5)
            tk.Label(details_frame, text=row["Tipo_Attivita"], font=("Arial", 10), fg="black", bg="#F1F1F1").grid(row=1, column=1, sticky="w", padx=5)
            tk.Label(details_frame, text="Data:", font=("Arial", 10, "bold"), fg="black", bg="#F1F1F1").grid(row=2, column=0, sticky="w", padx=5)
            tk.Label(details_frame, text=row["Data"], font=("Arial", 10), fg="black", bg="#F1F1F1").grid(row=2, column=1, sticky="w", padx=5)
            
            # Pulsante per rimuovere il record dai preferiti
            tk.Button(details_frame, text="Rimuovi dai preferiti", 
                      command=lambda nome=row["Nome_File"]: remove_from_favorites(nome)
                     ).grid(row=0, column=2, rowspan=3, padx=10)
    
    # Funzione per aggiungere un record ai preferiti
    def add_to_favorites(record_data):
        nonlocal favorites_df
        if not favorites_df["Nome_File"].isin([record_data["Nome_File"]]).any():
            favorites_df = pd.concat([favorites_df, pd.DataFrame([record_data])], ignore_index=True)
            favorites_df.to_csv("FileDataSaved.csv", sep=";", index=False)
            update_favorites_view()
            update_favorites_count()  # Aggiorna il contatore
        else:
            print("Record già presente nei preferiti.")

    # Funzione per rimuovere un record dai preferiti
    def remove_from_favorites(nome_file):
        nonlocal favorites_df
        favorites_df = favorites_df[favorites_df["Nome_File"] != nome_file]
        favorites_df.to_csv("FileDataSaved.csv", sep=";", index=False)
        update_favorites_view()
        update_favorites_count()  # Aggiorna il contatore

    # Creazione della finestra principale
    root = tk.Tk()
    root.title("Dashboard")
    root.geometry("800x450")
    root.configure(bg="#F9F9F9")

    # Caricamento delle immagini
    icon_walk = ImageTk.PhotoImage(Image.open("InterfaceImages/walk_icon.png").resize((20, 20)))
    icon_bookmark = ImageTk.PhotoImage(Image.open("InterfaceImages/bookmark_icon.png").resize((20, 20)))
    avatar_image = ImageTk.PhotoImage(Image.open("InterfaceImages/avatar_icon.png").resize((80, 80)))

    # Frame Sidebar
    sidebar = tk.Frame(root, bg="#3DC9A7", width=200, height=450)
    sidebar.pack(side="left", fill="y")

    # Contenitore Avatar con sfondo bianco
    avatar_frame = tk.Frame(sidebar, bg="white", width=100, height=100)
    avatar_frame.pack(pady=15)
    avatar_frame.pack_propagate(False)

    # Avatar Utente
    avatar_label = tk.Label(avatar_frame, image=avatar_image, bg="white")
    avatar_label.pack(expand=True)

    # Nome e attività
    user_label = tk.Label(sidebar, text="Simone Caiati", font=("Arial", 12, "bold"), fg="white", bg="#3DC9A7")
    user_label.pack()
    activity_label = tk.Label(sidebar, text="Attività fisica", font=("Arial", 10), fg="white", bg="#3DC9A7")
    activity_label.pack()

    # Pannello attività
    activity_frame = tk.Frame(sidebar, bg="#3DC9A7")
    activity_frame.pack(pady=10)

    # Modifica di create_activity_button per restituire canvas e id del testo
    def create_activity_button(text, icon, num_rowFile):
        frame = tk.Frame(activity_frame, bg="#3DC9A7", width=180, height=50)
        frame.pack(pady=5)
    
        canvas = tk.Canvas(frame, width=180, height=50, bg="#3DC9A7", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        create_rounded_rectangle(canvas, 0, 0, 180, 50, radius=15, fill="#2CA78F", outline="#2CA78F")
    
        # Posizionamento icona e testo
        canvas.create_image(15, 25, image=icon, anchor="center")
        canvas.create_text(80, 25, text=text, font=("Arial", 10, "bold"), fill="white")
        count_text_id = canvas.create_text(150, 25, text=num_rowFile, font=("Arial", 10), fill="white")
        return canvas, count_text_id

    # Creazione dei pulsanti sulla sidebar
    create_activity_button("Numero Attività\nCaricate", icon_walk, AllFileData.shape[0])
    favorites_button_canvas, favorites_counter_text_id = create_activity_button("Attività Salvate", icon_bookmark, favorites_df.shape[0])

    # Sezione Analytics
    analytics_frame = tk.Frame(root, bg="#F9F9F9", padx=20, pady=20)
    analytics_frame.pack(fill="both", expand=True)

    analytics_label = tk.Label(analytics_frame, text="Analytics", font=("Arial", 14, "bold"), fg="black", bg="#F9F9F9")
    analytics_label.grid(row=0, column=0, sticky="w", pady=10)

    # Tabs
    notebook = ttk.Notebook(analytics_frame)
    tab1 = ttk.Frame(notebook)
    tab2 = ttk.Frame(notebook)
    tab3 = ttk.Frame(notebook)
    notebook.add(tab1, text="Oggi")
    notebook.add(tab2, text="Passate")
    notebook.add(tab3, text="Salvate")
    notebook.grid(row=1, column=0, columnspan=3, pady=10, sticky="w")

    # ---------------------- Tab1: Oggi ----------------------
    labels = ["Numero Passi", "Velocità Media", "Tempo Impiegato"]
    units = ["feet", "m/s", "s"]
    values_today = [0, 0, 0]  # Valori placeholder (da sostituire con dati reali se disponibili)

    for i in range(3):
        canvas = tk.Canvas(tab1, width=100, height=100, bg="#F1F1F1", highlightthickness=0)
        canvas.grid(row=2, column=i, padx=20, pady=10)
        canvas.create_oval(10, 10, 90, 90, outline="#1DAA8D", width=3, fill="#D9EFE3")
        canvas.create_text(50, 40, text=str(values_today[i]), font=("Arial", 12, "bold"), fill="black")
        canvas.create_text(50, 60, text=units[i], font=("Arial", 10), fill="black")
        ttk.Label(tab1, text=labels[i], font=("Arial", 9), foreground="#B5B5B5", background="#F1F1F1").grid(row=3, column=i, pady=5)

    # Dettagli File in Tab1
    file_details_frame = tk.Frame(tab1, bg="#F1F1F1")
    file_details_frame.grid(row=4, column=0, columnspan=3, pady=10, sticky="w")
    detail_labels = ["Nome del file:", "Tipo attività:", "Data:"]
    detail_values = [file_name, activity, date_info]

    for i in range(3):
        tk.Label(file_details_frame, text=detail_labels[i], font=("Arial", 10, "bold"), fg="black", bg="#F1F1F1").grid(row=i, column=0, sticky="w", padx=5)
        tk.Label(file_details_frame, text=detail_values[i], font=("Arial", 10), fg="black", bg="#F1F1F1").grid(row=i, column=1, sticky="w", padx=5)
    
    # Pulsante per salvare il record di oggi
    def save_today():
        record = {
            "Numero_Passi": values_today[0],
            "Velocita_media": values_today[1],
            "Tempo_impiegato": values_today[2],
            "Nome_File": file_name,
            "Tipo_Attivita": activity,
            "Data": date_info
        }
        add_to_favorites(record)
    
    tk.Button(file_details_frame, text="Aggiungi ai preferiti", command=save_today).grid(row=0, column=2, rowspan=3, padx=10)

    # ---------------------- Tab2: Passate ----------------------
    canvas_tab2 = tk.Canvas(tab2, bg="#F1F1F1", width="460")
    scrollbar_tab2 = tk.Scrollbar(tab2, orient="vertical", command=canvas_tab2.yview)
    canvas_tab2.configure(yscrollcommand=scrollbar_tab2.set)
    canvas_tab2.pack(side="left", fill="both", expand=True)
    scrollbar_tab2.pack(side="right", fill="y")

    inner_frame = tk.Frame(canvas_tab2, bg="#F1F1F1")
    canvas_tab2.create_window((0, 0), window=inner_frame, anchor="nw")
    def on_configure(event):
        canvas_tab2.configure(scrollregion=canvas_tab2.bbox("all"))
    inner_frame.bind("<Configure>", on_configure)

    for index, row in AllFileData.iterrows():
        file_frame = tk.Frame(inner_frame, bg="#F1F1F1", bd=2, relief="groove", padx=10, pady=10)
        file_frame.pack(fill="x", padx=10, pady=5)
        circle_frame = tk.Frame(file_frame, bg="#F1F1F1")
        circle_frame.pack(side="top", fill="x")
        indicators = [
            ("Numero Passi", row["Numero_Passi"], "feet"), 
            ("Velocità Media", row["Velocita_media"], "m/s"), 
            ("Tempo Impiegato", row["Tempo_impiegato"], "s")
        ]
        for i, (label_text, value, unit) in enumerate(indicators):
            canvas = tk.Canvas(circle_frame, width=100, height=100, bg="#F1F1F1", highlightthickness=0)
            canvas.grid(row=0, column=i, padx=20, pady=10)
            canvas.create_oval(10, 10, 90, 90, outline="#1DAA8D", width=3, fill="#D9EFE3")
            canvas.create_text(50, 40, text=str(value), font=("Arial", 12, "bold"), fill="black")
            canvas.create_text(50, 60, text=unit, font=("Arial", 10), fill="black")
            ttk.Label(circle_frame, text=label_text, font=("Arial", 9), foreground="#B5B5B5", background="#F1F1F1").grid(row=1, column=i, pady=5)
    
        details_frame = tk.Frame(file_frame, bg="#F1F1F1")
        details_frame.pack(fill="x", pady=10)
        tk.Label(details_frame, text="Nome del file:", font=("Arial", 10, "bold"), fg="black", bg="#F1F1F1").grid(row=0, column=0, sticky="w", padx=5)
        tk.Label(details_frame, text=row["Nome_File"], font=("Arial", 10), fg="black", bg="#F1F1F1").grid(row=0, column=1, sticky="w", padx=5)
        tk.Label(details_frame, text="Tipo attività:", font=("Arial", 10, "bold"), fg="black", bg="#F1F1F1").grid(row=1, column=0, sticky="w", padx=5)
        tk.Label(details_frame, text=row["Tipo_Attivita"], font=("Arial", 10), fg="black", bg="#F1F1F1").grid(row=1, column=1, sticky="w", padx=5)
        tk.Label(details_frame, text="Data:", font=("Arial", 10, "bold"), fg="black", bg="#F1F1F1").grid(row=2, column=0, sticky="w", padx=5)
        tk.Label(details_frame, text=row["Data"], font=("Arial", 10), fg="black", bg="#F1F1F1").grid(row=2, column=1, sticky="w", padx=5)
        tk.Button(details_frame, text="Aggiungi ai preferiti", 
                  command=lambda r=row: add_to_favorites(r.to_dict())).grid(row=0, column=2, rowspan=3, padx=10)

    tk.Button(analytics_frame, text="Upload file", command=upload_file).grid(row=5, column=0, pady=10, sticky="w")

    # ---------------------- Tab3: Salvate ----------------------
    canvas_tab3 = tk.Canvas(tab3, bg="#F1F1F1")
    scrollbar_tab3 = tk.Scrollbar(tab3, orient="vertical", command=canvas_tab3.yview)
    canvas_tab3.configure(yscrollcommand=scrollbar_tab3.set)
    canvas_tab3.pack(side="left", fill="both", expand=True)
    scrollbar_tab3.pack(side="right", fill="y")

    favorites_inner_frame = tk.Frame(canvas_tab3, bg="#F1F1F1")
    canvas_tab3.create_window((0, 0), window=favorites_inner_frame, anchor="nw")
    favorites_inner_frame.bind("<Configure>", lambda event: canvas_tab3.configure(scrollregion=canvas_tab3.bbox("all")))

    update_favorites_view()

    root.mainloop()

def initial_interface():
    root = tk.Tk()
    root.title("Inserisci informazioni")
    root.geometry("300x250")
    
    tk.Label(root, text="Nome del file:").pack(pady=(20, 5))
    entry_file = tk.Entry(root)
    entry_file.pack(pady=5)
    
    tk.Label(root, text="Tipo Attività:").pack(pady=5)
    entry_attivita = tk.Entry(root)
    entry_attivita.pack(pady=5)
    
    tk.Label(root, text="Data (gg/mm/aaaa):").pack(pady=5)
    entry_data = tk.Entry(root)
    entry_data.pack(pady=5)
    
    def submit_info():
        file_name = entry_file.get()
        activity = entry_attivita.get()
        date_info = entry_data.get()
        root.destroy()
        main_interface(file_name, activity, date_info)
    
    tk.Button(root, text="Submit", command=submit_info).pack(pady=10)
    
    root.mainloop()
    
initial_interface()
