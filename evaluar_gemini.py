import os
import time
import pandas as pd
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv

# 1. Configuración inicial
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print(" Error: GEMINI_API_KEY no encontrada en el archivo .env")
    exit()

genai.configure(api_key=api_key)
AUDIO_DIR = Path("audios_yucatan")
OUTPUT_CSV = "resultados_gemini.csv"

# Usamos Flash porque es rapidísimo y excelente para estas tareas
MODEL_NAME = "gemini-1.5-flash" 

def main():
    print(f" Iniciando Evaluación con {MODEL_NAME}...\n")
    
    archivos = [f for f in AUDIO_DIR.iterdir() if f.suffix.lower() in ['.mp3', '.wav', '.m4a', '.mpeg', '.mpg']]
    resultados = []

    for archivo in archivos:
        print(f" Procesando: {archivo.name} ...")
        start = time.time()
        
        try:
            # Subir el archivo de audio a los servidores temporales de Google
            audio_file = genai.upload_file(path=str(archivo))
            
            # Inicializar el modelo
            model = genai.GenerativeModel(MODEL_NAME)
            
            # El PROMPT clave para el modelo
            prompt = """
            Eres un experto lingüista especializado en el español de la Península de Yucatán y la lengua maya. 
            Escucha este audio cuidadosamente. 
            Transcribe de la forma más exacta posible lo que dice la persona. Si utiliza palabras en maya mezcladas con español, escríbelas correctamente.
            Devuelve ÚNICAMENTE la transcripción, sin saludos ni explicaciones.
            """
            
            # Generar respuesta
            response = model.generate_content([prompt, audio_file])
            texto = response.text.strip()
            estado = "Éxito"
            
            # Limpiar el archivo del servidor de Google
            genai.delete_file(audio_file.name)
            
            tiempo = time.time() - start
            print(f" Listo ({tiempo:.1f}s): {texto[:60]}...")
            
        except Exception as e:
            texto = str(e)
            estado = "Error"
            print(f" Falló {archivo.name}: {e}")
            
        resultados.append({
            "Nombre_Archivo": archivo.name,
            "Transcripcion_Gemini": texto,
            "Estado": estado
        })

    # Guardar resultados
    df = pd.DataFrame(resultados)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n ¡Proceso terminado! Resultados guardados en: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
