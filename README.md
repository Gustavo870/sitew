BIBLIOTECA 


pip install requests beautifulsoup4 pillow imagehash numpy tensorflow 




CODIGO : 



import os, time, requests, random
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import imagehash
import numpy as np
import tensorflow as tf
from shutil import copy2

def buscar_imagens_google(palavra, max_imgs=400):
    urls, page = set(), 0
    headers = {"User-Agent":"Mozilla/5.0"}
    while len(urls) < max_imgs:
        start = page * 20
        url = ("https://www.google.com/search?tbm=isch&q={}&start={}"
               .format(palavra, start))
        resp = requests.get(url, headers=headers)
        soup = BeautifulSoup(resp.text, "html.parser")
        for img in soup.select("img"):
            src = img.get("src")
            if src and src.startswith("http"):
                urls.add(src)
                if len(urls) >= max_imgs:
                    break
        page += 1
        time.sleep(random.uniform(1, 2))
        if page > 20:
            break
    return list(urls)

def baixar_unicas(urls, pasta, palavra):
    os.makedirs(pasta, exist_ok=True)
    hashes, count = set(), 0
    headers = {"User-Agent":"Mozilla/5.0"}
    for u in urls:
        try:
            r = requests.get(u, headers=headers, timeout=10, verify=False)
            ct = r.headers.get("Content-Type", "")
            if not ct.startswith("image"):
                continue
            img = Image.open(BytesIO(r.content)).convert("RGB")
            h = imagehash.average_hash(img)
            if h in hashes:
                continue
            hashes.add(h)
            path = os.path.join(pasta, f"{palavra}_{count}.jpg")
            img.save(path)
            print(f"[{count+1}] Baixado: {path}")
            count += 1
        except Exception as e:
            print("Erro:", e)
            continue
    return count

def carregar_modelo():
    print("Carregando MobileNetV2...")
    return tf.keras.applications.MobileNetV2(weights='imagenet')

def filtrar_plastico(model, pasta_in, pasta_out):
    os.makedirs(pasta_out, exist_ok=True)
    termos = ['plastic','bottle','container','jar','bag','wrap','packaging']
    for fname in os.listdir(pasta_in):
        path = os.path.join(pasta_in, fname)
        try:
            img = Image.open(path).resize((224,224))
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x,0)
            x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
            preds = model.predict(x)
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]
            classes = [c[1].lower() for c in decoded]
            if any(t in ' '.join(classes) for t in termos):
                copy2(path, os.path.join(pasta_out, fname))
                print("Filtrada:", fname)
        except Exception as e:
            print("Erro filtro:", e)

if __name__ == "__main__":
    palavra = "plastic"  # usar em inglês para melhores resultados
    pasta_bruta = "imagens/plastico_google"
    pasta_filtrada = "imagens/plastico_filtrado"

    print("Buscando URLs de imagens...")
    urls = buscar_imagens_google(palavra, max_imgs=400)
    print("Total URLs coletadas:", len(urls))

    print("Baixando imagens únicas...")
    baixar_unicas(urls, pasta_bruta, palavra)

    model = carregar_modelo()

    print("Filtrando imagens relacionadas a plástico...")
    filtrar_plastico(model, pasta_bruta, pasta_filtrada)

    print("Pronto! Imagens filtradas estão em:", pasta_filtrada)
