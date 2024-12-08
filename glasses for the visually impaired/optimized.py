from difflib import SequenceMatcher
import cv2
import pytesseract
from PIL import Image
import re
import pygame
from gtts import gTTS
import os
import time
from datetime import datetime
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.models import load_model
import speech_recognition as sr
import sys
import json
import pickle

# Daha önceden eğittiğimiz modeli yükledik
model = load_model('trained_model.h5')

nltk.download('punkt_tab')

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

readed = ""  # Global birikmiş metin değişkeni
previous_text = ""  # Önceki algılanan metin
last_processed_time = 0  # Son işlenme zamanı

# Varsayılan hız ayarı ve ayar dosyası
settings_file = "settings.json"
default_settings = {
    "speed": "normal"  # Yavaş, Normal, Hızlı
}


# Ayarları yükle veya varsayılan ayarları oluştur
def load_settings():
    try:
        with open(settings_file, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        save_settings(default_settings)
        return default_settings


def save_settings(settings):
    with open(settings_file, "w", encoding="utf-8") as file:
        json.dump(settings, file, ensure_ascii=False, indent=4)


# Global ayarlar
settings = load_settings()


# Ses hızını ayarlama
def adjust_voice_speed(speed):
    if speed in ["yavaş", "normal", "hızlı"]:
        settings["speed"] = speed
        save_settings(settings)
        print(f"Ses hızı {speed} olarak ayarlandı.")
        read_text(f"Ses hızı {speed} olarak ayarlandı.")
    else:
        print("Geçersiz hız seçeneği. Lütfen yavaş, normal veya hızlı olarak bir değer girin.")
        read_text("Geçersiz hız seçeneği. Lütfen yavaş, normal veya hızlı olarak bir değer girin.")


# Ayarlar menüsü
def settings_menu():
    """Ayarlar menüsü."""
    print("Ayarlar Menüsü")
    read_text("Ayarlar menüsüne hoş geldiniz. Burada ses hızını ayarlayabilirsiniz.")

    while True:
        read_text("Ses hızını ayarlamak için yavaş, normal veya hızlı deyin.")
        speed = speech_to_text()
        if speed:
            adjust_voice_speed(speed.lower())
            break

    read_text("Ses hızı başarıyla kaydedildi.")

def speech_to_text():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Lütfen konuşun (5 saniye içinde):")
        try:
            # Mikrofonla sesi dinle (5 saniye içinde)
            audio_data = recognizer.listen(source, timeout=5)
            print("Ses algılandı, dönüştürülüyor...")

            # Google Web Speech API ile metne çevir
            text = recognizer.recognize_google(audio_data, language="tr-TR")
            print(f"Algılanan Metin: {text}")
            return text

        except sr.WaitTimeoutError:
            print("Hiç ses algılanmadı. Süre doldu.")
            return None
        except sr.UnknownValueError:
            print("Ses tanımlanamadı.")
            return None
        except sr.RequestError as e:
            print(f"API erişim hatası: {e}")
            return None

def process_command(command):
    """
    Gelen komuta göre işlem yapar.
    'ayarlar' komutunda ayarları çağırır.
    'çık' komutunda programı sonlandırır.
    """
    if command.lower() == "ayarlar":
        print("Ayarlar menüsüne yönlendiriliyorsunuz...")
        settings_menu()
    elif command.lower() == "çık":
        print("Program sonlandırılıyor...")
        read_text("Program sonlandırılıyor")
        sys.exit(0)  # Programı sonlandırır
    else:
        print(f"Bilinmeyen komut: {command}")


def read_text(text, lang="tr"):
    """Algılanan metni seslendirme."""
    try:
        temp_filename = "temp.mp3"

        # Global hız ayarını kontrol et
        speed = settings["speed"]
        slow = True if speed == "yavaş" else False

        # Metni ses dosyasına dönüştür
        tts = gTTS(text=text, lang=lang, slow=slow)
        tts.save(temp_filename)

        # Pygame ile ses dosyasını çal
        pygame.mixer.init()
        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        pygame.mixer.quit()
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    except Exception as e:
        print(f"Hata: {e}")


def extract_text(image):
    """Görüntüden metin çıkarma."""
    try:
        text = pytesseract.image_to_string(image, lang="tur")
        return text.strip()
    except Exception as e:
        print(f"Metin çıkarma hatası: {e}")
        return ""


def is_image_blurry(image, threshold=100):
    """Görüntü bulanıklık kontrolü."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold


def is_text_corrupt(text, corruption_threshold=0.1):
    """Metnin bozuk olup olmadığını kontrol eder."""
    clean_text = re.sub(r'[^a-zA-ZığüşöçİĞÜŞÖÇ0-9]', ' ', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    words = clean_text.split()
    ww = sum(not word.isalpha() and not word.isdigit() for word in words)
    return ww > len(words) * corruption_threshold


def is_text_similar(new_text, old_text, threshold=0.8):
    """Yeni metnin eski metne benzerliğini kontrol eder."""
    similarity = SequenceMatcher(None, new_text, old_text).ratio()
    return similarity > threshold


def add_to_readed(text):
    """Okunan metni birleştirir."""
    global readed
    if readed:
        readed += " "
    readed += text


def log_detected_text(text):
    """Algılanan metni log dosyasına kaydeder."""
    with open("detected_text.log", "a", encoding="utf-8") as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] {text}\n")

def summarize_text(text, percentage= 0.6):
    """Metni özetler ve özetlenmiş metni döndürür.
    Cümlelerin toplam sayısının belirli bir yüzdesi kadar özet oluşturur.

    Args:
        text (str): Özetlenecek metin.
        percentage (float): Özetin uzunluğu (varsayılan %40).

    Returns:
        str: Özetlenmiş metin.
    """
    try:
        # Metindeki toplam cümle sayısını belirle
        sentences = text.split('.')  # Nokta ile cümleleri ayır
        sentences = [s.strip() for s in sentences if s.strip()]  # Boş cümleleri çıkar
        total_sentences = len(sentences)

        if total_sentences == 0:
            return "Özetlenebilecek bir metin bulunamadı."

        # Özetlenecek cümle sayısını belirle (%40)
        summary_sentence_count = max(1, int(total_sentences * percentage))  # En az 1 cümle

        # Sumy ile özetleme işlemi
        parser = PlaintextParser.from_string(text, Tokenizer("turkish"))
        summarizer = LsaSummarizer()
        summarized_sentences = summarizer(parser.document, summary_sentence_count)

        # Özetlenmiş cümleleri birleştir
        summary = " ".join(str(sentence) for sentence in summarized_sentences)
        return summary
    except Exception as e:
        print(f"Özetleme hatası: {e}")
        return "Özet oluşturulamadı."

def is_meaningful(text, model, tokenizer, threshold = 0.5):
    '''Metnin anlamlı olup olmadığını kontrol eder.

    Args:
        text (str) = Mainden gelen detected text
        model = Daha önceden eğittiğimiz model
        tokenizer = Daha önceden eğittiğimiz modelin tokenizer verileri
        threshold = Eşik değer

    returns:
        True ya da False değeri döndürür
    '''

    text_seq = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_seq, maxlen= 100)
    prediction = model.predict(text_pad)[0][0]
    return prediction >= threshold


def display_menu():
    """Programın giriş menüsünü görüntüler ve sesli olarak okur."""
    menu_text = """
    Hoş Geldiniz! Bu program bir görüntüden metin algılama ve seslendirme uygulamasıdır.

    Programın Özellikleri:
    - 'X' tuşuna basarak kameradan alınan görüntüyü işleyebilirsiniz.
    - Metin algılanırken, metnin bulanık, bozuk veya yamuk olmamasına dikkat ediniz.
    - Program geliştirme aşamasında olduğu için hatalarla karşılaşabilirsiniz.
    - Metin okunduğu sırada sesli komutlar algılanmaz. Metnin okunması tamamlanana kadar beklenmelidir.
    - Her metin okumasının ardından program size komut girmeniz için 5 saniye süre tanır. 
      Bu süre içinde bir komut algılanmazsa, 'X' tuşuna basana kadar bekler.

    Ek Bilgi:
    - 'Q' tuşuna basarak programdan çıkabilirsiniz.

    İyi çalışmalar dileriz!
    """
    print(menu_text)
    read_text(menu_text, lang="tr")

def main():
    global previous_text, last_processed_time, last_error

    # Giriş menüsünü göster ve sesli bir biçimde oku
    display_menu()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        read_text("Kamera açılamadı!")
        return

    last_error = None
    while True:
        ret, frame = cap.read()
        if not ret:
            read_text("Kamera verisi alınamıyor!")
            break

        # Kamera görüntüsünü sürekli göster
        cv2.imshow("Kamera", frame)

        # x tuşuna basıldığında görüntüyü işle şeklinde değiştirildi
        if cv2.waitKey(1) & 0xFF == ord('x'):

            '''
            elapsed_process_time = time.time() - last_processed_time
            if elapsed_process_time < 10:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            '''

            # Görüntü bulanıklık kontrolü
            if is_image_blurry(frame):
                if last_error != "Görüntü bulanık, lütfen kamerayı sabitleyin.":
                    read_text("Görüntü bulanık, lütfen kamerayı sabitleyin.")
                    last_error = "Görüntü bulanık, lütfen kamerayı sabitleyin."
                continue

            # Metin algılama
            detected_text = extract_text(frame)

            if not detected_text:
                if last_error != "Hiç metin algılanmadı.":
                    read_text("Hiç metin algılanmadı.")
                    last_error = "Hiç metin algılanmadı."
                continue

            if is_text_corrupt(detected_text):
                if last_error != "Metin bozuk algılandı, lütfen tekrar deneyin.":
                    read_text("Metin bozuk algılandı, lütfen tekrar deneyin.")
                    last_error = "Metin bozuk algılandı, lütfen tekrar deneyin."
                continue

            ''' 
            with open("tokenizer.pickle", "rb") as handle:
                tokenizer = pickle.load(handle)
                if not is_meaningful(detected_text, model, tokenizer):
                    if last_error != "Anlamlı bir metin değil":
                        read_text("Anlamlı bir metin değil")
                        last_error = "Anlamlı bir metin değil"
                    continue
            '''

            if is_text_similar(detected_text, previous_text):
                if last_error != "Benzer metin algılandı, bekleniyor...":
                    read_text("Benzer metin algılandı, bekleniyor...")
                    last_error = "Benzer metin algılandı, bekleniyor..."
                continue

            # Yeni metin işlemleri
            print(f"Algılanan metin: {detected_text}")
            previous_text = detected_text
            add_to_readed(detected_text)
            read_text(detected_text, lang="tr")

            # Metni log dosyasına kaydet
            log_detected_text(detected_text)

            last_error = None
            #last_processed_time = time.time()

            # 5 saniye bekleyip komut bekleme
            print("Komut bekleniyor (5 saniye)...")
            command = speech_to_text()
            if command:
                process_command(command)
            else:
                print("Komut algılanmadı, 'X' tuşuna basana kadar bekleniyor...")

        # Çıkış kontrolü
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
