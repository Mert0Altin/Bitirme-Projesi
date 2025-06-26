import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

# --- YARDIMCI FONKSİYONLAR ---

def psnr(orijinal_img, islenmis_img):
    """
    İki görüntü arasındaki Tepe Sinyal-Gürültü Oranını (PSNR) hesaplar.
    """
    mse = np.mean((orijinal_img - islenmis_img) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr_degeri = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr_degeri

def metin_to_bit(metin):
    """
    Verilen metni bit dizisine çevirir.
    """
    # UTF-8 encoding ile byte'lara dönüştür
    byte_array = metin.encode('utf-8')
    bits = []
    for byte in byte_array:
        bits.extend([int(bit) for bit in format(byte, '08b')])
    return bits

def bit_to_metin(bit_dizisi):
    """
    Verilen bit dizisini metne çevirir.
    """
    if len(bit_dizisi) % 8 != 0:
        # Eksik bitleri 0 ile tamamla
        padding = 8 - (len(bit_dizisi) % 8)
        bit_dizisi.extend([0] * padding)
        
    byte_dizisi = [bit_dizisi[i:i+8] for i in range(0, len(bit_dizisi), 8)]
    byte_array = []
    for byte in byte_dizisi:
        try:
            byte_array.append(int("".join(map(str, byte)), 2))
        except Exception:
            byte_array.append(63)  # '?' karakteri
    try:
        return bytes(byte_array).decode('utf-8')
    except Exception:
        return "".join([chr(b) if 32 <= b <= 126 else '?' for b in byte_array])


# --- İÇERİK SAHİBİ TARAFI ---

def median_edge_detector(a, b, c):
    """
    Standart Medyan Kenar Algılayıcı (MED) tahmincisi.
    Bu, PEE algoritmasının temel bir parçasıdır. [3]
    """
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
    if a > b:
        a, b = b, a
    # Overflow'u önlemek için int kullan
    return int(a) + int(c) - int(b)

def reserve_room_and_preprocess(image):
    """
    Şifrelemeden Önce Alan Ayırma (VRBE) işlemini Tahmin Hatası Genişletme (PEE) ile uygular.
    Bu fonksiyon, veri gizleyici için boş LSB'ler (en anlamsız bitler) oluşturur. [4, 5, 6]
    """
    h, w = image.shape
    preprocessed_image = np.copy(image)
    # Konum haritası, taşma/batma (overflow/underflow) olan pikselleri ve
    # veri gömülemeyen pikselleri (kenarlar gibi) işaretler.
    location_map = np.zeros((h, w), dtype=int)
    # Gömme haritası, veri gizleyicinin veri gömebileceği pikselleri işaretler.
    embed_map = np.zeros((h, w), dtype=int)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # Komşu pikselleri al
            N = int(image[i-1, j])
            W = int(image[i, j-1])
            NW = int(image[i-1, j-1])
            
            # Tahmin değerini hesapla
            predicted_value = median_edge_detector(N, W, NW)
            
            # Tahmin hatasını hesapla
            current_pixel = int(image[i, j])
            prediction_error = current_pixel - predicted_value
            
            # PEE ile hatayı genişlet. Burada gizli bit olarak '0' gömüyoruz
            # ki LSB'ler boşalsın.
            expanded_error = 2 * prediction_error + 0
            
            # Yeni piksel değerini hesapla
            new_pixel_value = predicted_value + expanded_error
            
            # Taşma/Batma kontrolü
            if new_pixel_value > 255 or new_pixel_value < 0:
                location_map[i, j] = 1 # Bu piksel değiştirilmedi
            else:
                preprocessed_image[i, j] = new_pixel_value
                embed_map[i, j] = 1 # Bu piksele veri gömülebilir

    return preprocessed_image, location_map, embed_map

def encrypt(image, key):
    """
    Görüntüyü bir anahtar kullanarak basit bir akış şifrelemesi (XOR) ile şifreler.
    LSB'leri koruyarak sadece üst 7 biti şifreler.
    """
    rng = np.random.RandomState(key)
    keystream = rng.randint(0, 256, size=image.shape, dtype=np.uint8)
    # LSB'leri koru, sadece üst 7 biti şifrele
    lsb_mask = image & 1  # LSB'leri al
    upper_bits = image & 0b11111110  # Üst 7 biti al
    encrypted_upper = np.bitwise_xor(upper_bits, keystream & 0b11111110)
    return encrypted_upper | lsb_mask


# --- VERİ GİZLEYİCİ TARAFI ---

def embed_data_with_pee(image, secret_data_bits, embed_map):
    """
    PEE tabanlı veri gömme - gömülen veri prediction error'un LSB'sine gömülür
    """
    marked_image = np.copy(image)
    h, w = image.shape
    data_idx = 0
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if embed_map[i, j] == 1 and data_idx < len(secret_data_bits):
                # Komşu pikselleri al
                N = int(marked_image[i-1, j])
                W = int(marked_image[i, j-1])
                NW = int(marked_image[i-1, j-1])
                
                # Tahmin değerini hesapla
                predicted_value = median_edge_detector(N, W, NW)
                
                # Mevcut prediction error'u hesapla
                current_pixel = int(marked_image[i, j])
                current_error = current_pixel - predicted_value
                
                # Gömülecek biti al
                secret_bit = secret_data_bits[data_idx]
                
                # PEE ile hatayı genişlet ve gizli biti göm
                expanded_error = 2 * current_error + secret_bit
                
                # Yeni piksel değerini hesapla
                new_pixel_value = predicted_value + expanded_error
                
                # Taşma/Batma kontrolü
                if new_pixel_value > 255 or new_pixel_value < 0:
                    # Taşma varsa, bu pikseli atla
                    continue
                else:
                    marked_image[i, j] = new_pixel_value
                    data_idx += 1
    
    if data_idx < len(secret_data_bits):
        print(f"UYARI: Verinin tamamı gömülemedi. Kapasite: {data_idx} bit, Veri boyutu: {len(secret_data_bits)} bit.")

    return marked_image

def extract_data_with_pee(image, embed_map):
    """
    PEE tabanlı veri çıkarma - gömülen veri prediction error'un LSB'sinden çıkarılır
    """
    extracted_bits = []
    h, w = image.shape
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if embed_map[i, j] == 1:
                # Komşu pikselleri al
                N = int(image[i-1, j])
                W = int(image[i, j-1])
                NW = int(image[i-1, j-1])
                
                # Tahmin değerini hesapla
                predicted_value = median_edge_detector(N, W, NW)
                
                # Mevcut prediction error'u hesapla
                current_pixel = int(image[i, j])
                expanded_error = current_pixel - predicted_value
                
                # Gömülü biti çıkar (expanded error'un LSB'si)
                secret_bit = expanded_error & 1
                extracted_bits.append(secret_bit)
    
    return extracted_bits

def decrypt(encrypted_image, key):
    """
    Şifre çözme işlemi, şifreleme ile aynıdır (XOR özelliği).
    """
    rng = np.random.RandomState(key)
    keystream = rng.randint(0, 256, size=encrypted_image.shape, dtype=np.uint8)
    # LSB'leri koru, sadece üst 7 biti çöz
    lsb_mask = encrypted_image & 1  # LSB'leri al
    upper_bits = encrypted_image & 0b11111110  # Üst 7 biti al
    decrypted_upper = np.bitwise_xor(upper_bits, keystream & 0b11111110)
    return decrypted_upper | lsb_mask

def recover_image(decrypted_marked_image, location_map, embed_map):
    """
    Alıcı, şifreleme anahtarını kullanarak görüntüyü çözer ve orijinal görüntüyü
    kayıpsız olarak kurtarır. Veri gizleyicinin gömdüğü veriyi bilmesine gerek yoktur.
    """
    recovered_image = np.copy(decrypted_marked_image)
    h, w = decrypted_marked_image.shape

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if location_map[i, j] == 1:
                # Bu piksel hiç değiştirilmemişti, olduğu gibi kalır.
                continue
            
            # Komşu pikselleri kurtarılmış görüntüden almalıyız
            N = int(recovered_image[i-1, j])
            W = int(recovered_image[i, j-1])
            NW = int(recovered_image[i-1, j-1])
            
            predicted_value = median_edge_detector(N, W, NW)
            
            # İşaretlenmiş piksel değerinden işaretlenmiş hatayı bul
            marked_pixel_value = int(decrypted_marked_image[i, j])
            expanded_error = marked_pixel_value - predicted_value
            
            # Gömülü veriyi çıkar (expanded error'un LSB'si)
            embedded_bit = expanded_error & 1
            
            # Orijinal hatayı bul (gömülü veriyi çıkararak)
            original_error = expanded_error // 2
            
            # Orijinal pikseli kurtar
            original_pixel_value = predicted_value + original_error
            
            # Değerin 0-255 aralığında olduğundan emin ol
            recovered_image[i, j] = np.clip(original_pixel_value, 0, 255)
            
    return recovered_image


# --- ANA UYGULAMA AKIŞI ---

if __name__ == "__main__":
    # 1. Hazırlık
    try:
        orijinal_resim_pil = Image.open("lena_gray.png").convert("L")
        orijinal_resim = np.array(orijinal_resim_pil, dtype=np.uint8)
    except FileNotFoundError:
        print("HATA: 'lena_gray.png' dosyası bulunamadı. Lütfen standart Lena test görüntüsünü bu isimle kodun bulunduğu dizine kaydedin.")
        exit()

    sifreleme_anahtari = 12345
    veri_gizleme_anahtari = 67890 
    
    gizli_metin = "Bu, şifreli görüntüye gizlenmiş bir test mesajıdır. Bitirme tezi için !"
    gizli_veri_bitleri = metin_to_bit(gizli_metin)

    print("--- RDH-EI SİMÜLASYONU BAŞLADI ---")
    print(f"Orijinal Görüntü Boyutu: {orijinal_resim.shape}")
    print(f"Gizlenecek Metin: '{gizli_metin}'")
    print(f"Gizlenecek Bit Sayısı: {len(gizli_veri_bitleri)}")

    # 2. Alan ayırma ve veri gömme (şifresiz)
    print("\n1. Alan ayırma ve veri gömme işlemi...")
    on_islenmis_resim, konum_haritasi, gomme_haritasi = reserve_room_and_preprocess(orijinal_resim)
    kapasite = int(np.sum(gomme_haritasi))
    print(f"Veri gömmek için kullanılabilir kapasite: {kapasite} bit")

    # Gömülecek bitleri kapasiteyle sınırla
    if len(gizli_veri_bitleri) > kapasite:
        print(f"UYARI: Gömülecek veri kapasiteden büyük! Sadece ilk {kapasite} bit gömülecek.")
    gizli_veri_bitleri = gizli_veri_bitleri[:kapasite]

    # PEE tabanlı veri gömme
    veri_gomulu_resim = embed_data_with_pee(on_islenmis_resim, gizli_veri_bitleri, gomme_haritasi)

    # 3. Şifreleme
    print("2. Veri gömülü görüntü şifreleniyor...")
    sifreli_resim = encrypt(veri_gomulu_resim, sifreleme_anahtari)

    # 4. Alıcı: İki farklı senaryo
    print("\n[ALICI] İşlemler gerçekleştiriliyor...")

    # Senaryo A: Şifre çözülüp veri çıkarılıyor
    print("  Senaryo A: Şifre çözülüp veri çıkarılıyor...")
    cozulmus_resim = decrypt(sifreli_resim, sifreleme_anahtari)
    cikarilmis_bitler = extract_data_with_pee(cozulmus_resim, gomme_haritasi)
    cikarilmis_metin = bit_to_metin(cikarilmis_bitler[:len(gizli_veri_bitleri)])
    
    print(f"    Çıkarılan Metin: '{cikarilmis_metin.strip()}'")
    if cikarilmis_metin.strip() == gizli_metin.strip():
        print("    ✓ BAŞARILI: Veri başarıyla ve hatasız olarak çıkarıldı!")
    else:
        print("    ✗ HATA: Veri çıkarma işleminde sorun oluştu!")

    # Senaryo B: Sadece şifre çözülüp görüntü kurtarılıyor
    print("\n  Senaryo B: Sadece şifre çözülüp görüntü kurtarılıyor...")
    kurtarilmis_resim = recover_image(cozulmus_resim, konum_haritasi, gomme_haritasi)
    
    psnr_degeri = psnr(orijinal_resim, kurtarilmis_resim)
    print(f"    Orijinal ve Kurtarılmış Görüntü Arasındaki PSNR: {psnr_degeri:.2f} dB")
    
    if np.array_equal(orijinal_resim, kurtarilmis_resim):
        print("    ✓ BAŞARILI: Orijinal görüntü başarıyla ve kayıpsız olarak kurtarıldı!")
    
    # 5. Görsel Çıktıları Oluşturma (Tez için)
    print("\n3. Görsel çıktılar oluşturuluyor...")
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('RDH-EI Süreç Adımları ve Sonuçları', fontsize=16)

    axs[0, 0].imshow(orijinal_resim, cmap='gray')
    axs[0, 0].set_title('1. Orijinal Görüntü')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(on_islenmis_resim, cmap='gray')
    axs[0, 1].set_title('2. Alan Ayrılmış Görüntü (PEE)')
    axs[0, 1].axis('off')

    axs[0, 2].imshow(veri_gomulu_resim, cmap='gray')
    axs[0, 2].set_title('3. Veri Gömülü Görüntü')
    axs[0, 2].axis('off')

    axs[1, 0].imshow(sifreli_resim, cmap='gray')
    axs[1, 0].set_title('4. Şifreli + Veri Gömülü Görüntü')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(kurtarilmis_resim, cmap='gray')
    axs[1, 1].set_title(f'5. Kurtarılmış Orijinal Görüntü\n(PSNR: {psnr_degeri:.2f} dB)')
    axs[1, 1].axis('off')
    
    fark_resmi = np.abs(orijinal_resim.astype(float) - kurtarilmis_resim.astype(float))
    axs[1, 2].imshow(fark_resmi, cmap='gray')
    axs[1, 2].set_title(f'6. Fark Görüntüsü (Hata: {np.sum(fark_resmi)})')
    axs[1, 2].axis('off')

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()
    
    print("\n--- SİMÜLASYON TAMAMLANDI ---")