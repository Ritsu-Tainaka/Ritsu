# Manga Çevirici (v0.8)

Windows için: manga PDF/PNG/JPG/CBZ sayfalarını İngilizce → Türkçe çevirir, orijinal yazıyı siler,
Türkçeyi balona sığdırıp orijinale yakın font stilinde yazar, bölüm başına PDF (+ isteğe bağlı CBZ) üretir.

## Nasıl çalışır
1. **Yerel OCR:** yazılar bilgisayarında okunur (RapidOCR; kurulu değilse Tesseract). API'ye istek gitmez.
2. **Toplu çeviri:** Gemini'ye sayfa görseli değil **yalnızca metin** gider; ~60 kutu tek istekte çevrilir.
   Bir bölüm genelde 1-3 istek tutar.
3. **Silme + dizgi:** orijinal yazı silinir (OpenCV ya da Manga-LaMa), Türkçe balonun iç boşluğuna yerleştirilir.

## Kurulum
1. **Python 3.12** kur (3.10–3.12 çalışır; 3.13 ve üstü desteklenmez). Kurulumda "Add python to PATH" işaretli olsun.
2. `run.bat` çift tıkla (ilk açılışta gerekli paketleri ve RapidOCR'ı kendisi kurar).
3. Menü > Ayarlar: Gemini API anahtarı gir (https://aistudio.google.com/apikey). Model alanına
   "Modelleri Listele" ile güncel bir model seç (varsayılan: `gemini-flash-lite-latest`).
4. Yeni proje > "Klasör tara" > ana manga klasörünü seç > sıralamayı kontrol et > "Çeviriyi başlat".
5. EXE istersen `build_exe.bat`.

## v0.8'de neler değişti
- Yalnızca yerel OCR + metin çevirisi modu kaldı; Gemini'nin sayfa görselini okuduğu mod kaldırıldı.
- Colab ve GPU (Qwen, Surya, Big-LaMa, FLUX, MAGIv2) desteği kaldırıldı. ML balon dedektörü kaldırıldı;
  balonlar OpenCV ile bulunur. Yerel modeller: **RapidOCR** ve **Manga-LaMa**.
- **Çevrilemeyen metinler artık kaybolmuyor:** Gemini bir metni yanıtta atlarsa bir kez daha istenir; yine gelmezse
  balon orijinal kalır, bölüm "kısmi (N balon çevrilemedi)" görünür, sonraki "Çeviriyi başlat"ta yeniden denenir ve
  "Metinler" ekranında **ÇEVRİLEMEDİ** olarak listelenir.
- **Kutuları hizala** (Ayarlar > OCR): OCR kutusu ve silme alanı sayfadaki gerçek yazıya oturtulur. Sayfa düzenleyicide
  bir kutuyu taşıyınca kutu altındaki yazıya hizalanır; boyutlandırınca yalnızca silme alanı hizalanır.
- Sayfa düzenleyici: "Orijinal" metin düzenlenebilir; **"Seçili kutuyu yeniden çevir"** tek kutuyu küçük bir istekle çevirir.
  "Sayfayı yeniden tara ve çevir" sayfayı baştan OCR'lar.
- Sözlükten yalnızca o paketteki metinlerde geçen terimler gönderilir (daha az token).
- Renkli sayfalar şimdilik çevrilmez; günlükte "renkli sayfa" diye belirtilir ve PDF'e orijinal haliyle konur.

## Özellikler
- Balon temizleme: düz balonda renkle doldurma, desenli zeminde inpainting; balon çerçevesi korunur.
- Türkçe yerleştirme: balonun gerçek iç boşluğunu bulur, satırları dengeler, sığana kadar küçültür; İ/ı büyük harf kuralı doğru.
  Uzun kelimeler dar balonda heceden bölünür (HA-NIM-CI-ĞIM).
- Font eşleme: 7 stil × OFL lisanslı Türkçe karakterli font; Ayarlar'dan kendi fontunu bağlayabilirsin.
- Sayfa düzeltme ekranı: kutuyu sürükle/boyutlandır, çeviriyi/fontu elle değiştir, kutu ekle/sil, yeniden çiz.
- **Metinleri gözden geçir:** bölümün tüm orijinal/Türkçe metinleri tek tabloda; düzelt, "çevirme" işaretle,
  toplu bul/değiştir, sonra API'siz yeniden çiz. "Yalnızca sorunlular" ile sığmayan, boş ve çevrilemeyenleri süz.
- **"Yeniden çiz (kota harcamaz)":** seçili bölümleri kayıtlı çevirilerle yeniden çizer.
- Sayfa bazlı atomik kayıt, kaldığı yerden devam; API hatalarında bekleyip yeniden deneme (429/5xx), dakikalık hız sınırı.
- Proje başına SQLite (bölümler, manuel numara düzeltmeleri, sözlük). Gemini yeni terimleri "otomatik" önerir.
- API anahtarı Windows Kimlik Bilgileri Yöneticisi'nde (keyring) saklanır.

## Yazı silme
- **OpenCV:** hızlı, ek indirme yok; düz beyaz balonlarda yeterli.
- **Manga-LaMa:** desenli zemindeki yazıyı iz bırakmadan siler. Ayarlar > Yazı silme > Manga-LaMa, sonra
  "Seçili modeli indir" (~207 MB, Hugging Face mayocream/lama-manga-onnx) ve "Test et". CPU'da kutu başına ~1-3 sn.
  Model yoksa ya da hata verirse OpenCV kullanılır.

## OCR
- **RapidOCR (PaddleOCR modelleri):** önerilen; `run.bat` otomatik kurar (~15 MB).
- **Tesseract:** yedek. Kurulum: https://github.com/UB-Mannheim/tesseract/wiki
- Panele yapışık (açık) balonlar ve dikey yazılar sınırlı desteklenir; sorunlu sayfalar için sayfa düzenleme ekranını kullan.

## Bilinen sınırlar
- Renkli sayfalar şimdilik çevrilmez.
- Dikey yazı ve Japonca/Çince kaynak desteklenmez (yalnızca İngilizce kaynak).
- Bulut çeviri: metinler Google'a gönderilir (sayfa görselleri gönderilmez).

## Ücretsiz API kotası
Asıl sınır günlük istek sayısıdır (AI Studio'daki kota ekranından bak). Günlük kota dolarsa çeviri durur;
ertesi gün "Çeviriyi başlat" ile kaldığı yerden devam eder.

## Kaynak formatları
Klasör (PNG/JPG/WEBP), **PDF** ve **CBZ/ZIP** desteklenir. RAR tabanlı **.cbr** desteklenmez (7-Zip ile .cbz'ye çevir).

## Sürüm sabitleme (EXE / "numpy C-extensions failed" hatası)
requirements.txt eski/orta PC'lerde ve PyInstaller'da sorunsuz çalışan sürümlere sabitlendi.
Eski `.venv` klasörünü silip `build_exe.bat`'ı yeniden çalıştır.
Arayüzde beklenmeyen bir hata olursa pencerede gösterilir ve `%APPDATA%\MangaTranslator\hata.log` dosyasına yazılır.
