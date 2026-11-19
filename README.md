# AutoScan – Plaka Bölgesi Tespit Sistemi

## Proje Hakkında

AutoScan, araç fotoğraflarındaki plaka bölgelerini görüntü işleme teknikleriyle tespit etmeyi amaçlayan bir Python projesidir.

Sistem plaka karakterlerini okumaz (OCR yapmaz). Yalnızca plakanın bulunduğu bölgeyi tespit eder, araç fotoğrafı üzerinde işaretler ve ayrı bir görüntü olarak kırpar.

## Kullanılan Teknolojiler

- Python
- OpenCV
- NumPy
- Matplotlib
- Google Colab

## Kullanılan Görüntü İşleme Teknikleri

- Gri tonlama
- Blackhat işlemi
- Sobel ve Canny kenar tespiti
- Otsu ve adaptif eşikleme
- HSV renk maskeleme
- Morfolojik işlemler
- Kontur analizi
- Perspektif düzeltme ve kırpma

## Proje Dosyaları

- `main.py`: Python kaynak kodu.
- `AutoScan.ipynb`: Google Colab/Jupyter Notebook sürümü.
- `images.zip`: Projede kullanılan araç fotoğrafları.

## Google Colab Üzerinde Çalıştırma

1. `AutoScan.ipynb` dosyasını Google Colab ile açın.
2. Notebook hücrelerini sırayla çalıştırın veya **Çalışma zamanı → Tümünü çalıştır** seçeneğini kullanın.
3. Fotoğraf yükleme penceresi açıldığında `images.zip` dosyasını seçin.
4. ZIP arşivi otomatik olarak açılır ve fotoğraflar işlenir.
5. Sonuçlar notebook çıktısında görüntülenir ve `sonuclar` klasörüne kaydedilir.

**Not:** ZIP dosyasının içinde `images` adlı bir klasör bulunmalıdır.

## Çıktılar

- `*_tespit.png`: Plaka bölgesinin araç fotoğrafı üzerinde çerçeveyle işaretlenmiş hâli.
- `*_plaka.png`: Tespit edilen plaka bölgesinin perspektif düzeltilmiş ve kırpılmış hâli.
- `*_bulunamadi.png`: Plaka bölgesi tespit edilemeyen fotoğraf için oluşturulan çıktı.

## Projenin Amacı

Bu proje, görüntü işleme teknikleri kullanılarak araç fotoğraflarında plaka bölgesinin tespit edilmesini amaçlar. Plaka karakterlerini tanıma veya metne dönüştürme işlemi yapmaz.
