# 🩺 MediLlama: LLaMA 3.2 3B Medical Chatbot

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-red.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow.svg)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57-FF4B4B.svg)](https://streamlit.io/)
[![PEFT LoRA](https://img.shields.io/badge/PEFT-LoRA-orange.svg)](https://github.com/huggingface/peft)

Bu proje, LLaMA 3.2 3B Instruct modelinin QLoRA (4-bit nicelikleme) yöntemi ile tıbbi diyalog veri seti üzerinde ince ayar (Fine-Tuning) yapılmasını ve eğtilen modelin modern bir Streamlit arayüzü üzerinden test edilmesini sağlar.

Eğitim RTX 4060 Laptop GPU (8GB VRAM) gibi kısıtlı kaynaklara sahip ekran kartlarında çalışacak şekilde optimize edilmiştir.

---

## ✨ Özellikler

*   **Veri Seti**: Hugging Face üzerinde yer alan `ruslanmv/ai-medical-chatbot` veri setinin rastgele seçilmiş bir alt kümesiyle eğitilmiştir.
*   **Bellek Dostu (QLoRA)**: BitsAndBytes 4-bit nicelikleme, Gradient Checkpointing ve Paged 8-bit AdamW optimizasyonu sayesinde 8GB VRAM sınırını aşmadan eğitim tamamlanır.
*   **İnteraktif Terminal Sohbeti (`inference.py`)**: İnce ayardan önce ve sonra modelin üslup farklarını karşılaştırmalı olarak terminal üzerinden test etmenizi sağlar.
*   **Gelişmiş Streamlit Web Arayüzü (`app.py`)**:
    *   Kullanıcı dostu tıbbi asistan sohbet ekranı.
    *   **Yan Yana Karşılaştırma Modu**: Taban LLaMA modelinin yanıtı ile ince ayarlı modelin yanıtını aynı anda yan yana görebilme.
    *   VRAM tüketimini artırmadan `PeftModel.disable_adapter()` ile dinamik adaptör kontrolü.
    *   Sıcaklık (Temperature), Top-P ve Maksimum Token ayarlarını dinamik olarak değiştirme.

---

## 📁 Proje Yapısı

```text
├── medical_llama3_lora/     # İnce ayar sonucu kaydedilen LoRA ağırlıkları ve eğitim günlükleri
├── venv/                    # Python Sanal Ortamı (Git tarafından yok sayılır)
├── .gitignore               # Gereksiz dosyaları ve büyük ağırlıkları dışlayan Git ayarı
├── app.py                   # Streamlit Web Arayüzü
├── finetune.py              # Supervised Fine-Tuning (SFT) eğitim betiği
├── inference.py             # Karşılaştırmalı terminal test betiği
└── requirements.txt         # Gerekli kütüphaneler ve CUDA uyumlu PyTorch kurulum detayları
```

---

## 🚀 Hızlı Başlangıç

### 1. Kurulum ve Ortam Hazırlığı

Öncelikle projeyi klonlayıp ilgili klasöre geçin ve sanal ortamı kurun:

```powershell
# Sanal ortamı etkinleştirin
.\venv\Scripts\activate

# Windows Türkçe kodlama uyumsuzluğunu önlemek için UTF-8 modunu etkinleştirin (Kritik)
$env:PYTHONUTF8=1
```

Gerekli paketleri kurun:
```powershell
pip install -r requirements.txt
```

*Not: `requirements.txt` CUDA 12.4 uyumlu PyTorch tekerleğini doğrudan çekmek üzere yapılandırılmıştır.*

---

### 2. İnce Ayar (Fine-Tuning) Çalıştırma

Modeli eğitmek için aşağıdaki komutu kullanabilirsiniz. Parametreleri ekran kartınıza göre ayarlayabilirsiniz:

```powershell
python finetune.py --max_steps 100 --subset_size 1000
```

**Kullanılabilir Parametreler:**
*   `--subset_size`: Eğitimde kullanılacak veri adedi (Örn: 5000).
*   `--max_steps`: Eğitim adım sayısı.
*   `--batch_size`: Adım başına düşen örnek sayısı.
*   `--grad_accum_steps`: Gradyan biriktirme adımları (Büyük batch boyutlarını simüle eder).

---

### 3. Modeli Test Etme

#### A. Terminal Üzerinden Karşılaştırma ve Sohbet:
```powershell
python inference.py
```
Bu betik, taban model ile eğitilen LoRA adaptörlerini sırayla yükleyerek örnek sorulara verilen cevapları kıyaslar ve ardından interaktif bir diyalog penceresi açar.

#### B. Streamlit Web UI ile Görsel Arayüz:
```powershell
streamlit run app.py
```
Komut çalıştıktan sonra tarayıcınızda otomatik olarak `http://localhost:8501` adresi açılacaktır. Yan panelden ayarları değiştirerek modeli canlı olarak test edebilirsiniz.

---

## 📊 Örnek Karşılaştırma

| Hasta Sorusu | Orijinal LLaMA 3.2 | İnce Ayarlı Hekim Modeli |
| :--- | :--- | :--- |
| *My child has a mild fever (38C) and a runny nose. When should I contact a pediatrician?* | Detaylı, uzun maddeler halinde genel tavsiyeler ve çocuk hekimine gidilmesi gereken durum listesi sunar. | **"Hi, I have gone through your query... Regards, Dr. Sumanth"** formatında, doğrudan hekim üslubuyla ve yatıştırıcı bir dille yanıt verir. |

---

## ⚠️ Yasal Uyarı / Disclaimer

Bu proje sadece **eğitim ve araştırma** amacıyla geliştirilmiştir. Üretilen tıbbi yanıtlar klinik düzeyde doğrulanmamıştır ve kesinlikle **tıbbi tavsiye yerine geçmez**. Gerçek sağlık sorunlarınız için her zaman yetkili bir sağlık kuruluşuna başvurun.
