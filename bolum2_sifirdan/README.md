# Bölüm 2: ham tahtadan öğrenme

Bölüm 1'de ajan, elle tasarladığımız 17 özelliği görüyordu (uzaklıklar, boş alan, kuyruğa yol...).
Burada hiçbiri yok: ajan tahtayı 3x15x15'lik bir görüntü olarak alır (baş, gövde, yem) ve bir
evrişimli sinir ağı (CNN) ile Double DQN kullanarak oynamayı sıfırdan öğrenmeye çalışır.

Dosyalar: `snake_env.py` (motor), `agent.py` (CNN + DQN), `train.py`, `evaluate.py`, `play.py`, `render.py`.
Colab için: `colab_egitim.ipynb`.

## Colab'de eğitim
1. colab.research.google.com'da **Dosya → Not defteri yükle** ile `colab_egitim.ipynb`'yi açın.
2. **Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU**.
3. Hücreleri sırayla çalıştırın.

## Yerelde
```
python train.py                     # GPU yoksa çok yavaştır
python evaluate.py model_en_iyi.pt --oyun 200
python play.py
```

`--sekillendirme-yok` ile "yeme yaklaşma" ödülü kapatılabilir (çok daha zor bir deney).
