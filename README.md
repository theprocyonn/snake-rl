# Kendi kendine öğrenen yılan

Klasik yılan oyununu hem elle oynayabileceğiniz hem de bir yapay zekanın **sıfırdan öğrenerek**
oynamasını izleyebileceğiniz bir pekiştirmeli öğrenme (reinforcement learning) projesi.

Amaç: makine öğrenmesinin gerçekte nasıl çalıştığını görmek. Ajana nasıl oynayacağı söylenmez;
yalnızca ödül (yem +, ölüm −) alır ve deneme-yanılmayla öğrenir.

## Proje bölümleri

| Bölüm | İçerik | Durum |
|---|---|---|
| `web/` | Tarayıcıda çalışan sürüm: tablo tabanlı Q-öğrenme, canlı öğrenme eğrisi | Hazır |
| `bolum1_ozellik_tabanli/` | Python + PyTorch DQN. Ajan, elle tasarlanmış 17 özellik görür | Hazır |
| `bolum2_sifirdan/` | Ham tahta görüntüsünden öğrenme (el yapımı özellik yok) | Planlandı |

## Sonuçlar (Bölüm 1)

Rastgele hamle kapalıyken, her model için 200 oyun:

| Yöntem | Ortalama yem |
|---|---|
| Tablo tabanlı Q-öğrenme (web) | ~18 |
| DQN, gamma 0.95 | 57,0 |
| DQN, gamma 0.99 | 65,5 |
| DQN, gamma 0.99, 400.000 adım | 63,0 |
| DQN, gamma 0.99 + kuyruğa yol bilgisi (v2) | **73,4** (ortanca 74, en yüksek 119) |

Not: Her satır tek bir eğitimin sonucudur. Eğitimden eğitime birkaç puanlık oynama vardır,
bu yüzden küçük farkları kesin kanıt saymayın.

![Öğrenme eğrisi](bolum1_ozellik_tabanli/egitim_egrisi_v2.png)

### Neler öğrendik
- Ajan oynamayı gerçekten kendi başına öğreniyor, ama en büyük sıçramalar ona **ne gösterdiğimizden** geldi
  (boş alan, kuyruğa yol, ödül tasarımı), öğrenme algoritmasından değil.
- Gelecek ödüle verilen ağırlık (gamma) 0.95'ten 0.99'a çıkınca sonuç belirgin biçimde iyileşti.
- Eğitimi 2,5 kat uzatmak işe yaramadı: tavan, eğitim süresinden değil ajanın gördüğü bilgiden geliyordu.
- Eğitim grafiğindeki skorlar, rastgele keşif adımları yüzünden gerçek performansın yaklaşık yarısıdır.
  Gerçek gücü `evaluate.py` (rastgele hamle yok) ölçer.

## Kurulum

Python 3.10+ önerilir.

```
pip install -r requirements.txt
```

Windows'ta çok yeni Python sürümlerinde `pygame` derlenemeyebilir, bu yüzden `pygame-ce` kullanılır
(kodda yine `import pygame` yazılır).

## Kullanım (Bölüm 1)

```
cd bolum1_ozellik_tabanli
python train.py                         # eğit (yaklaşık 10-15 dk), eğri çizer
python evaluate.py model_en_iyi.pt      # rastgele hamlesiz 100 oyunla ölç
python play.py                          # 1: insan modu, 2: yapay zeka modu
```

İsteğe bağlı: `python train.py --adim 400000`, `python train.py --canli 50` (eğitimi pencerede izle).

Depodaki `model_en_iyi.pt`, eğitilmiş ajandır; eğitim yapmadan `python play.py` ile izleyebilirsiniz.

## Tarayıcı sürümü

`web/yilan.html` dosyasını tarayıcıda açmanız yeterli. Kurulum gerekmez.

## Dosyalar (Bölüm 1)

- `snake_env.py`: oyun motoru ve ajanın gördüğü 17 sayılık durum vektörü
- `agent.py`: sinir ağı, deneyim havuzu, hedef ağ, öğrenme adımı (DQN)
- `train.py`: eğitim döngüsü ve öğrenme eğrisi
- `evaluate.py`: modelleri rastgele hamlesiz ölçme
- `play.py`, `render.py`: pygame penceresi (insan ve yapay zeka modu)
