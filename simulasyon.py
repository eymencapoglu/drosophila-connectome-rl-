from ursina import *
import pandas as pd
import torch
import numpy as np

# ==========================================
# 1. VERİ OKUMA VE BEYİN KURULUMU
# ==========================================
print("Veriler yükleniyor ve beyin matrisleri oluşturuluyor...")

df_conn = pd.read_csv('connections_princeton.csv.gz')
df_filtered = df_conn.sort_values(by='syn_count', ascending=False).head(1500)

NUM_FLIES = 3
GRID_SIZE = 15

beyinler = []
for _ in range(NUM_FLIES):
    baslangic_beyni = torch.eye(6) * 0.5  
    beyinler.append(baslangic_beyni.clone())

flies_pos = np.random.randint(0, GRID_SIZE, (NUM_FLIES, 3))
sugars_pos = np.random.randint(2, GRID_SIZE-2, (2, 3))
ogrenme_hizi = 0.15

def mesafe_hesapla_3d(pos1, pos2):
    return np.sqrt(np.sum((pos1 - pos2)**2))

def en_yakin_sekeri_bul(sinek_pos):
    mesafeler = [mesafe_hesapla_3d(sinek_pos, s) for s in sugars_pos]
    hedef_idx = np.argmin(mesafeler)
    return sugars_pos[hedef_idx], hedef_idx, mesafeler[hedef_idx]

eski_mesafeler = [en_yakin_sekeri_bul(flies_pos[i])[2] for i in range(NUM_FLIES)]

# ==========================================
# 2. URSINA 3D OYUN MOTORU KURULUMU
# ==========================================
app = Ursina()

camera.position = (GRID_SIZE/2, GRID_SIZE, -GRID_SIZE)
camera.look_at((GRID_SIZE/2, GRID_SIZE/2, GRID_SIZE/2))
EditorCamera()

merkez = GRID_SIZE/2 - 0.5
Entity(model='cube', scale=(GRID_SIZE, 0.2, GRID_SIZE), color=color.gray, position=(merkez, -0.5, merkez))
Entity(model='cube', scale=(GRID_SIZE, GRID_SIZE, 0.1), color=color.white10, position=(merkez, merkez, -0.5))
Entity(model='cube', scale=(GRID_SIZE, GRID_SIZE, 0.1), color=color.white10, position=(merkez, merkez, GRID_SIZE-0.5))
Entity(model='cube', scale=(0.1, GRID_SIZE, GRID_SIZE), color=color.white10, position=(-0.5, merkez, merkez))
Entity(model='cube', scale=(0.1, GRID_SIZE, GRID_SIZE), color=color.white10, position=(GRID_SIZE-0.5, merkez, merkez))

fly_entities = []
renkler = [color.cyan, color.magenta, color.green]

for i in range(NUM_FLIES):
    fly_pivot = Entity(position=tuple(flies_pos[i]))
    
    # HATA ÇÖZÜMÜ: capsule yerine esnetilmiş sphere kullanıyoruz.
    govde = Entity(parent=fly_pivot, model='sphere', color=color.black, scale=(0.4, 0.4, 0.8))
    goz = Entity(parent=fly_pivot, model='sphere', color=renkler[i], scale=(0.3, 0.3, 0.3), position=(0, 0.1, 0.3))
    
    kanat_sag = Entity(parent=fly_pivot, model='cube', color=color.white50, scale=(0.5, 0.05, 0.3), position=(0.3, 0.2, -0.1))
    kanat_sol = Entity(parent=fly_pivot, model='cube', color=color.white50, scale=(0.5, 0.05, 0.3), position=(-0.3, 0.2, -0.1))
    
    fly_pivot.kanat_sag = kanat_sag
    fly_pivot.kanat_sol = kanat_sol
    
    fly_entities.append(fly_pivot)

sugar_entities = []
for s in sugars_pos:
    # HATA ÇÖZÜMÜ: octahedron yerine cube kullanıyoruz.
    sugar = Entity(model='cube', color=color.white, scale=0.6, position=tuple(s))
    sugar.animate_rotation_y(360, duration=2, loop=True)
    # pingpong argümanını kaldırdık, parlama efektini update döngüsünde yapacağız
    sugar_entities.append(sugar)

# ==========================================
# 3. SİMÜLASYON DÖNGÜSÜ
# ==========================================
ai_zamanlayici = 0
adim = 0

def update():
    global ai_zamanlayici, adim, flies_pos, sugars_pos, eski_mesafeler
    
    for fly in fly_entities:
        fly.kanat_sag.rotation_z = math.sin(time.time() * 20) * 30
        fly.kanat_sol.rotation_z = -math.sin(time.time() * 20) * 30
        
    # HATA ÇÖZÜMÜ: Şekerlerin parlama (scale) animasyonunu manuel matematik ile veriyoruz
    for sugar in sugar_entities:
        sugar.scale = 0.6 + math.sin(time.time() * 5) * 0.1

    ai_zamanlayici += time.dt
    if ai_zamanlayici < 0.25:
        return
    ai_zamanlayici = 0
    adim += 1

    for i in range(NUM_FLIES):
        hedef, hedef_idx, mevcut_mesafe = en_yakin_sekeri_bul(flies_pos[i])
        
        sensor = torch.zeros(6)
        if hedef[0] > flies_pos[i][0]: sensor[0] = (hedef[0] - flies_pos[i][0]) / GRID_SIZE
        elif hedef[0] < flies_pos[i][0]: sensor[1] = (flies_pos[i][0] - hedef[0]) / GRID_SIZE
        
        if hedef[1] > flies_pos[i][1]: sensor[2] = (hedef[1] - flies_pos[i][1]) / GRID_SIZE
        elif hedef[1] < flies_pos[i][1]: sensor[3] = (flies_pos[i][1] - hedef[1]) / GRID_SIZE
        
        if hedef[2] > flies_pos[i][2]: sensor[4] = (hedef[2] - flies_pos[i][2]) / GRID_SIZE
        elif hedef[2] < flies_pos[i][2]: sensor[5] = (flies_pos[i][2] - hedef[2]) / GRID_SIZE

        beyin_aktivitesi = torch.matmul(beyinler[i], sensor)
        biyolojik_gurultu = torch.randn(6) * 0.1
        motor_cikis = beyin_aktivitesi + biyolojik_gurultu
        aksiyon = torch.argmax(motor_cikis).item()

        if aksiyon == 0: flies_pos[i][0] += 1
        elif aksiyon == 1: flies_pos[i][0] -= 1
        elif aksiyon == 2: flies_pos[i][1] += 1
        elif aksiyon == 3: flies_pos[i][1] -= 1
        elif aksiyon == 4: flies_pos[i][2] += 1
        elif aksiyon == 5: flies_pos[i][2] -= 1

        flies_pos[i][0] = max(0, min(GRID_SIZE-1, flies_pos[i][0]))
        flies_pos[i][1] = max(0, min(GRID_SIZE-1, flies_pos[i][1]))
        flies_pos[i][2] = max(0, min(GRID_SIZE-1, flies_pos[i][2]))

        fly_entities[i].look_at(tuple(flies_pos[i]))
        fly_entities[i].position = tuple(flies_pos[i])

        yeni_mesafe = mesafe_hesapla_3d(flies_pos[i], hedef)
        aktif_sensor = torch.argmax(sensor).item()
        
        if yeni_mesafe == 0:
            beyinler[i][aksiyon, aktif_sensor] += ogrenme_hizi * 5 
            print(f"ADIM {adim}: Sinek {i+1} şekeri kaptı! Beyni güçlendi.")
            yeni_seker = np.random.randint(2, GRID_SIZE-2, 3)
            sugars_pos[hedef_idx] = yeni_seker
            sugar_entities[hedef_idx].position = tuple(yeni_seker)
            yeni_mesafe = mesafe_hesapla_3d(flies_pos[i], sugars_pos[hedef_idx])
        elif yeni_mesafe < eski_mesafeler[i]:
            beyinler[i][aksiyon, aktif_sensor] += ogrenme_hizi
        else:
            beyinler[i][aksiyon, aktif_sensor] -= ogrenme_hizi * 2

        eski_mesafeler[i] = yeni_mesafe

print("Yapay Zeka Hazır! 3D Simülasyon başlatılıyor...")
app.run()