# Drosophila Connectome 3D Simulation & Reinforcement Learning from Scratch

*[English](#english) | [Türkçe](#türkçe)*

---

<a name="english"></a>
## English

### Overview
This experimental computational-neuroscience & AI project explores **fruit fly (*Drosophila melanogaster*) brain-inspired navigation** in two complementary parts:
1. **`simulasyon.py`** — A real-time 3D **Ursina Engine** arena simulation that loads the **FlyWire / Princeton Drosophila Whole-Brain Connectome** (`connections_princeton.csv.gz`) and simulates 3 virtual flies navigating a 3D grid toward sugar sources with reward-modulated synaptic weight updates.
2. **`sinek_from_scratch/`** — A modular PyTorch reinforcement-learning and neural brain framework built from scratch (`brain/model.py`, `environment/`, `rl/`, `visualization/`, `config.py`, `train.py`).

### Development Timeline & Process
- **August 18, 2026:** Built the 3D Ursina connectome foraging simulation (`simulasyon.py`) and initialized the modular `sinek_from_scratch/` architecture.
- **August 19, 2026:** Implemented and trained the custom PyTorch brain/RL loop (`sinek_from_scratch/brain/model.py`, `train.py`, `config.py`), generating iterative brain checkpoints (`fly_brain_update_*.pt`).

### Excluded Large Files
To keep the repository lightweight for GitHub, the ~68.5 MB compressed FlyWire connectome CSVs (`connections_princeton.csv.gz`, `consolidated_cell_types.csv.gz`, `names.csv.gz`), training checkpoints (`checkpoints/*.pt`), and run logs (`logs/`) are excluded via `.gitignore`. Download the FlyWire Princeton connectome data from [Codex / FlyWire](https://codex.flywire.ai/api/download) and place it in the project root to run `simulasyon.py`.

### Setup & Usage
```bash
pip install -r requirements.txt

# 1. Run 3D Ursina simulation (requires connections_princeton.csv.gz in root)
python simulasyon.py

# 2. Run from-scratch PyTorch RL training
cd sinek_from_scratch
python train.py
```

### Acknowledgments & Open-Source Credits
We are deeply grateful to the neuroscience and open-source AI communities for making whole-brain connectome data publicly accessible:
- **[FlyWire / Princeton Drosophila Whole-Brain Connectome (Codex)](https://codex.flywire.ai/)** — Dorkenwald et al. (*Nature*, 2024) and Schlegel et al. (*Nature*, 2024), Princeton University & the FlyWire Consortium.
- **[Ursina Engine](https://www.ursinaengine.org/)** — Python 3D simulation & game engine.
- **[PyTorch](https://pytorch.org/)**, **[Pandas](https://pandas.pydata.org/)**, and **[NumPy](https://numpy.org/)**.

---

<a name="türkçe"></a>
## Türkçe

### Genel Bakış
Bu deneysel hesaplamalı sinirbilim ve yapay zeka projesi, **meyve sineği (*Drosophila melanogaster*) konnektomu ve beyninden esinlenen 3B yön bulma / pekiştirmeli öğrenme** çalışmalarını iki aşamada ele alır:
1. **`simulasyon.py`** — **FlyWire / Princeton Drosophila Konnektomu** (`connections_princeton.csv.gz`) verisini okuyarak **Ursina 3B motoru** içinde şeker kaynaklarına yönelen 3 sanal sineği ödül temelli sinaptik güncellemelerle simüle eder.
2. **`sinek_from_scratch/`** — Sıfırdan PyTorch ile kurgulanmış modüler beyin ve pekiştirmeli öğrenme (RL) eğitim altyapısı (`brain/`, `environment/`, `rl/`, `visualization/`, `config.py`, `train.py`).

### Geliştirme Süreci ve Tarihçesi
- **18 Ağustos 2026:** Ursina 3B simülasyonu (`simulasyon.py`) ve `sinek_from_scratch/` klasör mimarisi oluşturuldu.
- **19 Ağustos 2026:** Özel PyTorch beyin modeli (`brain/model.py`) ve RL eğitim döngüsü (`train.py`) geliştirilip eğitildi.

### Hariç Tutulan Büyük Dosyalar
~68.5 MB'lık sıkıştırılmış FlyWire konnektom verileri (`*.csv.gz`), eğitim ağırlıkları (`checkpoints/*.pt`) ve log dosyaları `.gitignore` ile hariç tutulmuştur. `simulasyon.py` dosyasını çalıştırmak için FlyWire verilerini [Codex](https://codex.flywire.ai/api/download) üzerinden indirip proje kök dizinine koyabilirsiniz.

### Açık Kaynak Teşekkürler ve Kaynaklar
- **[FlyWire / Princeton Drosophila Whole-Brain Connectome (Codex)](https://codex.flywire.ai/)** — Dorkenwald vd. (*Nature*, 2024) ve Schlegel vd. (*Nature*, 2024).
- **[Ursina Engine](https://www.ursinaengine.org/)**, **[PyTorch](https://pytorch.org/)**, **[Pandas](https://pandas.pydata.org/)** ve **[NumPy](https://numpy.org/)**.