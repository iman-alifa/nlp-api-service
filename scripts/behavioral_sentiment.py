"""
Uji perilaku (behavioral testing) modul analisis sentimen - gaya CheckList.

Akurasi agregat pada satu korpus menyembunyikan kegagalan sistematis: model bisa
mencetak 0,90 sambil selalu salah pada kalimat bernegasi, atau berubah pikiran
hanya karena ejaan slang. Ribeiro, Wu, Guestrin & Singh (2020) menunjukkan
kelemahan seperti itu baru terlihat bila kemampuannya diuji satu per satu.

Nilainya khusus untuk proyek ini: uji ini **tidak memerlukan data berlabel**.
Labelnya diketahui dari cara kalimatnya dibangun, sehingga generalitas modul
bisa diukur sekarang juga - sementara korpus produksi (komentar YouTube) belum
punya anotasi manusia sama sekali.

Tiga jenis uji, mengikuti istilah aslinya:

- **MFT** (Minimum Functionality Test) - kalimat dibangun dari templat sehingga
  labelnya pasti. Menguji kemampuan paling dasar.
- **INV** (Invariance) - perubahan yang TIDAK boleh mengubah label (ejaan slang,
  emoji, typo ringan). Menguji ketahanan terhadap ragam bahasa media sosial,
  yang justru ciri domain produksi.
- **DIR** (Directional) - perubahan yang harus menggeser prediksi ke arah
  tertentu, mis. menambahkan klausa negatif tidak boleh menaikkan skor positif.

Contoh:
    python scripts/behavioral_sentiment.py
    python scripts/behavioral_sentiment.py --show-failures
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import torch  # noqa: F401  (torch sebelum nltk - WinError 1114)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.sentiment_service import SentimentService  # noqa: E402

LARAVEL_CONFIG = {
    'case_folding': True, 'remove_punctuation': True, 'remove_numbers': False,
    'remove_stopwords': True, 'stemming': True, 'lemmatization': False,
    'custom_stopwords': [],
}

# ── Bahan templat ───────────────────────────────────────────────────────────

SUBJEK = ['pelayanannya', 'produknya', 'aplikasinya', 'makanannya', 'tempatnya',
          'harganya', 'kualitasnya', 'petugasnya']

POSITIF = ['bagus', 'memuaskan', 'ramah', 'cepat', 'nyaman', 'rapi']
NEGATIF = ['buruk', 'mengecewakan', 'lambat', 'kotor', 'berantakan', 'payah']

NEGASI = ['tidak', 'kurang', 'nggak', 'gak']
PENGUAT = ['sangat', 'benar-benar', 'amat', 'sungguh']

FAKTUAL = [
    'rapat dimulai pukul sembilan pagi',
    'kantor buka setiap hari senin sampai jumat',
    'antrian dibagi menjadi tiga jalur',
    'formulir tersedia di lantai dua',
    'jadwal keberangkatan tertera di layar',
]


def _mft_dasar() -> List[Tuple[str, str, str]]:
    """Kalimat lugas tanpa kerumitan apa pun - dasar dari segalanya."""
    kasus = []
    for subjek in SUBJEK[:5]:
        for kata in POSITIF[:4]:
            kasus.append(('MFT dasar positif', f'{subjek} {kata}', 'positive'))
        for kata in NEGATIF[:4]:
            kasus.append(('MFT dasar negatif', f'{subjek} {kata}', 'negative'))
    return kasus


def _mft_negasi() -> List[Tuple[str, str, str]]:
    """
    Negasi membalik polaritas.

    Inilah alasan `_sanitize_config` mematikan stemming dan stopword removal:
    keduanya membuang kata negasi, dan "tidak bagus" berubah menjadi "bagus".
    """
    kasus = []
    for subjek in SUBJEK[:4]:
        for negasi in NEGASI[:2]:
            for kata in POSITIF[:3]:
                kasus.append(('MFT negasi atas positif',
                              f'{subjek} {negasi} {kata}', 'negative'))
            for kata in NEGATIF[:3]:
                # "tidak buruk" bukan pujian, tetapi jelas BUKAN negatif.
                kasus.append(('MFT negasi atas negatif',
                              f'{subjek} {negasi} {kata}', 'not-negative'))
    return kasus


def _mft_kontras() -> List[Tuple[str, str, str]]:
    """
    Pada bahasa Indonesia, klausa setelah 'tapi' membawa penilaian utama.

    Kemampuan ini yang membuat analisis per-aspek bermakna; modul aspek memang
    memecah kalimat di konektor kontrastif justru karena ini.
    """
    kasus = []
    for subjek, lain in zip(SUBJEK[:4], SUBJEK[4:8]):
        for pos, neg in zip(POSITIF[:3], NEGATIF[:3]):
            kasus.append(('MFT kontras -> negatif',
                          f'{subjek} {pos} tapi {lain} {neg}', 'negative'))
            kasus.append(('MFT kontras -> positif',
                          f'{subjek} {neg} tapi {lain} {pos}', 'positive'))
    return kasus


def _mft_netral() -> List[Tuple[str, str, str]]:
    """Kalimat faktual tanpa penilaian - kelas paling lemah modul ini."""
    return [('MFT faktual netral', teks, 'neutral') for teks in FAKTUAL]


MFT_BUILDERS = [_mft_dasar, _mft_negasi, _mft_kontras, _mft_netral]


# ── Perturbasi untuk INV ────────────────────────────────────────────────────

def _slang(teks: str) -> str:
    """Ejaan khas media sosial - persis ragam bahasa domain produksi."""
    peta = {'tidak': 'gk', 'sangat': 'bgt', 'yang': 'yg', 'dengan': 'dgn',
            'saja': 'aja', 'sudah': 'udh', 'tempatnya': 'tmptnya'}
    return ' '.join(peta.get(w, w) for w in teks.split())


def _huruf_berulang(teks: str) -> str:
    """"bagus" -> "bagusss" - penekanan lazim pada komentar."""
    kata = teks.split()
    kata[-1] = kata[-1] + kata[-1][-1] * 3
    return ' '.join(kata)


def _emoji(teks: str) -> str:
    return teks + ' 😊😊'


def _kapital(teks: str) -> str:
    return teks.upper()


def _tanda_baca(teks: str) -> str:
    return teks + '!!!'


PERTURBASI = {
    'INV ejaan slang': _slang,
    'INV huruf berulang': _huruf_berulang,
    'INV emoji': _emoji,
    'INV huruf kapital': _kapital,
    'INV tanda baca': _tanda_baca,
}


def cocok(prediksi: str, harapan: str) -> bool:
    """`not-negative` menerima positive maupun neutral."""
    if harapan == 'not-negative':
        return prediksi != 'negative'
    return prediksi == harapan


async def main_async(args: argparse.Namespace) -> None:
    # `model_path` menimpa penemuan checkpoint, sehingga model pembanding bisa
    # diuji lewat jalur produksi yang sama persis.
    service = SentimentService(model_path=args.model) if args.model else SentimentService()
    if args.model:
        # Peta label dibaca dari config model pembanding, bukan diasumsikan.
        service._ensure_loaded()
        raw = getattr(service.model.config, 'id2label', {}) or {}
        alias = {'positive': 'positive', 'positif': 'positive',
                 'neutral': 'neutral', 'netral': 'neutral',
                 'negative': 'negative', 'negatif': 'negative'}
        peta = {int(k): alias.get(str(v).strip().lower()) for k, v in raw.items()}
        if all(peta.values()) and len(peta) == 3:
            import app.services.sentiment_service as modul
            modul.LABEL_MAP.update(peta)
        print(f'model: {args.model} | peta label: {modul.LABEL_MAP if all(peta.values()) else "bawaan"}')

    kasus: List[Tuple[str, str, str]] = []
    for builder in MFT_BUILDERS:
        kasus.extend(builder())

    # Basis INV/DIR: kalimat lugas yang seharusnya jelas bagi model.
    basis = [f'{s} {k}' for s in SUBJEK[:4] for k in POSITIF[:3]]
    basis += [f'{s} {k}' for s in SUBJEK[:4] for k in NEGATIF[:3]]

    semua_teks = [teks for _, teks, _ in kasus]
    teks_inv = [(nama, t, fn(t)) for nama, fn in PERTURBASI.items() for t in basis]
    dir_teks = [(t, f'{t} tapi petugasnya sangat kasar') for t in basis[:12]]

    # Satu panggilan untuk semuanya - pengurutan panjang di dalam service
    # membuatnya jauh lebih murah daripada memanggil per kelompok.
    gabungan = (
        semua_teks
        + [t for _, t, _ in teks_inv] + [p for _, _, p in teks_inv]
        + [a for a, _ in dir_teks] + [b for _, b in dir_teks]
    )
    hasil = await service.analyze(texts=gabungan, preprocessing_config=LARAVEL_CONFIG)
    prediksi = hasil['predictions']

    def ambil(offset: int, jumlah: int):
        return prediksi[offset:offset + jumlah]

    pos = 0
    hasil_mft = ambil(pos, len(semua_teks)); pos += len(semua_teks)
    inv_asal = ambil(pos, len(teks_inv)); pos += len(teks_inv)
    inv_ubah = ambil(pos, len(teks_inv)); pos += len(teks_inv)
    dir_asal = ambil(pos, len(dir_teks)); pos += len(dir_teks)
    dir_ubah = ambil(pos, len(dir_teks))

    kelompok: Dict[str, Dict[str, Any]] = {}
    kegagalan: List[Dict[str, Any]] = []

    def catat(nama: str, lulus: bool, detail: Dict[str, Any]) -> None:
        row = kelompok.setdefault(nama, {'lulus': 0, 'total': 0})
        row['total'] += 1
        if lulus:
            row['lulus'] += 1
        else:
            kegagalan.append({'uji': nama, **detail})

    # MFT
    for (nama, teks, harapan), p in zip(kasus, hasil_mft):
        catat(nama, cocok(p['sentiment'], harapan),
              {'teks': teks, 'harapan': harapan,
               'prediksi': p['sentiment'], 'confidence': p['confidence']})

    # INV - label tidak boleh berubah oleh perturbasi permukaan
    for (nama, asal, ubah), pa, pb in zip(teks_inv, inv_asal, inv_ubah):
        catat(nama, pa['sentiment'] == pb['sentiment'],
              {'teks': asal, 'diubah': ubah,
               'harapan': pa['sentiment'], 'prediksi': pb['sentiment'],
               'confidence': pb['confidence']})

    # DIR - menambah klausa negatif tidak boleh MENAIKKAN skor positif
    for (asal, ubah), pa, pb in zip(dir_teks, dir_asal, dir_ubah):
        naik = pb['scores']['positive'] - pa['scores']['positive']
        catat('DIR tambah klausa negatif', naik <= 0.05,
              {'teks': asal, 'diubah': ubah,
               'harapan': 'skor positif tidak naik',
               'prediksi': f'naik {naik:+.3f}', 'confidence': pb['confidence']})

    # ── Laporan ─────────────────────────────────────────────────────────────
    print(f'\n{len(kasus)} kasus MFT, {len(teks_inv)} INV, {len(dir_teks)} DIR\n')
    header = f"{'uji':32} {'lulus':>7} {'total':>7} {'rasio':>8}"
    print(header)
    print('-' * len(header))

    total_lulus = total_kasus = 0
    for nama in sorted(kelompok):
        row = kelompok[nama]
        total_lulus += row['lulus']
        total_kasus += row['total']
        rasio = row['lulus'] / row['total']
        tanda = '' if rasio >= 0.9 else ('  <- lemah' if rasio >= 0.6 else '  <- GAGAL')
        print(f"{nama:32} {row['lulus']:>7} {row['total']:>7} {rasio:>8.3f}{tanda}")

    print('-' * len(header))
    print(f"{'TOTAL':32} {total_lulus:>7} {total_kasus:>7} "
          f"{total_lulus / total_kasus:>8.3f}")

    if args.show_failures and kegagalan:
        print(f'\n{len(kegagalan)} kegagalan, {min(25, len(kegagalan))} pertama:')
        for row in kegagalan[:25]:
            print(f"  [{row['uji']}] {row['teks'][:52]!r}")
            print(f"      harap={row['harapan']} dapat={row['prediksi']} "
                  f"(conf {row['confidence']})")

    payload = {
        'summary': {nama: kelompok[nama] for nama in sorted(kelompok)},
        'total': {'lulus': total_lulus, 'total': total_kasus,
                  'rate': round(total_lulus / total_kasus, 4)},
        'failures': kegagalan,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f'\ntersimpan ke {args.out}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show-failures', action='store_true')
    parser.add_argument('--model', default=None,
                        help='uji model pembanding, bukan model produksi')
    parser.add_argument('--out', default='data/experiments/sentiment_behavioral.json')
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
