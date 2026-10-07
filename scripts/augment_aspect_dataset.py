"""
Augmentasi data latih ekstraksi aspek: substitusi istilah + contoh negatif.

MASALAH YANG DIATASI
--------------------
Diukur pada eksperimen K1 vs K2, faktor penentu apakah model mempelajari POLA
atau MENGHAFAL adalah keragaman leksikal label:

    K1  TTR  3,8%  | 10 istilah teratas menutupi 75,4% span | F1 lintas domain 0,039
    K2  TTR 45,5%  | 10 istilah teratas menutupi 13,0% span | F1 lintas domain 0,506

Bila sedikit kata sudah menjelaskan hampir seluruh label, menghafal kata-kata itu
adalah jalan termurah bagi gradient descent, dan sintaksis tidak pernah dipelajari.

CARA KERJA
----------
1. Substitusi istilah. Setiap span aspek diganti dengan istilah lain dari
   kumpulan istilah korpus, sementara POSISI dan KONTEKS kalimat dipertahankan.
   Pola sintaktisnya tetap, identitas katanya berubah - sehingga menghafal tidak
   lagi menghasilkan apa pun dan model dipaksa membaca posisi.

2. Contoh negatif. Kalimat tanpa aspek disertakan (jalur pembuatan data
   sebelumnya membuangnya). Tanpa contoh ini model tidak pernah diajari KAPAN
   TIDAK menandai apa pun, yang menaikkan false positive.

Offset dihitung ulang dengan benar setelah setiap substitusi, sehingga label
tetap sejajar dengan teks hasil augmentasi.

Pemakaian:
    python scripts/augment_aspect_dataset.py \
        --data data/external/combined_train_retrain.json \
        --negatives data/external/terma/train_gold.json \
        --out data/external/k5_train_retrain.json --factor 1
"""

import argparse
import json
import os
import random
import re
from collections import Counter
from typing import Any, Dict, List, Tuple


# Kata fungsi tidak layak menjadi pengganti aspek: pola sintaktisnya berbeda
# dan hasil substitusinya menjadi kalimat yang tidak masuk akal.
FUNCTION_WORDS = {
    'yang', 'dan', 'atau', 'tetapi', 'namun', 'karena', 'untuk', 'kepada',
    'dari', 'pada', 'dengan', 'oleh', 'dalam', 'atas', 'antara', 'tentang',
    'adalah', 'itu', 'ini', 'saya', 'kami', 'kita', 'anda', 'mereka', 'akan',
    'sudah', 'telah', 'sedang', 'masih', 'belum', 'tidak', 'bukan', 'juga',
    'hanya', 'saja', 'lebih', 'paling', 'sangat', 'agar', 'hingga', 'sampai',
    'sejak', 'setelah', 'sebelum', 'ketika', 'saat', 'bahwa', 'para', 'kalau',
    'jika', 'bisa', 'dapat', 'harus', 'sekali', 'banget', 'buat',
    'tersebut', 'demikian', 'begitu', 'seperti', 'sehingga', 'walaupun',
    'meskipun', 'selain', 'sedangkan', 'maupun', 'yaitu', 'yakni', 'antara',
    'sangat', 'cukup', 'agak', 'lagi', 'masing', 'setiap', 'semua', 'seluruh',
}


def load(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def spans_of(item: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Ambil offset span; hanya item yang membawa 'spans' yang bisa diaugmentasi."""
    return [(s['start'], s['end']) for s in item.get('spans') or []]


def substitute(
    text: str,
    spans: List[Tuple[int, int]],
    pool: List[str],
    rng: random.Random,
) -> Tuple[str, List[Dict[str, int]]]:
    """
    Ganti setiap span dengan istilah acak dari pool, hitung ulang offsetnya.

    Args:
        text: Kalimat asli.
        spans: Offset span aspek pada kalimat asli.
        pool: Kumpulan istilah pengganti.
        rng: Sumber acak (di-seed agar hasilnya dapat direproduksi).

    Returns:
        Tuple (teks_baru, spans_baru).
    """
    ordered = sorted(spans)
    out_parts: List[str] = []
    new_spans: List[Dict[str, int]] = []
    cursor = 0
    length = 0

    for start, end in ordered:
        # potongan teks sebelum span
        chunk = text[cursor:start]
        out_parts.append(chunk)
        length += len(chunk)

        replacement = rng.choice(pool)
        new_spans.append({'start': length, 'end': length + len(replacement)})
        out_parts.append(replacement)
        length += len(replacement)

        cursor = end

    out_parts.append(text[cursor:])
    return ''.join(out_parts), new_spans


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='payload retraining sumber')
    ap.add_argument('--out', required=True)
    ap.add_argument('--factor', type=int, default=1,
                    help='berapa salinan tersubstitusi per kalimat asli')
    ap.add_argument('--negatives', nargs='*', default=[],
                    help='berkas *_gold.json; kalimat tanpa aspek diambil dari sini')
    ap.add_argument('--max-negatives', type=int, default=0,
                    help='batasi jumlah contoh negatif (0 = semua)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data = load(args.data)

    # Pool substitusi diambil dari kata yang muncul di korpus tetapi TIDAK PERNAH
    # menjadi aspek.
    #
    # Ini inti tekniknya. Mengganti aspek dengan istilah aspek lain tidak
    # memutus kaitan kata->label: setiap kata pool tetap selalu berlabel aspek,
    # sehingga menghafal masih menguntungkan. Dengan memakai kata non-aspek,
    # kata yang sama muncul sebagai 'O' di kalimat asli dan 'B-ASPECT' di
    # kalimat hasil substitusi. Identitas kata menjadi tidak informatif dan
    # satu-satunya sinyal yang tersisa adalah posisi serta konteks sintaktisnya.
    aspect_types = {a.strip().lower() for item in data
                    for a in item.get('aspects', []) if isinstance(a, str)}
    aspect_words = {w for t in aspect_types for w in t.split()}

    freq: Counter = Counter()
    for item in data:
        for w in re.findall(r'[A-Za-z]{4,}', item['text']):
            wl = w.lower()
            if wl not in aspect_words and wl not in FUNCTION_WORDS:
                freq[wl] += 1

    pool = [w for w, c in freq.most_common(600) if c >= 2]
    if len(pool) < 20:
        raise SystemExit('kata non-aspek terlalu sedikit untuk dijadikan pool')

    out: List[Dict[str, Any]] = list(data)
    augmented = 0
    skipped_no_spans = 0

    for item in data:
        spans = spans_of(item)
        if not spans:
            skipped_no_spans += 1
            continue
        for _ in range(args.factor):
            new_text, new_spans = substitute(item['text'], spans, pool, rng)
            out.append({
                'text': new_text,
                'aspects': [new_text[s['start']:s['end']] for s in new_spans],
                'spans': new_spans,
                'augmented': True,
            })
            augmented += 1

    # ── Contoh negatif ──────────────────────────────────────────────────────
    negatives = 0
    for path in args.negatives:
        if not os.path.exists(path):
            print(f'  lewati (tidak ada): {path}')
            continue
        for doc in load(path):
            if doc.get('spans'):
                continue
            if args.max_negatives and negatives >= args.max_negatives:
                break
            out.append({'text': doc['text'], 'aspects': [], 'spans': [],
                        'negative': True})
            negatives += 1

    rng.shuffle(out)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ── Laporan keragaman ───────────────────────────────────────────────────
    def diversity(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        surf = Counter(a.lower() for it in items for a in it.get('aspects', []))
        total = sum(surf.values())
        if not total:
            return {'span': 0, 'unik': 0, 'ttr': 0.0, 'top10': 0.0, 'ambigu': 0.0}

        # Ambiguitas: berapa persen tipe aspek yang JUGA muncul di luar span
        # aspek. Makin tinggi, makin tidak berguna menghafal identitas kata.
        aspect_types = set(surf)
        outside: Counter = Counter()
        for it in items:
            text = it['text']
            covered = set()
            for sp in it.get('spans') or []:
                covered.update(range(sp['start'], sp['end']))
            for m in re.finditer(r'[A-Za-z]{2,}', text):
                if not (set(range(m.start(), m.end())) & covered):
                    outside[m.group().lower()] += 1
        ambigu = sum(1 for t in aspect_types
                     if all(w in outside for w in t.split()))
        return {
            'span': total, 'unik': len(surf),
            'ttr': len(surf) / total * 100,
            'top10': sum(c for _, c in surf.most_common(10)) / total * 100,
            'ambigu': ambigu / len(aspect_types) * 100,
        }

    before = diversity(data)
    after = diversity(out)

    print('=' * 62)
    print('AUGMENTASI DATA LATIH')
    print('=' * 62)
    print(f'  sumber                 : {args.data}')
    print(f'  kalimat asli           : {len(data)}')
    print(f'  tanpa offset (dilewati): {skipped_no_spans}')
    print(f'  salinan tersubstitusi  : {augmented}  (faktor {args.factor})')
    print(f'  contoh negatif         : {negatives}')
    print(f'  total keluaran         : {len(out)}')
    print()
    print(f"{'':24}{'sebelum':>12}{'sesudah':>12}")
    print(f"{'  span':24}{before['span']:>12}{after['span']:>12}")
    print(f"{'  istilah unik':24}{before['unik']:>12}{after['unik']:>12}")
    print(f"{'  TTR':24}{before['ttr']:>11.1f}%{after['ttr']:>11.1f}%")
    print(f"{'  10 teratas menutupi':24}{before['top10']:>11.1f}%{after['top10']:>11.1f}%")
    print(f"{'  tipe aspek ambigu':24}{before['ambigu']:>11.1f}%{after['ambigu']:>11.1f}%")
    print()
    print('  "tipe aspek ambigu" = istilah aspek yang juga muncul sebagai non-aspek.')
    print('  Inilah yang membuat menghafal identitas kata tidak lagi menguntungkan.')
    print()
    print(f'  -> {args.out}')


if __name__ == '__main__':
    main()
