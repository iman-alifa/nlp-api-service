"""
Tulis bagian "Hasil" pada docs/METODOLOGI_EKSTRAKSI_ASPEK.md dari results.json.

Dipisah dari run_experiments.py agar tabel bisa dibuat ulang tanpa melatih apa
pun, dan agar isi dokumen selalu berasal dari angka yang benar-benar terukur.

Pemakaian:
    python scripts/render_results.py
"""

import argparse
import json
import os
from typing import Any, Dict, List

TEST_ORDER = [
    ('terma-test', 'TermA test (hotel, manusia)'),
    ('keps-test', 'KEPS test (perbankan, manusia)'),
    ('youtube', 'YouTube (pajak, proyeksi)'),
    ('youtube-heldout', 'YouTube held-out (bersih)'),
]

# Set uji yang berasal dari domain data latih kondisi tersebut. Angkanya sah
# sebagai kinerja, tetapi BUKAN ukuran generalisasi dan tidak boleh dibandingkan
# dengan sel lintas domain.
IN_DOMAIN = {
    'K1': {'youtube'},
    'K2': set(),
    'K3': {'terma-test'},
    'K4': {'terma-test', 'keps-test'},
    'K5': {'terma-test', 'keps-test'},
    'K6': {'terma-test', 'youtube-heldout'},
}
# K1 lebih parah dari sekadar in-domain: 708 dari 885 dokumen set uji YouTube
# adalah data latihnya sendiri, jadi angkanya bahkan bukan uji tertahan.
TRAIN_OVERLAP = {'K1': {'youtube'}, 'K6': {'youtube'}}


def fmt(value: Any) -> str:
    return f'{value:.4f}'.replace('.', ',') if isinstance(value, (int, float)) else '—'


def render(results: Dict[str, Any]) -> str:
    lines: List[str] = []

    lines.append('## 6. Hasil')
    lines.append('')
    # Penanda: hanya blok di antara AUTO:BEGIN dan AUTO:END yang ditimpa,
    # supaya pembahasan yang ditulis manual di bawahnya tidak ikut terhapus.
    lines.append('<!-- AUTO:BEGIN - dihasilkan scripts/render_results.py, jangan disunting -->')
    lines.append('')
    lines.append('Dihasilkan oleh `scripts/run_experiments.py`, dirender dengan')
    lines.append('`scripts/render_results.py`. Seluruh kondisi diuji pada set uji yang sama.')
    lines.append('')

    # ── Tabel utama ─────────────────────────────────────────────────────────
    lines.append('### 6.1 F1 span-level (batas persis)')
    lines.append('')
    header = '| Kondisi latih | ' + ' | '.join(label for _, label in TEST_ORDER) + ' |'
    lines.append(header)
    lines.append('|---|' + '---:|' * len(TEST_ORDER))

    for cid in sorted(results):
        entry = results[cid]
        cond = entry.get('condition', {})
        row = f"| **{cid}** {cond.get('label', '')} |"
        for tid, _ in TEST_ORDER:
            v = entry.get('eval', {}).get(tid, {}).get('span', {}).get('f1')
            cell = fmt(v)
            if tid in TRAIN_OVERLAP.get(cid, set()):
                cell += ' ²'
            elif tid in IN_DOMAIN.get(cid, set()):
                cell += ' ¹'
            row += f' {cell} |'
        lines.append(row)

    lines.append('')
    lines.append('¹ In-domain: domain set uji ini ada di data latih kondisi tersebut.')
    lines.append('  Angkanya sah sebagai kinerja, tetapi **bukan** ukuran generalisasi.')
    lines.append('')
    lines.append('² Tumpang tindih data latih: 708 dari 885 dokumen set uji YouTube adalah')
    lines.append('  data latih K1, sehingga angkanya bahkan bukan uji tertahan. Nilai')
    lines.append('  in-domain K1 yang sah adalah F1 validasi pada tabel 6.2.')
    lines.append('')

    # ── Ringkasan pelatihan ─────────────────────────────────────────────────
    lines.append('### 6.2 Ringkasan pelatihan')
    lines.append('')
    lines.append('| Kondisi | train / val | split | epoch terbaik | F1 val sebelum | F1 val sesudah |')
    lines.append('|---|---:|---|---:|---:|---:|')
    for cid in sorted(results):
        t = results[cid].get('train')
        if not t:
            continue
        lines.append(
            f"| {cid} | {t['train_size']} / {t['val_size']} | `{t['split']}` | "
            f"{t['best_epoch']} | {fmt(t['val_f1_before'])} | {fmt(t['val_f1_after'])} |"
        )
    lines.append('')

    # ── Rincian per epoch ───────────────────────────────────────────────────
    lines.append('### 6.3 Kurva pelatihan per epoch (F1 validasi)')
    lines.append('')
    lines.append('| Kondisi | epoch 1 | epoch 2 | epoch 3 |')
    lines.append('|---|---:|---:|---:|')
    for cid in sorted(results):
        t = results[cid].get('train')
        if not t:
            continue
        cells = {e['epoch']: e['f1'] for e in t.get('epoch_history', [])}
        row = f'| {cid} |'
        for ep in (1, 2, 3):
            row += f' {fmt(cells.get(ep))} |'
        lines.append(row)
    lines.append('')

    # ── Token-level ─────────────────────────────────────────────────────────
    lines.append('### 6.4 F1 token-level (metrik longgar, sebagai pembanding)')
    lines.append('')
    lines.append('| Kondisi latih | ' + ' | '.join(label for _, label in TEST_ORDER) + ' |')
    lines.append('|---|' + '---:|' * len(TEST_ORDER))
    for cid in sorted(results):
        row = f'| {cid} |'
        for tid, _ in TEST_ORDER:
            v = results[cid].get('eval', {}).get(tid, {}).get('token', {}).get('f1')
            row += f' {fmt(v)} |'
        lines.append(row)
    lines.append('')
    lines.append('<!-- AUTO:END -->')
    lines.append('')

    return '\n'.join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', default='data/experiments/results.json')
    ap.add_argument('--doc', default='docs/METODOLOGI_EKSTRAKSI_ASPEK.md')
    args = ap.parse_args()

    with open(args.results, encoding='utf-8') as f:
        results = json.load(f)

    section = render(results)

    with open(args.doc, encoding='utf-8') as f:
        doc = f.read()

    start = doc.index('## 6. Hasil')
    end_marker = '<!-- AUTO:END -->\n'
    if end_marker in doc:
        # Hanya blok otomatis yang ditimpa. Pembahasan yang ditulis manual di
        # bawah penanda dipertahankan - sebelumnya seluruh bab 6 tergantikan
        # dan analisis yang sudah ditulis hilang tanpa peringatan.
        end = doc.index(end_marker) + len(end_marker)
        doc = doc[:start] + section + doc[end:]
    else:
        end = doc.index('## 7. Reproduksi')
        doc = doc[:start] + section + '\n---\n\n' + doc[end:]

    with open(args.doc, 'w', encoding='utf-8') as f:
        f.write(doc)

    print(section)
    print(f'\nditulis ke {args.doc}')


if __name__ == '__main__':
    main()
