"""
Jalankan matriks eksperimen fine-tuning ekstraksi aspek dan kumpulkan hasilnya.

PERTANYAAN YANG DIJAWAB
-----------------------
Apakah model mempelajari POLA ekstraksi aspek, atau menghafal daftar istilah
yang kebetulan ada di data latihnya?

Cara menjawabnya adalah dengan melatih pada beberapa sumber label yang berbeda,
lalu menguji SEMUANYA pada set uji yang sama - termasuk domain yang tidak pernah
dilihat saat latih. Model yang hanya menghafal kosakata akan runtuh di domain
lain; model yang menangkap pola tidak.

KONDISI LATIH
-------------
K1 leksikon-youtube : label hasil proyeksi leksikon (komentar YouTube pajak)
K2 leksikon-berita  : label hasil penyaringan+proyeksi anotasi LLM (berita)
K3 terma            : IndoNLU TermA, LABEL MANUSIA, ulasan hotel
K4 terma+keps       : TermA + KEPS, LABEL MANUSIA, dua domain
K5 K4 + augmentasi  : K4 + contoh negatif + substitusi istilah + bobot kelas
K6 terma+youtube    : TermA (pola + definisi ABSA) + data domain produksi

SET UJI (sama untuk semua kondisi)
----------------------------------
terma-test   1000 dok, label manusia, ulasan hotel
keps-test     247 dok, label manusia, tweet perbankan
youtube        885 dok, label proyeksi, komentar YouTube

Setiap proses latih/uji dijalankan sebagai subproses terpisah agar memori bersih
di antara kondisi (IndoBERT ~440MB per model).

Pemakaian:
    python scripts/run_experiments.py                 # semua
    python scripts/run_experiments.py --only K3 K4    # sebagian
    python scripts/run_experiments.py --eval-only     # lewati pelatihan
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

CONDITIONS: List[Dict[str, Any]] = [
    {
        'id': 'K1',
        'name': 'leksikon-youtube',
        'label': 'Leksikon (YouTube pajak)',
        'data': 'data/aspect_dataset_retrain.json',
        'eval_data': None,
        'model_dir': 'models_exp/k1_youtube',
        'human_labels': False,
    },
    {
        'id': 'K2',
        'name': 'leksikon-berita',
        'label': 'Leksikon+LLM (berita)',
        'data': 'data/aspect_news_retrain.json',
        'eval_data': None,
        'model_dir': 'models_exp/k2_berita',
        'human_labels': False,
    },
    {
        'id': 'K3',
        'name': 'terma',
        'label': 'TermA (label manusia)',
        'data': 'data/external/terma/train_retrain.json',
        'eval_data': 'data/external/terma/valid_retrain.json',
        'model_dir': 'models_exp/k3_terma',
        'human_labels': True,
    },
    {
        'id': 'K6',
        'name': 'terma+youtube',
        'label': 'TermA + YouTube in-domain',
        'data': 'data/external/k6_train_retrain.json',
        'eval_data': 'data/external/terma/valid_retrain.json',
        'model_dir': 'models_exp/k6_production',
        'human_labels': True,
    },
    {
        'id': 'K5',
        'name': 'terma+keps+augmentasi',
        'label': 'K4 + negatif + substitusi + bobot kelas',
        'data': 'data/external/k5_train_retrain.json',
        'eval_data': 'data/external/combined_valid_retrain.json',
        'model_dir': 'models_exp/k5_augmented',
        'human_labels': True,
        'extra_args': ['--class-weights'],
    },
    {
        'id': 'K4',
        'name': 'terma+keps',
        'label': 'TermA+KEPS (manusia, 2 domain)',
        'data': 'data/external/combined_train_retrain.json',
        'eval_data': 'data/external/combined_valid_retrain.json',
        'model_dir': 'models_exp/k4_combined',
        'human_labels': True,
    },
]

TEST_SETS = [
    {'id': 'terma-test', 'gold': 'data/external/terma/test_gold.json',
     'label': 'TermA test (hotel, manusia)'},
    {'id': 'keps-test', 'gold': 'data/external/keps/test_gold.json',
     'label': 'KEPS test (perbankan, manusia)'},
    {'id': 'youtube', 'gold': 'data/external/youtube/test_gold.json',
     'label': 'YouTube (pajak, proyeksi)'},
    {'id': 'youtube-heldout', 'gold': 'data/external/youtube/heldout_gold.json',
     'label': 'YouTube held-out (177 dok, tak pernah dilatih siapa pun)'},
]


def run(cmd: List[str], log_path: str) -> str:
    """Jalankan subproses, simpan seluruh keluaran, kembalikan stdout."""
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding='utf-8', errors='replace')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(proc.stdout or '')
        f.write('\n--- stderr ---\n')
        f.write(proc.stderr or '')
    if proc.returncode != 0:
        print(f'    GAGAL (exit {proc.returncode}) - lihat {log_path}')
    return proc.stdout or ''


def parse_eval(output: str) -> Optional[Dict[str, float]]:
    """Ambil angka P/R/F1 span-level dan token-level dari keluaran evaluasi."""
    def grab(section: str) -> Optional[Dict[str, float]]:
        m = re.search(
            section + r'.*?precision\s+([\d.]+)\s+recall\s+([\d.]+)\s+F1\s+([\d.]+)',
            output, re.S)
        if not m:
            return None
        return {'precision': float(m.group(1)), 'recall': float(m.group(2)),
                'f1': float(m.group(3))}

    span = grab(r'span-level')
    token = grab(r'token-level')
    if span is None:
        return None
    return {'span': span, 'token': token or {}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='*', default=None, help='id kondisi, mis. K3 K4')
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--resume', action='store_true',
                    help='lanjutkan dari checkpoint yang ada (default: mulai dari model dasar)')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--learning-rate', type=float, default=3e-5)
    ap.add_argument('--out-dir', default='data/experiments')
    args = ap.parse_args()

    order = {c['id']: i for i, c in enumerate(CONDITIONS)}
    conditions = sorted(
        (c for c in CONDITIONS if args.only is None or c['id'] in args.only),
        key=lambda c: order[c['id']],
    )

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, 'results.json')

    results: Dict[str, Any] = {}
    if os.path.exists(results_path):
        with open(results_path, encoding='utf-8') as f:
            results = json.load(f)

    for cond in conditions:
        cid = cond['id']
        print(f"\n{'=' * 70}\n{cid}  {cond['label']}\n{'=' * 70}")
        entry = results.setdefault(cid, {'condition': cond})
        entry['condition'] = cond

        ckpt = os.path.join(cond['model_dir'], 'aspect_retrained.pt')

        if not args.eval_only:
            # Setiap kondisi HARUS mulai dari IndoBERT dasar. AspectService
            # menemukan sendiri checkpoint di MODEL_PATH dan melanjutkannya
            # (perilaku yang benar untuk active learning, salah untuk
            # eksperimen: kondisi jadi mewarisi bobot run sebelumnya dan
            # metrik "sebelum" bukan lagi model dasar).
            if os.path.isdir(cond['model_dir']) and not args.resume:
                shutil.rmtree(cond['model_dir'])
                print(f"  checkpoint lama dihapus: {cond['model_dir']}")

            print(f"  melatih ... (data: {cond['data']})")
            cmd = [PY, '-u', 'scripts/train_aspect.py',
                   '--data', cond['data'],
                   '--model-path', cond['model_dir'],
                   '--epochs', str(args.epochs),
                   '--learning-rate', str(args.learning_rate),
                   '--report', os.path.join(args.out_dir, f'{cid}_train.json')]
            if cond['eval_data']:
                cmd += ['--eval-data', cond['eval_data']]
            cmd += cond.get('extra_args', [])

            started = time.time()
            out = run(cmd, os.path.join(args.out_dir, f'{cid}_train.log'))
            entry['train_minutes'] = round((time.time() - started) / 60, 1)

            report_path = os.path.join(args.out_dir, f'{cid}_train.json')
            if os.path.exists(report_path):
                with open(report_path, encoding='utf-8') as f:
                    rep = json.load(f)
                entry['train'] = {
                    'train_size': rep['train_size'], 'val_size': rep['val_size'],
                    'split': rep['split'], 'best_epoch': rep['best_epoch'],
                    'epoch_history': rep['epoch_history'],
                    'val_f1_before': rep['metrics_before']['f1'],
                    'val_f1_after': rep['metrics_after']['f1'],
                }
                print(f"    val F1 {rep['metrics_before']['f1']:.4f} -> "
                      f"{rep['metrics_after']['f1']:.4f} "
                      f"(epoch terbaik {rep['best_epoch']}, {entry['train_minutes']} menit)")
            else:
                print('    laporan latih tidak ditemukan')

        if not os.path.exists(ckpt):
            print(f'    checkpoint tidak ada: {ckpt} - evaluasi dilewati')
            continue

        entry.setdefault('eval', {})
        for ts in TEST_SETS:
            print(f"  evaluasi pada {ts['label']} ...")
            out = run([PY, '-u', 'scripts/evaluate_aspect.py',
                       '--gold', ts['gold'], '--checkpoint', ckpt],
                      os.path.join(args.out_dir, f"{cid}_eval_{ts['id']}.log"))
            parsed = parse_eval(out)
            if parsed:
                entry['eval'][ts['id']] = parsed
                print(f"    span F1 {parsed['span']['f1']:.4f}   "
                      f"token F1 {parsed['token'].get('f1', 0):.4f}")
            else:
                print('    gagal membaca hasil evaluasi')

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # ── Tabel ringkas ───────────────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print('MATRIKS HASIL  (F1 span-level, batas persis)')
    print('=' * 78)
    header = f"{'kondisi latih':32}" + ''.join(f"{t['id']:>15}" for t in TEST_SETS)
    print(header)
    print('-' * len(header))
    for cid, entry in results.items():
        cond = entry['condition']
        row = f"{cid + ' ' + cond['label']:32}"
        for ts in TEST_SETS:
            v = entry.get('eval', {}).get(ts['id'], {}).get('span', {}).get('f1')
            row += f"{(f'{v:.4f}' if v is not None else '-'):>15}"
        print(row)

    print(f"\n  hasil lengkap -> {results_path}")


if __name__ == '__main__':
    main()
