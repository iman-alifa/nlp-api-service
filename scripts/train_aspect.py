"""
Fine-tuning model ekstraksi aspek dari berkas payload retraining.

Membungkus AspectService.retrain() supaya seluruh disiplinnya ikut terpakai:
seed tetap, split stratifikasi, metrik diukur sebelum & sesudah, dan checkpoint
ditolak bila F1 token aspek menurun.

Berkas masukan berformat sama dengan body POST /api/retrain/aspect:
    [{"text": ..., "aspects": [...], "spans": [{"start": .., "end": ..}]}, ...]
Kunci "spans" bersifat opsional; bila ada, offset anotator dipakai apa adanya
sehingga label manusia tidak dirusak pencocokan string.

Contoh:
    # latih pada data berlabel manusia (IndoNLU TermA)
    python scripts/train_aspect.py \
        --data data/external/terma/train_retrain.json \
        --model-path models_terma --epochs 3

    # lalu ukur generalisasi ke domain lain
    python scripts/evaluate_aspect.py \
        --gold data/external/keps/test_gold.json \
        --checkpoint models_terma/aspect_retrained.pt
"""

import argparse
import asyncio
import json
import os
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='payload retraining (JSON)')
    ap.add_argument('--model-path', default=None,
                    help='folder penyimpanan checkpoint; default settings.model_path')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--learning-rate', type=float, default=3e-5)
    ap.add_argument('--force', action='store_true',
                    help='simpan checkpoint walau metrik menurun')
    ap.add_argument('--eval-data', default=None,
                    help='payload set validasi terpisah (split resmi dataset)')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--class-weights', action='store_true',
                    help='bobot kelas berbanding terbalik frekuensi pada loss BIO')
    ap.add_argument('--progress-file', default='data/experiments/progress.log',
                    help='berkas kemajuan real-time; pantau dengan tail -f')
    ap.add_argument('--report', default=None, help='tulis laporan JSON ke path ini')
    args = ap.parse_args()

    # MODEL_PATH harus disetel sebelum app.config diimpor (dibaca saat import).
    if args.model_path:
        os.environ['MODEL_PATH'] = os.path.abspath(args.model_path)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.services.aspect_service import AspectService

    with open(args.data, encoding='utf-8') as f:
        data = json.load(f)

    eval_data = None
    if args.eval_data:
        with open(args.eval_data, encoding='utf-8') as f:
            eval_data = json.load(f)

    gold = sum(1 for d in data if d.get('spans'))
    print(f'sampel        : {len(data)}')
    print(f'  dengan offset emas : {gold}')
    print(f'validasi      : {len(eval_data) if eval_data else "split stratifikasi 80/20"}')
    print(f'epochs        : {args.epochs}   lr: {args.learning_rate}   batch: {args.batch_size}')

    # Kemajuan ditulis segera dan di-flush, supaya bisa dipantau saat berjalan.
    # Tanpa ini keluaran Python ter-buffer dan baru muncul setelah proses selesai.
    progress_path = args.progress_file
    if progress_path:
        os.makedirs(os.path.dirname(progress_path) or '.', exist_ok=True)

    label = os.path.basename(args.model_path or 'model')

    def report(info):
        if not progress_path:
            return
        stamp = time.strftime('%H:%M:%S')
        if info['phase'] == 'train':
            pct = info['step'] / info['steps'] * 100
            line = (f"[{stamp}] {label} epoch {info['epoch']}/{info['epochs']} "
                    f"batch {info['step']}/{info['steps']} ({pct:.0f}%) "
                    f"loss {info['running_loss']}")
        elif info['phase'] == 'eval':
            line = f"[{stamp}] {label} epoch {info['epoch']}/{info['epochs']} mengevaluasi..."
        else:
            line = (f"[{stamp}] {label} EPOCH {info['epoch']}/{info['epochs']} SELESAI  "
                    f"loss {info['train_loss']}  val P {info['precision']:.4f} "
                    f"R {info['recall']:.4f} F1 {info['f1']:.4f}")
        with open(progress_path, 'a', encoding='utf-8') as f:
            f.write(line + os.linesep)
            f.flush()
        print(line, flush=True)

    report({'phase': 'eval', 'epoch': 0, 'epochs': args.epochs})

    svc = AspectService()
    started = time.time()
    result = asyncio.run(svc.retrain(
        data,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        force=args.force,
        eval_data=eval_data,
        batch_size=args.batch_size,
        progress_callback=report,
        use_class_weights=args.class_weights,
    ))
    minutes = (time.time() - started) / 60

    print('\n' + '=' * 58)
    print('HASIL FINE-TUNING')
    print('=' * 58)
    print(f'  durasi        : {minutes:.1f} menit')
    print(f'  seed          : {result["seed"]}')
    print(f'  train / val   : {result["train_size"]} / {result["val_size"]} ({result["split"]})')
    print(f'  epoch terbaik : {result["best_epoch"]} dari {args.epochs}')
    for e in result['epoch_history']:
        print(f'      epoch {e["epoch"]}  loss {e["train_loss"]:.4f}  '
              f'val P {e["precision"]:.4f}  R {e["recall"]:.4f}  F1 {e["f1"]:.4f}')
    print(f'  train loss    : {result["train_loss"]}')
    print(f'  F1 sebelum    : {result["metrics_before"]["f1"]}')
    print(f'  F1 sesudah    : {result["metrics_after"]["f1"]}')
    print(f'  delta         : {result["f1_delta"]:+}')
    print(f'  disimpan      : {result["saved"]}  -> {result["model_path"]}')

    bio = {k: v for k, v in result['bio_report'].items() if k != 'unmatched_examples'}
    print(f'  bio_report    : {bio}')

    if args.report:
        with open(args.report, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f'\n  laporan -> {args.report}')


if __name__ == '__main__':
    main()
