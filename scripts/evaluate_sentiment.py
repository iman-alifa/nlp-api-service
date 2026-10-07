"""
Evaluasi modul analisis sentimen terhadap data berlabel manusia.

Modul aspek dan pemodelan topik masing-masing punya harness pengukuran
(`evaluate_aspect.py`, `eval_topics.py`), sementara sentimen sama sekali belum
pernah diukur - padahal ia jalur yang paling banyak dipakai antarmuka. Skrip ini
menutup lubang itu.

Yang diukur adalah JALUR PRODUKSI, yaitu `SentimentService.analyze()` lengkap
dengan preprocessing, bukan model mentah. Itu penting karena kebijakan
preprocessing modul ini tidak netral (lihat `_sanitize_config`): ia sengaja
mematikan stemming/stopword removal, dan klaim tersebut harus bisa diuji dengan
angka, bukan dengan beberapa kalimat pilihan.

Dataset acuan:
- `data/external/smsa/test_gold.json` - SmSA / IndoNLU (Prosa), 500 dokumen.
  CATATAN PENTING: model dasar `mdhugol/indonesia-bert-sentiment-classification`
  di-fine-tune pada SmSA train, sehingga ini adalah uji IN-DOMAIN. Angkanya
  batas atas, bukan bukti generalisasi - persis seperti F1 0,97 in-domain pada
  modul aspek yang jatuh ke 0,029 lintas domain.

Contoh:
    python scripts/evaluate_sentiment.py
    python scripts/evaluate_sentiment.py --ablation
    python scripts/evaluate_sentiment.py --gold data/external/smsa/valid_gold.json
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch  # noqa: F401  (torch harus diimpor sebelum nltk - WinError 1114)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.sentiment_service import LABEL_MAP, LABEL_TO_ID, SentimentService  # noqa: E402
from app.utils.training import (  # noqa: E402
    classification_metrics,
    expected_calibration_error,
    fit_temperature,
)

# Konfigurasi yang benar-benar dikirim Laravel dari formulir analisis.
LARAVEL_CONFIG = {
    'case_folding': True, 'remove_punctuation': True, 'remove_numbers': False,
    'remove_stopwords': True, 'stemming': True, 'lemmatization': False,
    'custom_stopwords': [],
}

# Varian untuk ablasi kebijakan preprocessing. `_sanitize_config` memaksa
# stemming/stopword mati pada ketiganya; kolom "efektif" pada laporan
# menunjukkan apa yang benar-benar terjadi.
ABLATIONS: Dict[str, Optional[Dict[str, Any]]] = {
    'tanpa-preprocessing': None,
    'huruf+tanda-baca': {
        'case_folding': True, 'remove_punctuation': True, 'remove_numbers': False,
        'remove_stopwords': False, 'stemming': False, 'lemmatization': False,
        'custom_stopwords': [], 'fix_typos': False, 'normalize_slang': False,
    },
    'normalisasi-penuh': {
        'case_folding': True, 'remove_punctuation': True, 'remove_numbers': False,
        'remove_stopwords': False, 'stemming': False, 'lemmatization': False,
        'custom_stopwords': [],
    },
    # Sengaja MENGAKTIFKAN stemming + stopword removal untuk membuktikan bahwa
    # `_sanitize_config` benar-benar mematikannya: hasilnya harus identik dengan
    # 'normalisasi-penuh'. Bila suatu saat berbeda, pengaman itu bocor.
    'seperti-laravel': LARAVEL_CONFIG,
}


_ALIAS_LABEL = {
    'positive': 'positive', 'positif': 'positive',
    'neutral': 'neutral', 'netral': 'neutral',
    'negative': 'negative', 'negatif': 'negative',
}


def _terapkan_peta_label(service: SentimentService, model_name: str) -> None:
    """
    Selaraskan LABEL_MAP dengan config model pembanding.

    Urutan label berbeda-beda antar model dan TIDAK boleh diasumsikan: salah di
    sini membalik seluruh hasil tanpa memunculkan galat apa pun. Model yang
    config-nya masih `LABEL_0/1/2` dibiarkan memakai peta bawaan proyek, yang
    hanya sudah diverifikasi untuk model produksi.
    """
    import app.services.sentiment_service as modul

    raw = getattr(service.model.config, 'id2label', None) or {}
    peta = {int(k): _ALIAS_LABEL.get(str(v).strip().lower()) for k, v in raw.items()}

    if len(peta) == 3 and all(peta.values()):
        modul.LABEL_MAP.clear()
        modul.LABEL_MAP.update(peta)
        print(f'peta label {model_name}: {dict(sorted(peta.items()))}')
    else:
        print(f'peta label {model_name} tidak informatif ({raw}); memakai bawaan proyek')


def load_gold(path: str, limit: int = 0) -> List[Dict[str, str]]:
    """Baca file gold {text,label}; label wajib ada di LABEL_TO_ID."""
    with open(path, encoding='utf-8') as handle:
        rows = json.load(handle)

    clean = [
        r for r in rows
        if r.get('text', '').strip() and r.get('label') in LABEL_TO_ID
    ]
    dropped = len(rows) - len(clean)
    if dropped:
        print(f'  {dropped} baris dibuang (teks kosong atau label tak dikenal)')
    return clean[:limit] if limit else clean


def confidence_buckets(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Akurasi per pita confidence - kalibrasi, bukan sekadar rata-rata.

    Berguna untuk antarmuka: bila akurasi pada pita rendah jauh di bawah pita
    tinggi, confidence layak dipakai sebagai penyaring baris yang perlu dikoreksi
    manusia (umpan balik active learning). Bila datar, confidence tidak informatif.
    """
    edges = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.01)]
    out = []
    for low, high in edges:
        subset = [r for r in records if low <= r['confidence'] < high]
        if not subset:
            continue
        benar = sum(1 for r in subset if r['pred'] == r['gold'])
        out.append({
            'range': f'{low:.2f}-{high:.2f}',
            'n': len(subset),
            'share': round(len(subset) / len(records), 4),
            'accuracy': round(benar / len(subset), 4),
        })
    return out


async def run_once(service: SentimentService, gold: List[Dict[str, str]],
                   config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Jalankan satu konfigurasi penuh dan kembalikan metrik + catatan per baris."""
    texts = [r['text'] for r in gold]

    started = time.time()
    result = await service.analyze(texts=texts, preprocessing_config=config)
    elapsed = time.time() - started

    predictions = result['predictions']
    if len(predictions) != len(texts):
        raise AssertionError(
            f'Kontrak penjajaran rusak: {len(texts)} teks masuk, '
            f'{len(predictions)} prediksi keluar'
        )

    y_true = [LABEL_TO_ID[r['label']] for r in gold]
    y_pred = [LABEL_TO_ID.get(p['sentiment'], LABEL_TO_ID['neutral']) for p in predictions]

    records = [
        {
            'text': gold[i]['text'],
            'gold': y_true[i],
            'pred': y_pred[i],
            'confidence': predictions[i].get('confidence', 0.0),
            'method': predictions[i].get('method', '?'),
        }
        for i in range(len(gold))
    ]

    metrics = classification_metrics(y_true, y_pred, LABEL_MAP)

    # Kalibrasi: seberapa jujur angka confidence yang ditampilkan antarmuka.
    confidences = [r['confidence'] for r in records]
    correct = [r['pred'] == r['gold'] for r in records]
    metrics['ece'] = expected_calibration_error(confidences, correct)
    metrics['avg_confidence'] = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    metrics['confidence_gap'] = round(metrics['avg_confidence'] - metrics['accuracy'], 4)
    metrics['temperature'] = getattr(service, 'temperature', 1.0)

    metrics['seconds'] = round(elapsed, 1)
    metrics['docs_per_second'] = round(len(texts) / elapsed, 1) if elapsed else 0.0
    metrics['calibration'] = confidence_buckets(records)
    metrics['methods'] = {
        m: sum(1 for r in records if r['method'] == m)
        for m in sorted({r['method'] for r in records})
    }
    metrics['review_yield'] = review_yield(records)
    return {'metrics': metrics, 'records': records}


def review_yield(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Berapa persen kesalahan yang tertangkap bila meninjau baris di bawah ambang.

    Inilah angka yang menentukan `settings.sentiment_review_threshold`. Ambang
    yang baik menangkap sebagian besar kesalahan sambil menyisakan beban tinjau
    yang wajar; tanpa tabel ini pemilihannya hanya tebakan.

    Kolom `lift` = (porsi kesalahan tertangkap) / (porsi baris ditinjau).
    Nilai 1,0 berarti sama saja dengan meninjau acak.
    """
    total = len(records)
    total_salah = sum(1 for r in records if r['pred'] != r['gold'])
    if not total or not total_salah:
        return []

    out = []
    for threshold in (0.50, 0.70, 0.80, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99):
        ditinjau = [r for r in records if r['confidence'] < threshold]
        tertangkap = sum(1 for r in ditinjau if r['pred'] != r['gold'])
        porsi_tinjau = len(ditinjau) / total
        out.append({
            'threshold': threshold,
            'reviewed': len(ditinjau),
            'reviewed_share': round(porsi_tinjau, 4),
            'errors_caught': tertangkap,
            'error_recall': round(tertangkap / total_salah, 4),
            'lift': round((tertangkap / total_salah) / porsi_tinjau, 2) if porsi_tinjau else 0.0,
        })
    return out


def print_report(name: str, metrics: Dict[str, Any]) -> None:
    print(f'\n=== {name} ===')
    print(f"  accuracy      {metrics['accuracy']:.4f}")
    print(f"  macro F1      {metrics['macro_f1']:.4f}")
    print(f"  weighted F1   {metrics['weighted_f1']:.4f}")
    print(f"  ECE           {metrics['ece']:.4f}  (T={metrics['temperature']}, "
          f"rata2 confidence {metrics['avg_confidence']:.4f}, "
          f"selisih {metrics['confidence_gap']:+.4f})")
    print(f"  {metrics['seconds']} detik ({metrics['docs_per_second']} dok/detik)"
          f" | metode: {metrics['methods']}")

    print(f"\n  {'kelas':10} {'prec':>7} {'rec':>7} {'F1':>7} {'n':>6}")
    for label, row in metrics['per_class'].items():
        print(f"  {label:10} {row['precision']:>7.4f} {row['recall']:>7.4f} "
              f"{row['f1']:>7.4f} {row['support']:>6}")

    matrix = metrics.get('confusion_matrix')
    labels = metrics.get('confusion_labels', [])
    if matrix and labels:
        print('\n  confusion (baris=gold, kolom=prediksi)')
        print('    ' + ' ' * 10 + ' '.join(f'{c[:5]:>5}' for c in labels))
        for label, row in zip(labels, matrix):
            print(f"    {label:10} " + ' '.join(f'{v:>5}' for v in row))

    if metrics.get('review_yield'):
        print(f"\n  antrean tinjau — berapa kesalahan tertangkap per ambang")
        print(f"  {'ambang':>7} {'ditinjau':>9} {'porsi':>7} {'kesalahan':>10} "
              f"{'recall':>8} {'lift':>6}")
        for row in metrics['review_yield']:
            print(f"  {row['threshold']:>7.2f} {row['reviewed']:>9} "
                  f"{row['reviewed_share']:>7.3f} {row['errors_caught']:>10} "
                  f"{row['error_recall']:>8.3f} {row['lift']:>6.2f}")

    if metrics.get('calibration'):
        print(f"\n  {'confidence':12} {'n':>6} {'porsi':>7} {'akurasi':>8}")
        for bucket in metrics['calibration']:
            print(f"  {bucket['range']:12} {bucket['n']:>6} "
                  f"{bucket['share']:>7.3f} {bucket['accuracy']:>8.4f}")


def dump_errors(records: List[Dict[str, Any]], path: str, limit: int) -> None:
    """Simpan kesalahan berkeyakinan tinggi - bahan analisis kualitatif skripsi."""
    salah = [r for r in records if r['pred'] != r['gold']]
    salah.sort(key=lambda r: -r['confidence'])
    payload = [
        {
            'text': r['text'][:400],
            'gold': LABEL_MAP[r['gold']],
            'pred': LABEL_MAP[r['pred']],
            'confidence': r['confidence'],
        }
        for r in salah[:limit]
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f'\n{len(salah)} kesalahan; {len(payload)} teratas tersimpan ke {path}')


async def fit_temperature_on(service: SentimentService, valid_path: str,
                            test_path: str) -> Dict[str, Any]:
    """
    Cari suhu kalibrasi pada set VALIDASI, lalu laporkan efeknya pada set UJI.

    Ini yang menghasilkan angka `settings.sentiment_temperature`. Suhu WAJIB
    dicari pada set validasi dan dilaporkan pada set uji yang tidak dipakai
    mencarinya - kalau tidak, penurunan ECE hanyalah overfitting.
    """
    from app.preprocessing.text_cleaner import TextCleaner

    if not service.model or not service.tokenizer:
        raise RuntimeError('Kalibrasi butuh bobot model; jalur cadangan tidak punya logit')

    cleaner = TextCleaner(SentimentService._sanitize_config(LARAVEL_CONFIG))

    def logits_and_labels(path: str):
        rows = load_gold(path)
        texts = cleaner.clean_texts([r['text'] for r in rows])
        chunks = []
        for i in range(0, len(texts), 128):
            batch = service.tokenizer(
                texts[i:i + 128], return_tensors='pt',
                truncation=True, max_length=512, padding=True,
            ).to(service.device)
            with torch.no_grad():
                chunks.append(service.model(**batch).logits.detach().cpu())
        labels = torch.tensor([LABEL_TO_ID[r['label']] for r in rows])
        return torch.cat(chunks), labels

    z_valid, y_valid = logits_and_labels(valid_path)
    z_test, y_test = logits_and_labels(test_path)

    temperature = fit_temperature(z_valid, y_valid)
    print(f'\nT optimal dari {valid_path} (n={len(y_valid)}): {temperature}')

    def score(z, y, t):
        probs = torch.softmax(z / t, dim=1)
        conf, pred = probs.max(1)
        correct = (pred == y)
        return (
            expected_calibration_error(conf.tolist(), correct.tolist()),
            round(correct.float().mean().item(), 4),
            round(conf.mean().item(), 4),
        )

    print(f"\n  {'set':6} {'T':>8} {'ECE':>8} {'akurasi':>9} {'confid':>8}")
    hasil: Dict[str, Any] = {'temperature': temperature}
    for nama, z, y in (('valid', z_valid, y_valid), ('test', z_test, y_test)):
        for t in (1.0, temperature):
            ece, acc, conf = score(z, y, t)
            label = 'tanpa' if t == 1.0 else 'dengan'
            hasil[f'{nama}_{label}'] = {'ece': ece, 'accuracy': acc, 'avg_confidence': conf}
            print(f'  {nama:6} {t:>8.4f} {ece:>8.4f} {acc:>9.4f} {conf:>8.4f}')

    print('\nAkurasi HARUS identik pada kedua suhu - temperature scaling menjaga argmax.')
    return hasil


async def main_async(args: argparse.Namespace) -> None:
    print(f'Memuat {args.gold}')
    gold = load_gold(args.gold, args.limit)
    from collections import Counter
    print(f'{len(gold)} dokumen | distribusi: {dict(Counter(r["label"] for r in gold))}')

    # `--model` menguji model pembanding lewat JALUR PRODUKSI yang sama, tanpa
    # mengubah `settings.sentiment_base_model`. Dengan begitu kandidat bisa
    # diukur lengkap sebelum diputuskan menggantikan model yang sedang dipakai.
    service = SentimentService(model_path=args.model) if args.model else SentimentService()
    service._ensure_loaded()

    if args.model:
        _terapkan_peta_label(service, args.model)

    if args.temperature is not None:
        # Suhu milik BOBOT tertentu; kandidat harus memakai suhunya sendiri,
        # bukan suhu model produksi.
        service.temperature = args.temperature
        print(f'suhu dipaksa ke {args.temperature}')

    print(f'model diuji: {service.model_name} | T={service.temperature}')

    if args.fit_temperature:
        hasil = await fit_temperature_on(service, args.valid, args.gold)
        # Tujuannya BUKAN args.out. Kalibrasi dan evaluasi menghasilkan bentuk
        # data yang berbeda, dan menulis keduanya ke berkas yang sama membuat
        # urutan perintah yang didokumentasikan menghapus hasil evaluasi utama
        # secara senyap - lalu tabel di naskah kehilangan sumbernya.
        tujuan = args.calibration_out
        os.makedirs(os.path.dirname(tujuan), exist_ok=True)
        with open(tujuan, 'w', encoding='utf-8') as handle:
            json.dump(hasil, handle, ensure_ascii=False, indent=2)
        print(f'\ntersimpan ke {args.out}')
        return

    if args.rule_based:
        # Paksa jalur cadangan supaya mutunya ikut terukur, bukan diasumsikan.
        # Ini yang benar-benar dipakai bila bobot model gagal diunduh.
        service._ensure_loaded()
        service.model = None
        service.tokenizer = None
        hasil = await run_once(service, gold, LARAVEL_CONFIG)
        print_report('jalur cadangan (daftar kata)', hasil['metrics'])
        payload = {'gold': args.gold, 'n': len(gold), 'mode': 'rule-based',
                   'metrics': hasil['metrics']}
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f'\ntersimpan ke {args.out}')
        return

    if args.ablation:
        ringkasan = {}
        terakhir: Dict[str, Any] = {}
        for name, config in ABLATIONS.items():
            hasil = await run_once(service, gold, config)
            print_report(name, hasil['metrics'])
            ringkasan[name] = {
                k: hasil['metrics'][k]
                for k in ('accuracy', 'macro_f1', 'weighted_f1', 'seconds')
            }
            terakhir = hasil

        print('\n=== RINGKASAN ABLASI ===')
        print(f"  {'konfigurasi':22} {'acc':>8} {'macroF1':>9} {'wF1':>8}")
        for name, row in ringkasan.items():
            print(f"  {name:22} {row['accuracy']:>8.4f} "
                  f"{row['macro_f1']:>9.4f} {row['weighted_f1']:>8.4f}")
        payload: Dict[str, Any] = {'gold': args.gold, 'n': len(gold), 'ablation': ringkasan}
    else:
        hasil = await run_once(service, gold, LARAVEL_CONFIG)
        print_report('jalur produksi (konfigurasi Laravel)', hasil['metrics'])
        terakhir = hasil
        payload = {
            'gold': args.gold, 'n': len(gold),
            'model': service.model_name,
            'metrics': hasil['metrics'],
        }

    if args.errors:
        dump_errors(terakhir['records'], args.errors, args.error_limit)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f'\ntersimpan ke {args.out}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', default='data/external/smsa/test_gold.json')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--model', default=None,
                        help='uji model pembanding tanpa mengubah setting produksi')
    parser.add_argument('--temperature', type=float, default=None,
                        help='paksa suhu kalibrasi (mis. hasil --fit-temperature)')
    parser.add_argument('--fit-temperature', action='store_true',
                        help='cari suhu kalibrasi di --valid, laporkan di --gold')
    parser.add_argument('--valid', default='data/external/smsa/valid_gold.json')
    parser.add_argument('--rule-based', action='store_true',
                        help='paksa jalur cadangan daftar kata')
    parser.add_argument('--ablation', action='store_true',
                        help='bandingkan beberapa kebijakan preprocessing')
    parser.add_argument('--errors', default='data/experiments/sentiment_errors.json')
    parser.add_argument('--calibration-out',
                        default='data/experiments/sentiment_calibration.json',
                        help='tujuan hasil --fit-temperature; sengaja berbeda dari '
                             '--out agar tidak menimpa hasil evaluasi utama')
    parser.add_argument('--error-limit', type=int, default=40)
    parser.add_argument('--out', default='data/experiments/sentiment_eval.json')
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
