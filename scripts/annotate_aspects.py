"""
Anotasi aspek otomatis setara-manusia untuk data fine-tuning IndoBERT.

MASALAH YANG DIPERBAIKI
-----------------------
Anotasi LLM sebelumnya (label_aspect.py -> gemini-1.5-flash) punya penilaian
per-kalimat yang umumnya masuk akal, tetapi tiga cacat sistematis:

1. Tidak konsisten (56% kemunculan terlewat). Penyebabnya `text.find(sub)` yang
   hanya menandai kemunculan PERTAMA, ditambah batch 200 kalimat per prompt
   sehingga model kehilangan konsistensi antar-kalimat. Untuk token
   classification ini supervisi yang saling bertentangan: kata yang sama
   diajarkan sebagai ASPECT di satu kalimat dan O di kalimat lain.
2. 12,4% span berkepala verba ("lolos", "menang", "merelokasi StarOne").
   Aspek adalah target opini, bukan predikat.
3. 6% span terlalu panjang (sampai 17 kata) dan memuat kurung penjelas.

PENDEKATAN
----------
Bukan menganotasi ulang dari nol, melainkan:

  panen kandidat  ->  normalisasi  ->  saring linguistik  ->  proyeksi konsisten

Penilaian semantik "apa yang layak disebut aspek" tetap berasal dari model yang
membaca tiap kalimat dalam konteks (seperti anotator manusia). Yang diganti
adalah bagian yang memang lemah pada anotasi otomatis: keajekan dan kaidah
bentuk. Hasilnya konsisten menurut konstruksi - setiap kemunculan istilah yang
sudah diterima ditandai di seluruh korpus.

TUJUAN: PEMBELAJARAN POLA
-------------------------
Karena istilahnya ribuan dan tersebar di banyak konteks sintaktis, model tidak
bisa menghafal daftar kata; ia harus mempelajari posisi dan bentuk frasa nomina
yang menjadi target pembicaraan. Itulah yang diinginkan agar model bisa dipakai
lintas domain.

Pemakaian:
    python scripts/annotate_aspects.py \
        --source C:/Downloads/label_studio_aspect_annotated.json \
        --target C:/Downloads/label_studio_preannotated.json \
        --out data/aspect_news
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory  # noqa: E402

STEMMER = StemmerFactory().create_stemmer()

# Awalan pembentuk verba. Kata berkepala salah satu awalan ini adalah predikat,
# bukan target opini -- kecuali bila terdaftar di NOUN_EXCEPTIONS.
VERBAL_PREFIXES = ('me', 'di', 'ter', 'ber')

# Nomina yang kebetulan berawalan seperti verba. Tanpa daftar ini, nama produk
# dan nomina lazim ikut terbuang (mis. 'Mentari' -> dikira men- + 'tari').
NOUN_EXCEPTIONS = {
    'mentari', 'menteri', 'merek', 'meja', 'media', 'medali', 'metode', 'mesin',
    'menit', 'medan', 'membran', 'memori', 'mental', 'menu', 'merger', 'mesir',
    'beras', 'berita', 'bendera', 'bensin', 'benua', 'bentuk', 'berkas',
    'terminal', 'teroris', 'termin', 'teras', 'tertib', 'terminologi',
    'direksi', 'direktur', 'dividen', 'divisi', 'diskon', 'diesel', 'diet',
    'dirut', 'dinas', 'diploma', 'disiplin', 'distribusi', 'distributor',
    'berkah', 'bearing', 'design', 'device',
}

# Adjektiva/adverbia evaluatif: ini KATA OPINI, bukan aspek. Dipisahkan supaya
# tidak ikut tertandai (kesalahan yang membuat model belajar melabeli sifat).
OPINION_WORDS = {
    'bagus', 'baik', 'buruk', 'jelek', 'hebat', 'mantap', 'keren', 'unggul',
    'kalah', 'menang', 'lolos', 'gagal', 'sukses', 'berhasil', 'hadir',
    'puas', 'kecewa', 'senang', 'sedih', 'marah', 'mahal', 'murah', 'cepat',
    'lambat', 'lama', 'baru', 'besar', 'kecil', 'tinggi', 'rendah', 'kuat',
    'lemah', 'mudah', 'sulit', 'susah', 'penting', 'utama', 'positif',
    'negatif', 'optimistis', 'pesimistis', 'yakin', 'ragu', 'siap', 'aman',
    'bahaya', 'sempurna', 'parah', 'tepat', 'salah', 'benar', 'wajar',
}

# Kata fungsi yang tidak boleh menjadi kepala frasa nomina.
FUNCTION_WORDS = {
    'yang', 'dan', 'atau', 'tetapi', 'namun', 'karena', 'sebab', 'jika',
    'kalau', 'untuk', 'kepada', 'dari', 'pada', 'di', 'ke', 'dengan', 'oleh',
    'dalam', 'atas', 'bawah', 'antara', 'tentang', 'terhadap', 'sebagai',
    'adalah', 'ialah', 'itu', 'ini', 'saya', 'kami', 'kita', 'anda', 'dia',
    'mereka', 'nya', 'akan', 'sudah', 'telah', 'sedang', 'masih', 'belum',
    'tidak', 'bukan', 'juga', 'hanya', 'saja', 'lebih', 'paling', 'sangat',
    'agar', 'supaya', 'hingga', 'sampai', 'sejak', 'setelah', 'sebelum',
    'ketika', 'saat', 'bahwa', 'para', 'se', 'per', 'lalu', 'kemudian',
    'tak', 'tanpa', 'meski', 'walau', 'walaupun', 'meskipun', 'seluruh',
    'semua', 'setiap', 'beberapa', 'banyak', 'sedikit', 'seorang', 'sebuah',
    'suatu', 'kedua', 'ketiga', 'sendiri', 'tersebut', 'demikian',
}

# Satuan waktu & ukuran: penanggalan bukan aspek.
TIME_UNITS = {
    'tahun', 'bulan', 'minggu', 'hari', 'jam', 'menit', 'detik', 'pekan',
    'januari', 'februari', 'maret', 'april', 'mei', 'juni', 'juli', 'agustus',
    'september', 'oktober', 'november', 'desember', 'senin', 'selasa', 'rabu',
    'kamis', 'jumat', 'sabtu', 'minggu',
}

MAX_TERM_WORDS = 4
MIN_TERM_CHARS = 3


# ── Normalisasi istilah ─────────────────────────────────────────────────────

def normalize_term(raw: str) -> Optional[str]:
    """
    Rapikan satu istilah kandidat menjadi frasa nomina yang bersih.

    Membuang kurung penjelas, tanda baca di ujung, angka di ekor, dan memangkas
    frasa yang kepanjangan. Mengembalikan None bila tidak tersisa yang berguna.

    Contoh:
        'kantor akuntan publik ( KAP )' -> 'kantor akuntan publik'
        'kinerja 2003'                  -> 'kinerja'
    """
    term = raw.strip()
    if not term:
        return None

    # Buang kurung penjelas beserta isinya
    term = re.sub(r'\s*[\(\[].*?[\)\]]\s*', ' ', term)
    term = re.sub(r'\s*[\(\[].*$', ' ', term)

    # Rapikan spasi di sekitar tanda hubung dan tanda baca ujung
    term = re.sub(r'\s*-\s*', '-', term)
    term = term.strip(' .,;:!?"\'`-–—')
    term = re.sub(r'\s+', ' ', term).strip()

    if not term:
        return None

    words = term.split()

    # Buang angka/tahun di ekor: 'kinerja 2003' -> 'kinerja'
    while words and re.fullmatch(r'[\d.,%]+', words[-1]):
        words.pop()
    # Buang satuan waktu di ekor: 'kinerja tahun' -> 'kinerja'
    while len(words) > 1 and words[-1].lower() in TIME_UNITS:
        words.pop()

    if not words:
        return None

    # Frasa kepanjangan dipangkas ke kepala + dua pewatas; frasa aspek yang baik
    # pendek, dan span panjang membuat batas BIO jadi kabur.
    if len(words) > MAX_TERM_WORDS:
        words = words[:3]

    term = ' '.join(words)
    return term if len(term) >= MIN_TERM_CHARS else None


# ── Uji kelayakan sebagai aspek ─────────────────────────────────────────────

def is_verbal(word: str) -> bool:
    """True bila kata tampak sebagai verba berimbuhan."""
    w = word.lower()
    if w in NOUN_EXCEPTIONS or w in OPINION_WORDS:
        return False
    if not any(w.startswith(p) for p in VERBAL_PREFIXES):
        return False
    # Awalan hanya dianggap verbal bila pengupasannya menghasilkan akar berbeda,
    # artinya Sastrawi memang mengenalinya sebagai bentuk berimbuhan.
    stem = STEMMER.stem(w)
    return stem != w and len(stem) >= 3


def reject_reason(term: str, lowercase_seen: Set[str]) -> Optional[str]:
    """
    Kembalikan alasan penolakan, atau None bila istilah layak jadi aspek.

    Args:
        term: Istilah yang sudah dinormalisasi.
        lowercase_seen: Himpunan kata yang pernah muncul huruf kecil di korpus;
            dipakai untuk membedakan nomina umum dari nama diri.
    """
    words = term.split()
    low = [w.lower() for w in words]
    head = low[0]

    if len(term) < MIN_TERM_CHARS:
        return 'terlalu pendek'
    if any(c.isdigit() for c in term):
        return 'mengandung angka'
    if head in FUNCTION_WORDS:
        return 'berkepala kata fungsi'
    if all(w in FUNCTION_WORDS for w in low):
        return 'seluruhnya kata fungsi'
    if head in OPINION_WORDS:
        return 'berkepala kata opini'
    if head in TIME_UNITS:
        return 'berkepala satuan waktu'
    if is_verbal(head):
        return 'berkepala verba'
    if head.endswith('kan') and is_verbal(head):
        return 'berkepala verba'

    # Akronim konsep (RUPS, BUMN, KAP, PPN) adalah nomina umum di domainnya dan
    # tetap diterima. Yang dibuang hanya nama diri berkapital seperti
    # 'Madrid' atau 'Mentari' -- itu ranah NER, dan melatihnya membuat model
    # belajar menandai kapitalisasi alih-alih pola frasa nomina.
    alpha = [w for w in words if w[:1].isalpha()]
    if alpha and all(w.isupper() and len(w) <= 6 for w in alpha):
        return None

    if alpha and all(w[0].isupper() for w in alpha):
        if not any(w.lower() in lowercase_seen for w in alpha):
            return 'nama diri'

    return None


# ── Muat & panen ────────────────────────────────────────────────────────────

def load_items(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def harvest_terms(items: List[Dict[str, Any]]) -> Counter:
    """Kumpulkan istilah kandidat dari span anotasi sumber."""
    terms: Counter = Counter()
    for it in items:
        text = it['data']['text']
        for pred in it.get('predictions') or []:
            for res in pred.get('result') or []:
                value = res.get('value', {})
                if 'start' not in value:
                    continue
                # Ambil dari offset, bukan field 'text': pada berkas preannotated
                # 36,7% field 'text' tidak sinkron dengan offsetnya.
                surface = text[value['start']:value['end']]
                normalized = normalize_term(surface)
                if normalized:
                    terms[normalized] += 1
    return terms


def build_lowercase_vocab(texts: List[str]) -> Set[str]:
    """Kata-kata yang pernah muncul dalam huruf kecil di korpus."""
    seen: Set[str] = set()
    for t in texts:
        for m in re.finditer(r'\b[a-z][a-z\-]+\b', t):
            seen.add(m.group())
    return seen


# ── Proyeksi konsisten ──────────────────────────────────────────────────────

def project(text: str, patterns: List[Tuple[str, Any]]) -> List[Tuple[int, int, str]]:
    """
    Tandai SEMUA kemunculan setiap istilah pada satu teks.

    Inilah yang memperbaiki inkonsistensi 56%: anotasi lama hanya menandai
    kemunculan pertama karena memakai str.find().
    """
    found: List[Tuple[int, int, str]] = []
    for term, rx in patterns:
        for m in rx.finditer(text):
            found.append((m.start(), m.end(), term))
    return resolve_overlaps(found)


def resolve_overlaps(spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """Sisakan span terpanjang bila bertumpang tindih ('saham publik' > 'saham')."""
    kept: List[Tuple[int, int, str]] = []
    for start, end, term in sorted(spans, key=lambda s: (-(s[1] - s[0]), s[0])):
        if any(not (end <= ks or start >= ke) for ks, ke, _ in kept):
            continue
        kept.append((start, end, term))
    return sorted(kept, key=lambda s: s[0])


def compile_patterns(terms: List[str]) -> List[Tuple[str, Any]]:
    """Pola batas-kata, tidak peka huruf besar, terpanjang lebih dulu."""
    out = []
    for term in sorted(terms, key=len, reverse=True):
        escaped = r'\s+'.join(re.escape(w) for w in term.split())
        out.append((term, re.compile(rf'(?<!\w){escaped}(?!\w)', re.IGNORECASE)))
    return out


# ── Keluaran ────────────────────────────────────────────────────────────────

def to_label_studio(records, model_version: str):
    return [{
        'data': {'text': rec['text']},
        'predictions': [{
            'model_version': model_version,
            'result': [{
                'from_name': 'label', 'to_name': 'text', 'type': 'labels',
                'value': {'start': s, 'end': e, 'text': rec['text'][s:e],
                          'labels': ['ASPECT']},
            } for s, e, _ in rec['spans']],
        }],
    } for rec in records]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', required=True,
                    help='berkas beranotasi sebagai sumber kandidat istilah')
    ap.add_argument('--target', required=True,
                    help='berkas yang akan dianotasi (teksnya)')
    ap.add_argument('--out', default='data/aspect_news')
    ap.add_argument('--min-freq', type=int, default=1,
                    help='frekuensi minimal istilah agar dipakai')
    ap.add_argument('--include-empty', action='store_true',
                    help='sertakan kalimat tanpa aspek sebagai contoh negatif')
    args = ap.parse_args()

    source_items = load_items(args.source)
    target_items = load_items(args.target)
    texts = [it['data']['text'] for it in target_items]

    raw_terms = harvest_terms(source_items)
    lowercase_vocab = build_lowercase_vocab(texts)

    accepted: List[str] = []
    rejected: Counter = Counter()
    rejected_examples: Dict[str, List[str]] = {}

    for term, freq in raw_terms.items():
        if freq < args.min_freq:
            continue
        reason = reject_reason(term, lowercase_vocab)
        if reason:
            rejected[reason] += 1
            rejected_examples.setdefault(reason, [])
            if len(rejected_examples[reason]) < 5:
                rejected_examples[reason].append(term)
        else:
            accepted.append(term)

    patterns = compile_patterns(accepted)

    records = []
    surface_freq: Counter = Counter()
    for text in texts:
        spans = project(text, patterns)
        for s, e, _ in spans:
            surface_freq[text[s:e].lower()] += 1
        records.append({'text': text, 'spans': spans})

    covered = sum(1 for r in records if r['spans'])
    n_spans = sum(len(r['spans']) for r in records)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    ls_path = f'{args.out}_label_studio.json'
    with open(ls_path, 'w', encoding='utf-8') as f:
        json.dump(to_label_studio(records, 'rule_filtered_consistent_v1'),
                  f, ensure_ascii=False, indent=1)

    payload = [{'text': r['text'], 'aspects': [r['text'][s:e] for s, e, _ in r['spans']]}
               for r in records if r['spans'] or args.include_empty]
    rt_path = f'{args.out}_retrain.json'
    with open(rt_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    lex_path = f'{args.out}_lexicon.json'
    with open(lex_path, 'w', encoding='utf-8') as f:
        json.dump({'accepted': sorted(accepted),
                   'rejected_counts': dict(rejected),
                   'rejected_examples': rejected_examples},
                  f, ensure_ascii=False, indent=1)

    print('=' * 66)
    print('ANOTASI ASPEK OTOMATIS')
    print('=' * 66)
    print(f'  kandidat mentah        : {len(raw_terms)}')
    print(f'  diterima               : {len(accepted)}')
    print(f'  ditolak                : {sum(rejected.values())}')
    for reason, count in rejected.most_common():
        print(f'      {reason:26} {count:5}  {rejected_examples[reason][:3]}')
    print()
    print(f'  teks                   : {len(texts)}')
    print(f'  teks dengan aspek      : {covered} ({covered / len(texts) * 100:.1f}%)')
    print(f'  total span             : {n_spans}')
    print(f'  bentuk permukaan unik  : {len(surface_freq)}')
    print(f'  sampel untuk retrain   : {len(payload)}')
    print()
    print(f'  -> {ls_path}')
    print(f'  -> {rt_path}')
    print(f'  -> {lex_path}')


if __name__ == '__main__':
    main()
