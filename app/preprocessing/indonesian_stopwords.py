from pathlib import Path
from typing import Set

class IndonesianStopwords:
    """Indonesian stopwords loader"""
    
    def __init__(self, custom_stopwords: list = None):
        self.stopwords = self._load_default_stopwords()
        
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
    
    def _load_default_stopwords(self) -> Set[str]:
        """Load default Indonesian stopwords from file"""
        stopwords_file = Path(__file__).parent.parent.parent / "data" / "stopwords_id.txt"
        
        if not stopwords_file.exists():
            # Fallback to basic stopwords
            return self._get_basic_stopwords()
        
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    
    def _get_basic_stopwords(self) -> Set[str]:
        """Basic Indonesian stopwords as fallback"""
        return {
            'ada', 'adalah', 'adanya', 'adapun', 'agak', 'agaknya', 'agar',
            'akan', 'akankah', 'akhir', 'akhiri', 'akhirnya', 'aku', 'akulah',
            'amat', 'amatlah', 'anda', 'andalah', 'antar', 'antara', 'antaranya',
            'apa', 'apaan', 'apabila', 'apakah', 'apalagi', 'apatah', 'artinya',
            'asal', 'asalkan', 'atas', 'atau', 'ataukah', 'ataupun', 'awal',
            'awalnya', 'bagai', 'bagaikan', 'bagaimana', 'bagaimanapun',
            'bagaimanakah', 'bagi', 'bagian', 'bahkan', 'bahwa', 'bahwasanya',
            'baik', 'bakal', 'bakalan', 'balik', 'banyak', 'banyaknya', 'bapak',
            'baru', 'bawah', 'beberapa', 'begini', 'beginian', 'beginikah',
            'beginilah', 'begitu', 'begitukah', 'begitulah', 'begitupun',
            'bekerja', 'belakang', 'belakangan', 'belum', 'belumlah', 'benar',
            'benarkah', 'benarlah', 'berada', 'berakhir', 'berakhirlah',
            'berakhirnya', 'berapa', 'berapakah', 'berapalah', 'berapapun',
            'berarti', 'berawal', 'berbagai', 'berdatangan', 'beri', 'berikan',
            'berikut', 'berikutnya', 'berjumlah', 'berkali', 'berkata',
            'berkehendak', 'berkeinginan', 'berkenaan', 'berlainan', 'berlalu',
            'berlangsung', 'berlebihan', 'bermacam', 'bermaksud', 'bermula',
            'bersama', 'bersiap', 'bertanya', 'berturut', 'bertutur', 'berujar',
            'berupa', 'besar', 'betul', 'betulkah', 'biasa', 'biasanya', 'bila',
            'bilakah', 'bisa', 'bisakah', 'boleh', 'bolehkah', 'bolehlah',
            'buat', 'bukan', 'bukankah', 'bukanlah', 'bukannya', 'bulan', 'bung',
            'cara', 'caranya', 'cukup', 'cukupkah', 'cukuplah', 'cuma', 'dahulu',
            'dalam', 'dan', 'dapat', 'dari', 'daripada', 'datang', 'dekat',
            'demi', 'demikian', 'demikianlah', 'dengan', 'depan', 'di', 'dia',
            'diantara', 'diberi', 'diberikan', 'dibuat', 'didapat', 'digunakan',
            'dikatakan', 'dikerjakan', 'diketahui', 'dikira', 'dilakukan',
            'dimaksud', 'dimaksudkan', 'diminta', 'dimulai', 'dini', 'dipastikan',
            'diperbuat', 'dipergunakan', 'diperkirakan', 'diperlihatkan',
            'diperlukan', 'dipersoalkan', 'dipertanyakan', 'dipunyai', 'diri',
            'dirinya', 'disampaikan', 'disebut', 'disebutkan', 'disini',
            'ditambahkan', 'ditanya', 'ditanyakan', 'ditegaskan', 'ditujukan',
            'ditunjuk', 'ditunjukkan', 'diucapkan', 'diungkapkan', 'dong', 'dua',
            'dulu', 'empat', 'enggak', 'enggaknya', 'entah', 'entahlah'
        }
    
    def get_stopwords(self) -> Set[str]:
        """Get all stopwords"""
        return self.stopwords
    
    def is_stopword(self, word: str) -> bool:
        """Check if word is stopword"""
        return word.lower() in self.stopwords
    
    def add_stopwords(self, words: list):
        """Add custom stopwords"""
        self.stopwords.update(words)