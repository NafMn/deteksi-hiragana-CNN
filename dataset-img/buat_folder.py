import os
folder_tujuan = r"D:\open cv\deteksi-hiragana CNN\dataset-img"
# Data huruf Hiragana
hiragana_data = [
    # 'あ', 'い', 'う', 'え', 'お',
    # 'か', 'き', 'く', 'け', 'こ',
    # 'さ', 'し', 'す', 'せ', 'そ',
    # 'た', 'ち', 'つ', 'て', 'と',
    # 'な', 'に', 'ぬ', 'ね', 'の',
    # 'は', 'ひ', 'ふ', 'へ', 'ほ',
    # 'ま', 'み', 'む', 'め', 'も',
    # 'や', 'ゆ', 'よ',
    # 'ら', 'り', 'る', 'れ', 'ろ',
    # 'わ', 'を', 'ん'
    'ば', 'び', 'ぶ', 'べ', 'ぼ',
    'ぱ', 'ぴ', 'ぷ', 'ぺ', 'ぽ'
]

# Membuat folder berdasarkan huruf Hiragana
for hiragana in hiragana_data:
    folder_path = os.path.join(folder_tujuan, hiragana)
    os.makedirs(folder_path, exist_ok=True)

print("okee,.. dah jadi foldernya")

