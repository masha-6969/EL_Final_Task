import spacy
import os

# spaCyモデルのロード (初回のみダウンロードが必要)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Please run:")
    print("python -m spacy download en_core_web_sm")
    exit()

def read_text_file(filepath):
    """指定されたファイルパスからテキストを読み込む"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None

def normalize_and_pos_tag(text):
    """テキストを正規化し、品詞タグを付与してトークンのリストを返す"""
    doc = nlp(text.lower()) # 小文字化してspaCyで処理
    tokens_with_pos = []
    for token in doc:
        # 句読点と空白以外のトークンを対象とする
        if token.is_alpha or token.is_digit: # 英字か数字のみを保持
            tokens_with_pos.append({
                'text': token.text,
                'pos': token.pos_ # 品詞タグ
            })
    return tokens_with_pos

def find_longest_common_patterns(tokens1, tokens2, min_length=1):
    """
    2つのトークンリストから最長共通パターンを見つける (LCSベース)。
    品詞情報も保持する。
    """
    len1 = len(tokens1)
    len2 = len(tokens2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    patterns = set() # テキストパターン (重複排除用)
    results = [] # 最終結果

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if tokens1[i-1]['text'] == tokens2[j-1]['text']:
                dp[i][j] = dp[i-1][j-1] + 1
                
                if dp[i][j] >= min_length:
                    start_index_1 = i - dp[i][j]
                    
                    current_pattern_tokens_text = [t['text'] for t in tokens1[start_index_1:i]]
                    pattern_text = " ".join(current_pattern_tokens_text)

                    current_pattern_tokens_pos = [t['pos'] for t in tokens1[start_index_1:i]]
                    pos_pattern_text = "-".join(current_pattern_tokens_pos)

                    if pattern_text not in patterns:
                        patterns.add(pattern_text)
                        
                        results.append({
                            'pattern': pattern_text,
                            'pos_pattern': pos_pattern_text,
                            'length': len(current_pattern_tokens_text),
                            'count': 1 # このカウントは現在プレースホルダーです。
                        })
            else:
                dp[i][j] = 0 # 不一致の場合はリセット
    
    filtered_results = []
    results.sort(key=lambda x: x['length'], reverse=True) 

    for r in results:
        is_sub_pattern = False
        for existing_r in filtered_results:
            if existing_r['pattern'] != r['pattern'] and r['pattern'] in existing_r['pattern'] and existing_r['length'] > r['length']:
                is_sub_pattern = True
                break
        if not is_sub_pattern:
            filtered_results.append(r)

    filtered_results.sort(key=lambda x: (x['length'], x['count']), reverse=True)

    return filtered_results

def find_pos_discrepancies(tokens1, tokens2):
    """
    両方のトークンリストに存在するが、異なる品詞タグを持つ単語を見つける。
    各不一致のインスタンスを報告する。
    """
    discrepancies = set() # (word, pos_text1, pos_text2) のタプルを格納し、重複を排除

    # 各テキストの単語と品詞のペアのセットを作成
    # これにより、それぞれのテキストに存在する (単語, 品詞) のユニークな組み合わせが得られる
    # 例: ("run", "VERB"), ("run", "NOUN")
    text1_word_pos_pairs = {(token['text'], token['pos']) for token in tokens1}
    text2_word_pos_pairs = {(token['text'], token['pos']) for token in tokens2}

    # テキスト1に存在する各単語の品詞情報を走査
    for word1_text, word1_pos in text1_word_pos_pairs:
        # この単語がテキスト2にも存在し、かつ品詞が異なる場合を探す
        for word2_text, word2_pos in text2_word_pos_pairs:
            if word1_text == word2_text and word1_pos != word2_pos:
                # 不一致をタプルとしてセットに追加 (重複排除のため)
                # 報告順序を考慮し、常に (word, pos_from_text1, pos_from_text2) の形式にする
                discrepancies.add((word1_text, word1_pos, word2_pos))
    
    # セットをリストに変換し、ソートして一貫性のある出力にする
    sorted_discrepancies = sorted(list(discrepancies), key=lambda x: (x[0], x[1], x[2]))

    # 出力形式に合わせて辞書のリストに変換
    formatted_discrepancies = [
        {'word': d[0], 'pos_text1': d[1], 'pos_text2': d[2]}
        for d in sorted_discrepancies
    ]

    return formatted_discrepancies


def write_results_to_file(filepath, common_patterns, pos_discrepancies):
    """解析結果をファイルに書き込む"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("---\n")
            f.write("## Shared Token Patterns (Text and POS Match)\n") # 英語のセクションヘッダー
            if not common_patterns:
                f.write("No shared token patterns found.\n") # 英語メッセージ
            else:
                for i, r in enumerate(common_patterns):
                    f.write(f"{i + 1}. \"{r['pattern']}\" (Length: {r['length']} tokens, POS: {r['pos_pattern']})\n") # 英語メッセージ

            f.write("\n---\n")
            f.write("## POS Discrepancies (Same Word, Different POS)\n") # 英語のセクションヘッダー
            if not pos_discrepancies:
                f.write("No POS discrepancies found.\n") # 英語メッセージ
            else:
                f.write("Words with POS Discrepancies:\n") # 英語メッセージ
                for d in pos_discrepancies:
                    f.write(f"  Text1: \"{d['word']}\" POS: {d['pos_text1']}\n") # 英語メッセージ
                    f.write(f"  Text2: \"{d['word']}\" POS: {d['pos_text2']}\n") # 英語メッセージ
                    f.write("\n") # 各エントリ間に空行を追加して読みやすくする

    except Exception as e:
        print(f"Error writing to file {filepath}: {e}")

# メイン処理
if __name__ == "__main__":
    text1_path = 'text1.txt'
    text2_path = 'text2.txt'
    output_path = 'out.txt'

    # テスト用のダミーテキストファイルを生成
    with open(text1_path, 'w', encoding='utf-8') as f:
        f.write("The quick brown fox jumps over the lazy dog. He runs quickly. My project is a success. I can run fast. I like to run. This is a big run.")
    with open(text2_path, 'w', encoding='utf-8') as f:
        f.write("A brown fox quickly jumps over the lazy canine. He runs. Her project was a success. He takes a morning run. It is a very good run.")

    print(f"Reading {text1_path}...") # 英語メッセージ
    text1_content = read_text_file(text1_path)
    print(f"Reading {text2_path}...") # 英語メッセージ
    text2_content = read_text_file(text2_path)

    if text1_content is None or text2_content is None:
        print("Exiting due to file read errors.") # 英語メッセージ
    else:
        print("Normalizing and POS tagging text1...") # 英語メッセージ
        tokens1 = normalize_and_pos_tag(text1_content)
        print("Normalizing and POS tagging text2...") # 英語メッセージ
        tokens2 = normalize_and_pos_tag(text2_content)

        print("Finding common patterns...") # 英語メッセージ
        common_patterns = find_longest_common_patterns(tokens1, tokens2)

        print("Finding POS discrepancies...") # 英語メッセージ
        pos_discrepancies = find_pos_discrepancies(tokens1, tokens2)

        print(f"Writing results to {output_path}...") # 英語メッセージ
        write_results_to_file(output_path, common_patterns, pos_discrepancies)
        print("Done.") # 英語メッセージ