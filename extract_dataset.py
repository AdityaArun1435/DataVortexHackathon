"""
Data Vortex Round 2 - dataset recovery

The provided PDF renders each column (text_id, post_text, sentiment_label,
topic_category) as a SEPARATE text object anchored at a fixed x-position on
each row. Because post_text is long and overflows rightward, it visually
overlaps the sentiment_label/topic_category columns, and any naive text
extractor (or copy-paste) that reads left-to-right by x-position interleaves
the characters into garbage. The underlying character stream is not actually
corrupted, the columns just need to be re-separated by their known fixed
x-anchors instead of by reading order.
"""

import pdfplumber
import pandas as pd

PDF_PATH = '/mnt/user-data/uploads/Labeled_Social_NLP_Training_Data.pdf'

# fixed column anchors, taken from the header row (page 1)
ANCHORS = [
    ('text_id', 52.65),
    ('post_text', 128.4),
    ('sentiment_label', 204.15),
    ('topic_category', 279.9),
]
TOL = 1.5

rows = []
with pdfplumber.open(PDF_PATH) as pdf:
    for page_num, page in enumerate(pdf.pages):
        chars = page.chars
        if not chars:
            continue
        lines = {}
        for c in chars:
            key = round(c['top'], 0)
            lines.setdefault(key, []).append(c)

        for top, line_chars in sorted(lines.items()):
            if page_num == 0 and top < 60:
                continue  # header row

            col_idx = 0
            col_text = {name: [] for name, _ in ANCHORS}
            last_x = None
            for c in line_chars:
                x0 = c['x0']
                if col_idx == 0 and x0 >= ANCHORS[1][1] - TOL:
                    col_idx = 1
                elif col_idx == 1 and last_x is not None and x0 < last_x - 3:
                    # post_text overflowed past its column and just ended;
                    # a backward jump means sentiment_label is starting
                    col_idx = 2
                elif col_idx == 2 and last_x is not None and x0 < last_x - 3:
                    # sentiment_label itself overflowed and just ended
                    col_idx = 3
                elif col_idx == 2 and x0 >= ANCHORS[3][1] - TOL:
                    # short sentiment_label (Positive/Negative/Neutral) ended
                    # without a backward jump, topic_category starts right at
                    # its own fixed anchor
                    col_idx = 3
                col_name = ANCHORS[col_idx][0]
                col_text[col_name].append(c['text'])
                last_x = x0

            row = {name: ''.join(col_text[name]).strip() for name, _ in ANCHORS}
            if row['text_id']:
                rows.append(row)

df = pd.DataFrame(rows)
print(df.shape)
print(df.head(15).to_string())
df.to_csv('recovered_dataset_raw.csv', index=False)
