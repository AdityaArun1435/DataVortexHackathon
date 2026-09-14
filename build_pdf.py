from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

styles = getSampleStyleSheet()
title_style = ParagraphStyle('TitleCustom', parent=styles['Title'], fontSize=20, spaceAfter=6)
h2 = ParagraphStyle('H2', parent=styles['Heading2'], spaceBefore=16, spaceAfter=8)
body = ParagraphStyle('BodyCustom', parent=styles['Normal'], fontSize=10.5, leading=15)

doc = SimpleDocTemplate(
    "eda_report.pdf", pagesize=letter,
    leftMargin=0.8*inch, rightMargin=0.8*inch,
    topMargin=0.7*inch, bottomMargin=0.7*inch
)

story = []

story.append(Paragraph("Data Vortex — Round 1 Phase 1", title_style))
story.append(Paragraph("Exploratory Data Analysis Report", styles['Heading3']))
story.append(Spacer(1, 12))

story.append(Paragraph("1. Dataset Overview", h2))
story.append(Paragraph(
    "The recovered dataset consisted of 12,360 raw post records and 1,500 user records. "
    "After removing 360 exact duplicate rows, the cleaned posts dataset contains 12,000 rows. "
    "The users table required no corrections: no nulls, no duplicate user IDs, and full "
    "referential integrity with the posts table (every user_id in posts exists in users).",
    body
))

story.append(Paragraph("2. Cleaning Summary", h2))
table_data = [
    ["Issue", "Column", "Count", "Fix Applied"],
    ["Exact duplicate rows", "post_id", "360", "Dropped, kept first occurrence"],
    ["Mixed timestamp formats\n(epoch / DD-MM-YYYY / ISO8601)", "timestamp", "12,360", "Parsed to single datetime dtype"],
    ["Missing platform", "platform", "1,784", "Filled as \"Unknown\" (no basis to infer)"],
    ["Literal \"NULL\" strings", "text_content", "23", "Converted to true NaN"],
    ["HTML-escaped text", "text_content", "328", "Unescaped (&amp; \u2192 &, etc.)"],
    ["Genuinely missing text", "text_content", "1,711", "Left as NaN, not fabricated"],
    ["Negative likes (impossible)", "likes", "509", "Converted to NaN before imputation"],
    ["Missing/negative likes", "likes", "2,323", "Imputed with platform-median, flagged in likes_imputed"],
]
t = Table(table_data, colWidths=[1.6*inch, 0.9*inch, 0.7*inch, 2.4*inch])
t.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c3e50')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('FONTSIZE', (0,0), (-1,-1), 8.5),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')]),
    ('TOPPADDING', (0,0), (-1,-1), 5),
    ('BOTTOMPADDING', (0,0), (-1,-1), 5),
]))
story.append(t)

story.append(Paragraph("3. Visual Summary", h2))
story.append(Image("eda_charts.png", width=6.4*inch, height=6.4*inch*(1050/1520)))
story.append(PageBreak())

story.append(Paragraph("4. Key Findings", h2))
findings = [
    ("Platform distribution", "Roughly even across the 5 real platforms (1,989\u20132,074 posts each), "
     "with 1,784 posts of unknown platform. No single platform dominates the corruption, "
     "suggesting it was applied close to uniformly at random rather than targeting one platform."),
    ("Posting volume over time", "Stable at roughly 975\u20131,040 posts/month from May 2024 to April 2025, "
     "with no visible growth, decay, or seasonal spike."),
    ("Engagement correlation", "Likes, shares, comments, and follower_count show near-zero correlation "
     "with each other (|r| \u2264 0.02). This indicates the engagement numbers in this dataset are synthetic "
     "or randomly generated rather than reflecting real social dynamics \u2014 a more popular account "
     "(higher followers) does not get more likes here. This is an important caveat for Phase 2: "
     "a SQL correlation analysis on this data correctly returning near-zero correlation is the right "
     "answer, not a bug in the query."),
    ("User base diversity", "Spans 10 languages fairly evenly (1,119\u20131,369 users each) and is "
     "geographically diverse, with no single location dominating."),
]
for heading, text in findings:
    story.append(Paragraph(f"<b>{heading}.</b> {text}", body))
    story.append(Spacer(1, 8))

story.append(Paragraph("5. Files", h2))
files = [
    "cleaned_posts.csv / cleaned_users.csv \u2014 cleaned output tables",
    "cleaned_dataset_merged.csv \u2014 single-file join of posts and users on user_id, for submission",
    "clean_pipeline.py \u2014 full reproducible cleaning script",
    "cleaning_log.txt \u2014 plain-text log of every transformation applied",
    "eda_charts.png \u2014 platform volume, posting trend, engagement, and language distribution charts",
]
for f in files:
    story.append(Paragraph(f"\u2022 {f}", body))

doc.build(story)
print("PDF built")
