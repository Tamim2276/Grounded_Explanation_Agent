# The fake two-page paper the stub backend searches, and the questions asked of it.
#
# Boxes are (x_min, y_min, x_max, y_max) in PDF points on an A4 page, measured
# with PyMuPDF; notebook Section 13 renders a real PDF from exactly these records,
# so every box the graph returns can be checked against the page. The numbers are
# consistent, not typed to match: the All row of Table 3 is the 80/20 weighted
# average of the two subsets, and the Results paragraph quotes its +3.9.

FAKE_PAPER = {
    "paper_id": "demo-2026-001",
    "pages": 2,
    "front": [
        {"page": 1, "size": 16, "font": "hebo", "bbox": (60.0, 56.0, 535.0, 84.9),
         "text": "Cross-Lingual Transfer for Multilingual Sentiment Analysis"},
        {"page": 1, "size": 9, "font": "heit", "bbox": (60.0, 88.9, 535.0, 105.6),
         "text": "Synthetic two-page excerpt generated for this demo. Not a real publication."},
    ],
    "headings": [
        {"page": 1, "section": "1", "text": "1   Introduction",   "bbox": (60.0, 119.6, 535.0, 141.8)},
        {"page": 1, "section": "2", "text": "2   Method",         "bbox": (60.0, 188.3, 535.0, 210.5)},
        {"page": 1, "section": "3", "text": "3   Training Setup", "bbox": (60.0, 257.0, 535.0, 279.2)},
        {"page": 2, "section": "4", "text": "4   Results",        "bbox": (60.0, 56.0, 535.0, 78.2)},
        {"page": 2, "section": "5", "text": "5   Conclusion",     "bbox": (60.0, 254.3, 535.0, 276.5)},
    ],
    "chunks": [
        {"id": "c1", "page": 1, "section": "1", "bbox": (60.0, 143.8, 535.0, 176.3),
         "text": "Sentiment classifiers work well for languages with large labelled datasets, "
                 "but many languages have little or no labelled data. We study how far "
                 "cross-lingual transfer from English can close this gap."},
        {"id": "c2", "page": 1, "section": "2", "bbox": (60.0, 212.5, 535.0, 245.0),
         "text": "We fine-tune a multilingual encoder on English product reviews and apply it "
                 "to twelve other languages without any target-language labels."},
        {"id": "c3", "page": 1, "section": "3", "bbox": (60.0, 281.2, 535.0, 313.7),
         "text": "We train for 20 epochs with early stopping on the validation set. "
                 "Figure 2 shows the validation loss after each epoch."},
        {"id": "c4", "page": 2, "section": "4", "bbox": (60.0, 80.2, 535.0, 126.4),
         "text": "Table 3 compares our model with the baseline on each test subset. Averaged "
                 "over all test data, our model achieves an accuracy gain of 3.9 points over "
                 "the baseline. Most of this gain comes from high-resource languages, where "
                 "training data is plentiful."},
        {"id": "c5", "page": 2, "section": "5", "bbox": (60.0, 278.5, 535.0, 311.0),
         "text": "Cross-lingual transfer improves accuracy overall, but the benefit is uneven "
                 "across languages. Narrowing the remaining gap for languages with little "
                 "training data is left to future work."},
    ],
    "tables": [
        {"id": "t3", "label": "Table 3", "page": 2, "bbox": (122.5, 134.4, 472.5, 214.4),
         "col_widths": (120.0, 80.0, 80.0, 70.0), "row_h": 20.0,
         "caption": "Table 3: Accuracy (%) of the baseline and our model on each test subset.",
         "caption_bbox": (60.0, 222.4, 535.0, 240.3),
         "header": ["Subset", "Baseline", "Ours", "Gain"],
         "rows": [["High-resource", "88.1", "92.5", "+4.4"],
                  ["Low-resource",  "71.3", "73.0", "+1.7"],
                  ["All",           "84.7", "88.6", "+3.9"]]},
    ],
    "figures": [
        {"id": "f2", "label": "Figure 2", "page": 1, "bbox": (150.0, 321.7, 445.0, 521.7),
         "caption": "Figure 2: Validation loss per training epoch. The loss drops quickly "
                    "and plateaus after epoch 10.",
         "caption_bbox": (60.0, 527.7, 535.0, 545.6),
         "epochs": list(range(1, 21))},
    ],
    # share of test examples per subset -- what makes the All row an average
    "subset_weights": {"High-resource": 0.8, "Low-resource": 0.2},
}

PAPER_ID = FAKE_PAPER["paper_id"]

# The running example of the thesis: needs a text claim AND a table.
QUERY = ("Does the reported accuracy gain in Section 4 hold for the low-resource subset, "
         "and which table reports it?")
TEXT_QUERY   = "How many epochs is the model trained for?"
TABLE_QUERY  = "Which table reports accuracy for each test subset?"
FIGURE_QUERY = "How does the validation loss change after epoch 10 in Figure 2?"
UNANSWERABLE = "What accuracy does the model reach on the Swahili test set?"
THREE_PART_QUERY = ("Does the accuracy gain in Section 4 hold for the low-resource subset; "
                    "which table reports it; and how does the validation loss in Figure 2 "
                    "change after epoch 10?")
