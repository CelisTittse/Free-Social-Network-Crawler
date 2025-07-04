# Free-Social-Network-Crawler

**The Verification of Real Connections from Webpages**

## Project Overview

The core idea behind this tool is straightforward: starting from a small set of known individuals, can publicly accessible websites be used to detect potential social links between them and others? By identifying and comparing personal names mentioned on the same webpages, the system attempts to infer possible social proximity between people—such as friends, colleagues, or collaborators—without requiring structured datasets or access to social media.

To achieve this, the pipeline processes a combination of raw web text, individual metadata, and outputs from Named Entity Recognition (NER) to detect co-mentions of names. It then filters, ranks, and analyses these mentions, measuring proximity based on both textual distance and frequency of appearance. The approach also includes automated language detection to support multilingual data and removes social media and commercial content.

## Key Features

* Recognition of personal names using a multilingual transformer-based NER model
* Filtering of known names and removal of social media or platform-related pages
* Sentence-level distance calculation between known and co-mentioned names
* Language classification to support cross-lingual data handling
* Export of results into a structured format for further analysis

## Limitations and Ethical Considerations

While the pipeline shows promise, particularly within digital humanities or historical research contexts, it is subject to several important limitations:

* **Data Quality:** The analysis relies entirely on unstructured, open web data, which is frequently incomplete, inconsistent, or unreliable. Determining whether co-mentioned individuals are actually connected in real life is inherently uncertain.

* **Ambiguity in Names:** It is often unclear whether repeated mentions of a name refer to the same person. Although the system applies basic filtering, distinguishing between individuals with common names, pseudonyms, or titles remains a challenge.

* **Population Bias:** The model was evaluated using a small and demographically narrow group—primarily affluent, white students and early-career professionals. The findings therefore cannot be generalised to more diverse or representative populations.

* **Model Performance:** The Wikineural Multilingual NER model, in combination with word- and character-level filtering, performed moderately well in extracting personal names. However, the results remain cautious and partial. The system frequently fails to identify all relevant names present on a webpage, often capturing only fragments of the underlying network.

* **Privacy Risks:** Although the tool is relatively modest in scope and unlikely to attract commercial misuse, it does raise ethical questions regarding the inference of personal relationships from publicly available data. The likelihood of abuse is considered low—particularly when compared with existing commercial tools—but responsible usage remains essential.

## Conclusions

This study developed a lightweight, semi-automated method for identifying potential social connections between individuals using only public web data and minimal prior knowledge. The algorithm produced conservative but reasonably dependable results in controlled settings and was able to reveal possible social ties—particularly when names of public figures or commonly occurring names were filtered out.

Nonetheless, significant limitations in both the data and evaluation methods mean that the outcomes should be regarded as exploratory rather than definitive. The approach serves as a proof of concept for detecting weak signals of social proximity in public text sources. However, attempting to reconstruct an individual’s social network using this method is rather like trying to solve a puzzle with most of the pieces missing. While it may provide useful leads for further qualitative research, it cannot substitute for verified relational data.
