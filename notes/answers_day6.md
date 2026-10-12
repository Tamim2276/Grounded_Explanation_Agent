# Day 6: real grounded answers (dev)

## 1803.03467v4#0: Which dataset has the most 4-hop triples?

**Gold:** Bing-News. (Table 1)

**Answer** (ok): The 'Bing-News' dataset has the most 4-hop triples with 6,322,548 triples.

- The 'Bing-News' dataset has the most 4-hop triples with 6,322,548 triples. [E1 cell # 4-hop triples / Bing-News = 6,322,548]
- The 'MovieLens-1M' dataset has 923,718 4-hop triples. [E1 cell # 4-hop triples / MovieLens-1M = 923,718]
- The 'Book-Crossing' dataset has 71,628 4-hop triples. [E1 cell # 4-hop triples / Book-Crossing = 71,628]

  - box p.6 Table 1 cell # 4-hop triples / Bing-News (510.2, 172.7, 555.6, 183.0)
  - box p.6 Table 1 cell # 4-hop triples / MovieLens-1M (380.4, 172.7, 445.6, 183.0)
  - box p.6 Table 1 cell # 4-hop triples / Book-Crossing (445.6, 172.7, 509.9, 183.0)

## 1803.03467v4#1: How does the number of common k-hop neighbors change as the hop distance increases for items with and without common raters?

**Gold:** The number of common k-hop neighbors generally decreases as the hop distance increases for both items with and without common raters. However, the number of common k-hop neighbors is consistently higher for items with common raters than for items without common raters. (Figure 4)

**Answer** (ok): The number of common k-hop neighbors increases as the hop distance increases for items with common raters in MovieLens-1M, Book-Crossing, and Bing-News datasets.

- The number of common k-hop neighbors increases as the hop distance increases for items with common raters in MovieLens-1M, Book-Crossing, and Bing-News datasets. [E1 cell (a) MovieLens-1M with common rater(s) / 1 = 4.3; E1 cell (a) MovieLens-1M with common rater(s) / 2 = 25.8; E1 cell (a) MovieLens-1M with common rater(s) / 3 = 42.6; E1 cell (a) MovieLens-1M with common rater(s) / 4 = 59.3]
- The number of common k-hop neighbors increases as the hop distance increases for items without common raters in MovieLens-1M, Book-Crossing, and Bing-News datasets. [E1 cell (a) MovieLens-1M without common rater / 1 = 1.1; E1 cell (a) MovieLens-1M without common rater / 2 = 9.3; E1 cell (a) MovieLens-1M without common rater / 3 = 27; E1 cell (a) MovieLens-1M without common rater / 4 = 45.9]
- The number of common k-hop neighbors increases as the hop distance increases for items with common raters in MovieLens-1M, Book-Crossing, and Bing-News datasets. [E1 cell (b) Book-Crossing with common rater(s) / 1 = 2.7; E1 cell (b) Book-Crossing with common rater(s) / 2 = 4.6; E1 cell (b) Book-Crossing with common rater(s) / 3 = 6.8; E1 cell (b) Book-Crossing with common rater(s) / 4 = 10.9]
- The number of common k-hop neighbors increases as the hop distance increases for items without common raters in MovieLens-1M, Book-Crossing, and Bing-News datasets. [E1 cell (b) Book-Crossing without common rater / 1 = 0.5; E1 cell (b) Book-Crossing without common rater / 2 = 1; E1 cell (b) Book-Crossing without common rater / 3 = 3.1; E1 cell (b) Book-Crossing without common rater / 4 = 6.3]

  - box p.7 Figure 4 bar (a) MovieLens-1M with common rater(s) / 1 (75.4, 140.1, 80.9, 144.0)
  - box p.7 Figure 4 bar (a) MovieLens-1M with common rater(s) / 2 (99.8, 119.7, 105.3, 144.0)
  - box p.7 Figure 4 bar (a) MovieLens-1M with common rater(s) / 3 (124.2, 103.7, 129.7, 144.0)
  - box p.7 Figure 4 bar (a) MovieLens-1M with common rater(s) / 4 (148.6, 87.9, 154.1, 144.0)
  - box p.7 Figure 4 bar (a) MovieLens-1M without common rater / 1 (82.3, 143.1, 87.9, 144.0)
  - box p.7 Figure 4 bar (a) MovieLens-1M without common rater / 2 (106.7, 135.3, 112.3, 144.0)
  - box p.7 Figure 4 bar (a) MovieLens-1M without common rater / 3 (131.1, 118.5, 136.7, 144.0)
  - box p.7 Figure 4 bar (a) MovieLens-1M without common rater / 4 (155.5, 100.6, 161.1, 144.0)
  - box p.7 Figure 4 bar (b) Book-Crossing with common rater(s) / 1 (198.5, 134.0, 204.2, 144.2)
  - box p.7 Figure 4 bar (b) Book-Crossing with common rater(s) / 2 (223.3, 126.9, 229.0, 144.2)
  - box p.7 Figure 4 bar (b) Book-Crossing with common rater(s) / 3 (248.1, 118.7, 253.8, 144.2)
  - box p.7 Figure 4 bar (b) Book-Crossing with common rater(s) / 4 (272.9, 103.0, 278.6, 144.2)
  - box p.7 Figure 4 bar (b) Book-Crossing without common rater / 1 (205.6, 142.5, 211.3, 144.2)
  - box p.7 Figure 4 bar (b) Book-Crossing without common rater / 2 (230.4, 140.5, 236.1, 144.2)
  - box p.7 Figure 4 bar (b) Book-Crossing without common rater / 3 (255.2, 132.5, 260.9, 144.2)
  - box p.7 Figure 4 bar (b) Book-Crossing without common rater / 4 (280.0, 120.5, 285.6, 144.2)

## 1803.03467v4#2: Which model performs the best in terms of AUC on the MovieLens-1M dataset?

**Gold:** RippleNet* (Table 3)

**Answer** (unsupported_claims_dropped): Every claim failed verification; nothing can be stated.

- ~~The highest AUC value for the MovieLens-1M dataset is 0.920, achieved with a ripple set size of 32.~~ [E1 cell MovieLens-1M / 32 = 0.920]
- ~~The highest AUC value for the MovieLens-1M dataset is 0.920, achieved with a ripple set size of 32.~~ [E1 cell MovieLens-1M / 32 = 0.920]
- ~~The highest AUC value for the MovieLens-1M dataset is 0.920, achieved with a ripple set size of 32.~~ [E1 cell MovieLens-1M / 32 = 0.920]
- ~~The AUC values for the MovieLens-1M dataset are 0.903, 0.908, 0.911, 0.918, 0.920, and 0.919 for ripple set sizes of 2, 4, 8, 16, 32, and 64, respectively.~~ [E1 cell MovieLens-1M / 2 = 0.903; E1 cell MovieLens-1M / 4 = 0.908; E1 cell MovieLens-1M / 8 = 0.911; E1 cell MovieLens-1M / 16 = 0.918]


## 1803.03467v4#3: What is the role of the ripple sets in the RippleNet framework?

**Gold:** The ripple sets are used to propagate a user's preferences from his or her click history to his or her relevant entities. (Figure 2)

**Answer** (unsupported_claims_dropped): The ripple sets in RippleNet are used to interact with the item embedding iteratively for obtaining the responses of user u with respect to item v.

- The ripple sets in RippleNet are used to interact with the item embedding iteratively for obtaining the responses of user u with respect to item v. [E1 quote "These ripple sets are used to interact with the item embedding (the yellow block) iterativ"]
- ~~The ripple sets in RippleNet are used to propagate user interests through the knowledge graph.~~ [E1 quote "RippleNet takes a user u and an item v as input, and outputs the predicted probability tha"]
- The ripple sets in RippleNet are used to interact with the item embedding iteratively for obtaining the responses of user u with respect to item v. [E1 quote "These ripple sets are used to interact with the item embedding (the yellow block) iterativ"]
- The size of the ripple sets in RippleNet can be varied to investigate the robustness of the framework. [E3 quote "We vary the size of a user’s ripple set in each hop to further investigate the robustness "]
- ~~The ripple sets in RippleNet are used to propagate user interests through the knowledge graph.~~ [E1 quote "RippleNet takes a user u and an item v as input, and outputs the predicted probability tha"]
- ~~The ripple sets in RippleNet are automatically discovered by the framework according to relevance probability.~~ [E2 quote "RippleNet automatically discovers the possible explanation paths according to relevance pr"]
- The ripple sets in RippleNet are used to interact with the item embedding iteratively for obtaining the responses of user u with respect to item v. [E1 quote "These ripple sets are used to interact with the item embedding (the yellow block) iterativ"]
- The size of the ripple sets in RippleNet can be varied to investigate the robustness of the framework. [E3 quote "We vary the size of a user’s ripple set in each hop to further investigate the robustness "]

  - box p.3 Section 3.1 sentence (53.5, 221.6, 295.0, 274.5)
  - box p.3 Section 3.1 sentence (53.5, 221.6, 295.0, 274.5)
  - box p.8 Section 4.5 sentence (318.0, 676.0, 559.6, 695.8)
  - box p.3 Section 3.1 sentence (53.5, 221.6, 295.0, 274.5)
  - box p.8 Section 4.5 sentence (318.0, 676.0, 559.6, 695.8)

## 1812.06589v2#0: How do the different methods compare in terms of their ability to generate realistic faces?

**Gold:** The baseline method generates faces that are blurry and unrealistic, while the other methods generate faces that are more realistic. (Figure 6)

**Answer** (ok): The figure shows examples of generated faces by the method compared to the ground truth and other methods.

- The figure shows examples of generated faces by the method compared to the ground truth and other methods. [E1 quote "Figure 5: Generation examples of our method comparing with Ground Truth (G.T.) (a), and Zh"]
- The figure shows examples of generated faces by the method compared to the ground truth and other methods. [E1 quote "Figure 5: Generation examples of our method comparing with Ground Truth (G.T.) (a), and Zh"]
- The zoomed view of Figure 5 highlights the differences in the details of the generated faces. [E2 observation "The zoomed view of Figure 5 shows more detail in the generated faces, indicating differenc"]

  - box p.5 Figure 5 caption (315.0, 243.5, 558.0, 262.4)
  - box p.5 Figure 5 caption (315.0, 243.5, 558.0, 272.4)
  - box p.5 Figure 5 (patch map) (331.5, 127.7, 553.9, 229.9)

## 1812.06589v2#1: What is the effect of adding DA to the baseline method?

**Gold:** Adding DA to the baseline method improves the PSNR and SSIM values, while slightly decreasing the LMD value. (Table 3)

**Answer** (unsupported_claims_dropped): DA plays an important role in talking face generation.

- DA plays an important role in talking face generation. [E3 quote "DA plays an important role in talking face generation, comparing Table 3 (b) to baseline."]
- The ablation study experiments show that DA contributes to the performance improvement. [E3 quote "DA plays an important role in talking face generation, comparing Table 3 (b) to baseline."]
- ~~The evidence does not directly state the effect of adding DA to the baseline method.~~ [E1 observation "The figure shows the pipeline of the proposed method, but does not specify the effect of D"]
- The ablation study experiments show that DA contributes to the performance improvement. [E3 quote "DA plays an important role in talking face generation, comparing Table 3 (b) to baseline."]
- ~~The evidence does not directly state the effect of adding DA to the baseline method.~~ [E3 quote "DA plays an important role in talking face generation, comparing Table 3 (b) to baseline."]
- ~~The effect of adding DA to the baseline method is significant, as it improves the results.~~ [E3 quote "DA plays an important role in talking face generation, comparing Table 3 (b) to baseline."]
- The ablation study experiments show that DA contributes to the performance improvement. [E3 quote "DA plays an important role in talking face generation, comparing Table 3 (b) to baseline."]
- ~~The evidence does not directly state the effect of adding DA to the baseline method.~~ [E3 quote "DA plays an important role in talking face generation, comparing Table 3 (b) to baseline."]

  - box p.6 Section 4.4 sentence (54.0, 584.9, 225.8, 594.8)
  - box p.6 Section 4.4 sentence (54.0, 584.9, 225.8, 594.8)
  - box p.6 Section 4.4 sentence (54.0, 584.9, 225.8, 594.8)
  - box p.6 Section 4.4 sentence (54.0, 584.9, 225.8, 594.8)

## 1812.06589v2#2: Which method performed the best on the GRID dataset?

**Gold:** AMIE (Ours) (Table 2)

**Answer** (unsupported_claims_dropped): Our method achieved the highest SSIM and the lowest LMD on the GRID dataset, demonstrating the effectiveness and robustness of our method.

- Our method achieved the highest SSIM and the lowest LMD on the GRID dataset, demonstrating the effectiveness and robustness of our method. [E1 quote "From Table 2, we observe that our model achieves the highest SSIM and the lowest LMD, demo"]
- ~~The quantitative results in Table 2 show that our method outperforms other methods on the GRID dataset.~~ [E1 quote "From Table 2, we observe that our model achieves the highest SSIM and the lowest LMD, demo"]
- Our method achieved the highest SSIM and the lowest LMD on the GRID dataset, demonstrating the effectiveness and robustness of our method. [E1 quote "From Table 2, we observe that our model achieves the highest SSIM and the lowest LMD, demo"]
- ~~The quantitative results in Table 2 show that our method outperforms other methods on the GRID dataset.~~ [E1 quote "From Table 2, we observe that our model achieves the highest SSIM and the lowest LMD, demo"]
- The user study on the LRW dataset indicates that our model achieves the highest rating in both realistic and synchronization. [E2 quote "Table 4 clearly demonstrates that our model achieves the highest rating in both realistic "]
- ~~Our method achieved the highest SSIM and the lowest LMD on the GRID dataset, demonstrating the effectiveness and robustness of our method.~~ [E4 cell AMIE (Ours) / Evaluation on = 0.97; E4 cell AMIE (Ours) / GRID = 0.78]
- ~~The quantitative results in Table 2 show that our method outperforms other methods on the GRID dataset.~~ [E4 cell AMIE (Ours) / Evaluation on = 0.97; E4 cell AMIE (Ours) / GRID = 0.78]
- The user study on the LRW dataset indicates that our model achieves the highest rating in both realistic and synchronization. [E2 quote "Table 4 clearly demonstrates that our model achieves the highest rating in both realistic "]

  - box p.5 Section L sentence (315.0, 571.5, 558.0, 603.3)
  - box p.5 Section L sentence (315.0, 571.5, 558.0, 603.3)
  - box p.6 Section 4.5 sentence (315.0, 453.9, 558.0, 474.8)
  - box p.6 Section 4.5 sentence (315.0, 453.9, 558.0, 474.8)

## 1812.06589v2#3: What is the role of the frame discriminator in the proposed method?

**Gold:** The frame discriminator is used to detect whether the generated frame and audio are matched or not. (Figure 2)

**Answer** (unsupported_claims_dropped): The Frame Discriminator is fed by the pairs of the real frame and audio clip { f i , a i } , or the pairs of the generated frame and corresponding audio clip { ˆ f i , a i } .

- The Frame Discriminator is fed by the pairs of the real frame and audio clip { f i , a i } , or the pairs of the generated frame and corresponding audio clip { ˆ f i , a i } . [E3 quote "Frame discriminator is fed by the pairs of the real frame and audio clip { f i , a i } , o"]
- The Frame Discriminator outputs the probability of whether the inputs (audio and frame) are matched. [E3 quote "The output of the discriminator is the probability of whether the inputs (audio and frame)"]
- The Frame Discriminator is fed by the pairs of the real frame and audio clip { f i , a i } , or the pairs of the generated frame and corresponding audio clip { ˆ f i , a i } . [E3 quote "Frame discriminator is fed by the pairs of the real frame and audio clip { f i , a i } , o"]
- The Frame Discriminator outputs the probability of whether the inputs (audio and frame) are matched. [E3 quote "The output of the discriminator is the probability of whether the inputs (audio and frame)"]
- The Frame Discriminator is fed by the pairs of the real frame and audio clip { f i , a i } , or the pairs of the generated frame and corresponding audio clip { ˆ f i , a i } . [E3 quote "Frame discriminator is fed by the pairs of the real frame and audio clip { f i , a i } , o"]
- The Frame Discriminator outputs the probability of whether the inputs (audio and frame) are matched. [E3 quote "The output of the discriminator is the probability of whether the inputs (audio and frame)"]
- ~~The Frame Discriminator is part of the proposed method and is used to distinguish between real and generated frames.~~ [E3 quote "Frame discriminator is fed by the pairs of the real frame and audio clip { f i , a i } , o"]
- ~~The Frame Discriminator consists of an Image CNN, an Audio FC, and a classifier.~~ [E3 quote "The discriminator consists of an Image CNN, an Audio FC, and a classifier."]

  - box p.3 Section 3.1 sentence (315.0, 306.5, 558.0, 327.4)
  - box p.3 Section 3.1 sentence (315.0, 341.5, 558.0, 362.4)
  - box p.3 Section 3.1 sentence (315.0, 306.5, 558.0, 327.4)
  - box p.3 Section 3.1 sentence (315.0, 341.5, 558.0, 362.4)
  - box p.3 Section 3.1 sentence (315.0, 306.5, 558.0, 327.4)
  - box p.3 Section 3.1 sentence (315.0, 341.5, 558.0, 362.4)

## 1804.07931v2#0: What are the two auxiliary tasks that are used in the ESMM architecture for CVR modeling?

**Gold:** The two auxiliary tasks are CTR and CTCVR. (Figure 2)

**Answer** (ok): The two auxiliary tasks used in the ESMM architecture for CVR modeling are CTR and CTCVR.

- The two auxiliary tasks used in the ESMM architecture for CVR modeling are CTR and CTCVR. [E3 quote "In ESMM, two auxiliary tasks of predicting the post-view click-through rate (CTR) and post"; E4 observation "The figure shows two auxiliary tasks: pCTR and pCTCVR."]

  - box p.2 Section 1 paragraph (53.8, 87.0, 295.6, 271.2)
  - box p.2 Figure 2 labels: Auxiliary Tasks, pCTR, pCTCVR (484.3, 85.1, 524.9, 114.5)

## 1804.07931v2#1: What is the relationship between clicks and impressions?

**Gold:** Clicks are a subset of impressions. (Figure 1)

**Answer** (unsupported_claims_dropped): The number of clicks is significantly smaller than the number of impressions.

- The number of clicks is significantly smaller than the number of impressions. [E3 cell Public Dataset / #impression = 84M; E3 cell Public Dataset / #click = 3.4M]
- The relationship between clicks and impressions is not one-to-one, as there are many impressions that do not result in a click. [E3 cell Public Dataset / #impression = 84M; E3 cell Public Dataset / #click = 3.4M]
- ~~The relationship between clicks and impressions is that clicks are a subset of impressions.~~ [E1 quote "Training space is composed of samples with clicked impressions. It is only part of the inf"]
- The number of clicks is significantly smaller than the number of impressions. [E3 cell Public Dataset / #impression = 84M; E3 cell Public Dataset / #click = 3.4M]
- The relationship between clicks and impressions is not one-to-one, as there are many impressions that do not result in a click. [E3 cell Public Dataset / #impression = 84M; E3 cell Public Dataset / #click = 3.4M]

  - box p.4 Table 1 cell Public Dataset / #impression (171.1, 120.7, 217.6, 132.5)
  - box p.4 Table 1 cell Public Dataset / #click (218.0, 120.7, 245.4, 132.5)
  - box p.4 Table 1 cell Public Dataset / #impression (171.1, 120.7, 217.6, 132.5)
  - box p.4 Table 1 cell Public Dataset / #click (218.0, 120.7, 245.4, 132.5)
  - box p.4 Table 1 cell Public Dataset / #impression (171.1, 120.7, 217.6, 132.5)
  - box p.4 Table 1 cell Public Dataset / #click (218.0, 120.7, 245.4, 132.5)
  - box p.4 Table 1 cell Public Dataset / #impression (171.1, 120.7, 217.6, 132.5)
  - box p.4 Table 1 cell Public Dataset / #click (218.0, 120.7, 245.4, 132.5)
