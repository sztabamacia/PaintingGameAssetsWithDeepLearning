# Human and machine collaboration for painting game assets with deep learning
This is a code and data repository for the above publication. Feel free to ask any questions using the issue tracker or by emailing me directly.

**Abstract:** Modern games are among the most intricate pieces of software ever devised. However, artists have little to no means of porting their creative work from title to title as the aesthetics change and older models grow to look obsolete. Zooming into the topic of assisted game art generation, the literature is notably scarce, although advances towards automated asset generation are of paramount interest to the field. In this work, we investigate the use of deep learning algorithms to create pixel art sprites from line art sketches to produce artwork of sufficient quality to be used within a game product with little to no manual editing by human artists. Such a problem contrasts with well-known tasks studied in the literature, which are based on natural pictures, boast massive datasets, and are much more tolerant to noise. In addition, we conducted a case study of applying current technology to the drawing pipeline of an upcoming game title, attaining useful and positive results that may fast-track the game development, supporting the argument that current image generation state-of-the-art is ready to be used in some real-world tasks.

**Publication link:** https://www.sciencedirect.com/science/article/abs/pii/S1875952122000210

**License:** all code and data available in this repository is released under GPL-3 and is meant for educational and academic purpuses only. All images belong to Onanim Studios and should not be used for any commercial purpuses by third parties. Use of the data by other academic works is fully endorsed and we encourage you to contact us if you need help or to simply share your research with us.

**Citation:**
```
@article{paintinggameassetswithdeeplearning,
    title = {Human and Machine Collaboration for Painting Game Assets with Deep Learning},
    journal = {Entertainment Computing},
    pages = {100497},
    year = {2022},
    issn = {1875-9521},
    doi = {https://doi.org/10.1016/j.entcom.2022.100497},
    author = {Ygor Rebouças Serpa and Maria Andréia Formico Rodrigues},
    keywords = {Machine Learning, Deep Learning, Procedural Content Generation, Image Generation, Games},
}
```

## Getting Started

This repository contains the basic steps to reproduce our results. All data related to the Lucy, Sarah and Saulo characters can be found at "./dataset/" and all code is located at the "./research/" folder. All experiments can be reproduced by configuring the train.py script following the directions provided on the article. All generated results are saved to the "./results/" folder and are named by "task_name" and are timestamped for ease of use.

For any help reproducing the results or trying to extend our solution, feel free to contact Ygor directly or use the issue tracker.