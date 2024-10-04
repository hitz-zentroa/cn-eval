<p align="center">
    <h2 align="center"> A LLM-based Ranking Method for the Evaluation of Automatic Counter-Narrative Generation </h2>


<p align="center">
    <a href="https://twitter.com/intent/tweet?text=Wow+this+new+model+is+amazing:&url=https%3A%2F%2Fgithub.com%2Fhitz-zentroa%2FGoLLIE"><img alt="Twitter" src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fhitz-zentroa%2FGoLLIE"></a>
    <a href="https://github.com/hitz-zentroa/cn-eval/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/hitz-zentroa/cn-eval"></a>
    <a href="https://hitz-zentroa.github.io/cn-eval/"><img alt="Blog" src="https://img.shields.io/badge/📒-Blog Post-blue"></a>
    <a href="https://arxiv.org/abs/2406.15227"><img alt="Paper" src="https://img.shields.io/badge/📖-Paper-orange"></a>
<br>
     <a href="http://www.hitz.eus/"><img src="https://img.shields.io/badge/HiTZ-Basque%20Center%20for%20Language%20Technology-blueviolet"></a>
    <a href="http://www.ixa.eus/?language=en"><img src="https://img.shields.io/badge/IXA-%20NLP%20Group-ff3333"></a>
    <br>
     <br>
</p>

<p align="justify">

This repository contains the code accompanying our paper, "An LLM-based Ranking Method for the Evaluation of Automatic Counter-Narrative Generation".

In this work, we address the limitations of traditional evaluation metrics for counter-narrative (CN) generation, such as BLEU, ROUGE, and BERTScore. These conventional metrics often overlook a critical factor: the specific hate speech (HS) that the counter-narrative is responding to. Without accounting for this context, it's challenging to truly assess the quality of generated CNs.

To tackle this, we propose a novel automatic evaluation approach. Our method uses a pairwise, tournament-style ranking system to evaluate CNs, relying on JudgeLM—a specialized language model trained to assess text quality. JudgeLM allows us to compare CNs directly without human intervention, transforming the subjective task of CN evaluation into manageable binary classification problems.

For thorough validation, we test our approach on two distinct datasets: CONAN and CONAN-MT, ensuring that our method generalizes well across different corpora.

Below, you can find a visualization of the correlation matrix, which demonstrates the effectiveness of our ranking method in comparison with traditional metrics.
</p>

<p align="center">
<img src="img/total_spearman.png">
</p>



## Citation
```bibtex
@inproceedings{
}
```
