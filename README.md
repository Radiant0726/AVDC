<div align="center">
<br>
<h1>Empowering Long-form Omni-modal001 Understanding with Robust Audio Perception</h1>
</div>
  
We propose **AVDC (Audio-Visual Decoupled Captions)** dataset, a large-scale dataset designed to disentangle visual and auditory semantics, to improve omni understanding in multimodal models.
While recent multimodal models have achieved strong performance in vision-language tasks, they often struggle with fine-grained audio-visual alignment, mainly due to the lack of such structured data. 
This project addresses that gap by providing both data and training pipelines for better omni-modal perception.


## Dataset Pipeline

We propose an automatic pipeline for audio-visual decoupled caption generation, 
where multiple audio and language models extract and verify auditory information, 
temporally align it with visual content to produce segment-level captions, 
and finally aggregate them into a coherent global caption with subsequent verification and refinement.


