# SmolVLM for Object Detection

Updates so far:

- Special Tokens:
    - adding special tokens didn't work unfortunately as this is how the project had started, the model collapsed and never learnt to predict the special tokens.
    - switched from 1024 tokens to 256 tokens, still the same issue.
    - inspired by [Aritra's Gemma 3 Finetuning](https://x.com/ariG23498/status/1922606702462894531) I then switched to just normal text instead of special tokens.
    - I even considered averaging out 

- Training
    - model: [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) -- its just a really good model for the size and for the GPU poors, what can I say
    - I trained it on COCO for 7 epochs took about 10hours with < 10gigs of VRAM.
    - As you know there are a LOT of bboxes in 1 COCO image so for each class I chose the bbox with the maximum area to make it easy on the model so its decent at detecting large objects.
    - There's option to enable synonyms as well which I generated for each label to make it "open vocabulary" -- didnt include in this so please update the trainer.py to get rid of that if you want to try it :)
    
Here are the basic results:




Still a long way to go, but hey atleast it works :)


```
A man's heart plans his way, But the LORD directs his steps. Proverbs 16:9
```