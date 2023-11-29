# Text controlled Photorealistic Image generation for Indian Roads

We use [stable diffusion](https://github.com/CompVis/stable-diffusion/tree/main) model. Thanks to [dataset](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiZ9Mra6NmCAxVqc_UHHbdEBqkQFnoECAsQAQ&url=https%3A%2F%2Fidd.insaan.iiit.ac.in%2F&usg=AOvVaw2KScr8t0QB89GahDBS3BL7&opi=89978449) made by IIIT hyderabad we were able to finetune the stale diffusion model on properly annotated dataset with different views(rear,frot,left side, right side). We have finetuned stable diffusion model on 30% percent of images using LoRA-finetuning. Thanks to [notebook](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image) provided from hugging-face. 

## IN-PROGRESS WRK:
### Objective : 
Generate object at position as given in text command. example: Generate red car at position [x,y] --> should generate image of red car at that position.

### Approach : 
We have bounding box of objects in the Image. We create prompts (object1: position1 , object2: position2) as input and model generates image based on the prompt. Now we add an object detection module over generated image that identifies object position1' and position2' in image which are compared to the input position1 and position2. 

      Loss = Sum(abs(position1' - position1'), abs(position2'-position2))

### Hypothesis :
This loss should reinforce model to generate predictions with given position in prommpt  thus enhanching control over posotion of generation with text.

-------------

### Performance of finetuned stable diffusion model
Prompt : Crowded Afternoon on road

![heavy_traffic_streets_2](https://github.com/ritikbhardwaj1296/hackathon/assets/90241581/dbd72e86-cc12-42e3-ae36-eaf0f97b7abc)


Prompt: Streets on festival day celebration (original prompt wasn't exactly this but meaning was same ; it was generated from chatgpt when asked to give a prompt of intermediate difficulty for generation )

![chat_gpt_2](https://github.com/ritikbhardwaj1296/hackathon/assets/90241581/b2b037f9-122a-4802-9b7a-2258699978b8)

Prompt: Heavy trafiic on streets on rainy day

![Heavy_traffic_and_rain](https://github.com/ritikbhardwaj1296/hackathon/assets/90241581/d9ad59ac-43d5-46e2-803f-70cd6f170452)


