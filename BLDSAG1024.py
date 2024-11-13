import torch
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from transformers import SamModel, SamProcessor
from PIL import Image
from diffusers import DDIMScheduler
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)
import torchvision.transforms as T
import torch.nn.functional as F

# CrossAttnStoreProcessor 클래스 정의 (SAG에 필요)
class CrossAttnStoreProcessor:
    def __init__(self):
        # 어텐션 확률을 저장할 변수 초기화
        self.attention_probs = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        # Query 생성
        query = attn.to_q(hidden_states)

        # Encoder Hidden States 설정
        encoder_hidden_states = encoder_hidden_states or hidden_states
        if attn.norm_cross and encoder_hidden_states is not hidden_states:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # Key와 Value 생성
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Query, Key, Value를 헤드 차원으로 변환
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # 어텐션 확률 계산 및 저장
        self.attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # 어텐션 가중치와 Value를 곱하여 Hidden States 업데이트
        hidden_states = torch.bmm(self.attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # 선형 변환 및 드롭아웃 적용
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

# BLDCSAG1024 클래스 정의
class BLDCSAG1024:
    def __init__(self,model_path, controlnet, prompt, negative_prompt, blending_start_percentage, device):
        # 초기화: 입력된 파라미터들을 인스턴스 변수로 저장
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.init_image = 'image.png'
        self.mask = 'mask.png'
        self.model_path = model_path
        self.blending_start_percentage = blending_start_percentage
        self.device = device
        self.output_path = 'output.png'
        self.latent_list = []
        self.degraded_latents_list = []
        # Stable Diffusion 2.1 파이프라인 로드 (UNet 및 ControlNet 포함)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_path,
            controlnet= controlnet,
            torch_dtype=torch.float16
        ).to(self.device)

        # VAE, UNet, 텍스트 인코더, 토크나이저, 스케줄러 로드
        self.vae = self.pipe.vae.to(self.device)
        self.unet = self.pipe.unet
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder.to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(self.model_path, subfolder="scheduler")

    @torch.no_grad()
    def generate_image(
        self,
        height=1024,
        width=1024,
        kernel_size = 1,
        num_inference_steps=100,
        guidance_scale=7.5,
        generator=None,
        sag_scale=0.8,
        seed = None,
    ):
        # 랜덤 시드 설정
        if generator is None:
            generator = torch.Generator(device=self.device)
            if seed is None:
                generator.manual_seed(random.randint(1, 2147483647))
            else :
                generator.manual_seed(seed)


        # 배치 크기 설정 (고정값 1)
        batch_size = 1
        prompts = self.prompt
        negative_prompts = self.negative_prompt

        # 2. 이미지 로드 및 리사이즈
        image = Image.open(self.init_image).convert("RGB").resize((width, height), Image.BILINEAR)
        image_np = np.array(image)

        # 3. Canny Edge 이미지 생성
        canny_image = self._create_canny_image(image_np)
        Image.fromarray(canny_image).save('canny.png')
        controlnet_cond = self._prepare_control_image(canny_image)

        # 4. 원본 이미지를 latent space로 변환
        source_latents = self._image2latent(image)

        # 5. 마스크 로드 및 처리
        latent_mask = self._read_mask(self.mask, dest_size=(height // 8, width // 8))
        mask_np = latent_mask.squeeze().cpu().numpy().astype(np.uint8)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded_mask_np = cv2.dilate(mask_np, kernel, iterations=1)
        eroded_mask = torch.tensor(eroded_mask_np).unsqueeze(0).unsqueeze(0).to(self.device).half()
        

        # 6. 텍스트 임베딩 생성
        text_embeddings = self._get_text_embeddings(prompts)
        uncond_embeddings = self._get_text_embeddings(negative_prompts)

        # 초기 Latent 설정
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, height // 8, width // 8),       # latent의 batch size, channel, dimension 정의
            generator=generator,                                                       # 시드 고려
            device=self.device,                                                        # 연산 어디서?
            dtype=torch.float16                                                        # data type 결정
        )

        # 타임스텝 설정
        self.scheduler.set_timesteps(num_inference_steps)                             # num_inference_steps에 따른 time steps 정의
        timesteps = self.scheduler.timesteps

        # batch_size에 맞도록 텐서 크기 맞추기 (batch_size만큼 반복 생성)
        source_latents = source_latents.repeat(batch_size, 1, 1, 1)
        controlnet_cond = controlnet_cond.repeat(batch_size, 1, 1, 1).to(self.device).half()

        # Initialization for SAG
        store_processor = CrossAttnStoreProcessor()
        original_attn_processors = self.unet.attn_processors                # save attention processors of Unet 
        map_size = None                                                     # initialize the size of attention map 

        # function to get attention map size can be used in generate_image() function
        def get_map_size(module, input, output):
            nonlocal map_size
            if isinstance(output, tuple):            # If the type of output is tuple 
                output_tensor = output[0]            # get the contents of the first index 
            else:
                output_tensor = output
            map_size = output_tensor.shape[-2:]     # take the contents of the last two indices, i.e. h, w from (batch_size, channels, h, w)

        # 어텐션 프로세서와 후크 등록
        self.unet.mid_block.attentions[0].transformer_blocks[0].attn1.processor = store_processor
        self.unet.mid_block.attentions[0].register_forward_hook(get_map_size)

        # 타임스텝 루프 시작
        blending_start_step = int(len(timesteps) * self.blending_start_percentage)        # 전체 time step에서 blending percentage 곱해 blending start stage 지정 
        for i, t in enumerate(timesteps):
            
            # latent map 저장
            # if i % 10 == 0 :
            #     self.latent_list.append(latents)
            
            # time step t에서 Latent model input 저장
            latent_model_input = self.scheduler.scale_model_input(latents, t)

            # CFG를 위해 복제 -> 배치 크기를 2배로 늘리고, 하나는 conditional prediction에, 나머지 하나는 unconditional prediction에 사용 
            latent_model_input = torch.cat([latent_model_input] * 2, dim=0)
            controlnet_cond_in = torch.cat([controlnet_cond] * 2, dim=0)
            
            # unconditional할 때와 conditional(text_embeddings) 때의 텍스트 임베딩을 결합
            combined_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)

            # ControlNet 적용
            controlnet_output = self.pipe.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=combined_embeddings,
                controlnet_cond=controlnet_cond_in,                
                return_dict=False,        # controlnet의 출력을 딕셔너리 형태가 아닌 튜플 형태로 반환
            )

            # UNet을 통해 노이즈 예측, 괄호 안 변수를 input으로 취하고, .sample을 사용하여 예측된 노이즈 텐서를 output으로 반환
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=combined_embeddings,
                down_block_additional_residuals=controlnet_output[0],
                mid_block_additional_residual=controlnet_output[1],
            ).sample           

            # 예측한 노이즈를 Unconditional & Conditional 분리
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            
            # CFG 적용한 noise 다시 예측
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # attention_probs 메서드 이용하여 score 계산, 이전에 저장한 store_processor 반으로 분리 -> 두 가지 상황(unconditional, conditional)에 따른 어텐션 맵 저장 
            uncond_attn, cond_attn = store_processor.attention_probs.chunk(2)
            uncond_attn = uncond_attn.detach()     # uncond_attn을 추후의 연산에서 사용할 때, 그 값이 자동 미분에 포함되지 않도록 
            
            # store_processor에 저장된 어텐션 확률 초기화
            store_processor.attention_probs = None     

            # SAG 적용
            if sag_scale > 0.0:
                # x0와 epsilon 예측
                pred_x0 = self.pred_x0(latents, noise_pred_uncond, t)
                eps = self.pred_epsilon(latents, noise_pred_uncond, t)

                # SAG 마스킹 이용한 noise prediction
                degraded_latents = self.sag_masking(pred_x0, uncond_attn, map_size, t, eps)

                # latent map 저장
                # if i % 10 == 0 :
                #     self.degraded_latents_list.append(degraded_latents)
                
                # Degraded 입력 준비
                degraded_latent_model_input = self.scheduler.scale_model_input(degraded_latents, t)

                # Degraded 입력에 대한 ControlNet 적용
                degraded_controlnet_output = self.pipe.controlnet(
                    degraded_latent_model_input,
                    t,
                    encoder_hidden_states=uncond_embeddings,
                    controlnet_cond=controlnet_cond,
                    return_dict=False,
                )

                # UNet 적용 (Unconditional embeddings 사용)
                degraded_noise_pred = self.unet(
                    degraded_latent_model_input,
                    t,
                    encoder_hidden_states=uncond_embeddings,
                    down_block_additional_residuals=degraded_controlnet_output[0],
                    mid_block_additional_residual=degraded_controlnet_output[1],
                ).sample

                # noise_pred 업데이트
                noise_pred += sag_scale * (noise_pred - degraded_noise_pred)

            # latents 업데이트
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            
            
            # background 마스크 이용해서 Blended latent diffusion denoise step 진행 
            # latents에는 latent space의 정보 -> background 마스크 영역만 latent 정보 가져오기 
            # source_latents에는 원본이미지의 latent space 차원 정보
            if i >= blending_start_step:
                    latents = latents * eroded_mask + source_latents * (1 - eroded_mask)  #eroded_mask_2 or eroded_mask
        
          # (128, 128) 형태

        # 이미지 출력
        # for i in range(10):
        #     # 새 figure를 만듦
        #     fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        #     fig.suptitle(f"Latent Image Set {i+1}")
            
        #     # 4개의 채널을 2x2 subplot에 표시
        #     for j in range(4):
        #         # 각 채널을 GPU에서 CPU로 옮기고, numpy로 변환
        #         image_tensor = self.latent_list[i].squeeze().cpu().numpy()
                
        #         # 서브플롯에 이미지 출력
        #         row, col = divmod(j, 2)  # 2x2 위치 계산
        #         axes[row, col].imshow(image_tensor[j], cmap='gray')
        #         axes[row, col].set_title(f"Channel {j+1}")
        #         axes[row, col].axis('off')
            
        #     # 여백 조정 및 표시
        #     plt.tight_layout()
        #     plt.subplots_adjust(top=0.9)  # 제목 공간 확보
        #     plt.show()
            
        # for i in range(10):
        #     # 새 figure를 만듦
        #     fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        #     fig.suptitle(f"Degraded_Latent Image Set {i+1}")
            
        #     # 4개의 채널을 2x2 subplot에 표시
        #     for j in range(4):
        #         # 각 채널을 GPU에서 CPU로 옮기고, numpy로 변환
        #         image_tensor = self.degraded_latents_list[i].squeeze().cpu().numpy()
                
        #         # 서브플롯에 이미지 출력
        #         row, col = divmod(j, 2)  # 2x2 위치 계산
        #         axes[row, col].imshow(image_tensor[j], cmap='gray')
        #         axes[row, col].set_title(f"Channel {j+1}")
        #         axes[row, col].axis('off')
            
        #     # 여백 조정 및 표시
        #     plt.tight_layout()
        #     plt.subplots_adjust(top=0.9)  # 제목 공간 확보
        #     plt.show()

        # 원래의 어텐션 프로세서로 복원
        self.unet.set_attn_processor(original_attn_processors)

        # Latents를 이미지로 디코딩
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        # 후처리 및 반환
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")

        return images

    @torch.no_grad()
    def _image2latent(self, image):
        # 이미지를 텐서로 변환하고 정규화
        image = np.array(image).astype(np.float32) / 127.5 - 1
        image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(self.device).half()
        # VAE를 통해 Latent로 인코딩
        latents = self.vae.encode(image)["latent_dist"].mean
        latents = latents * 0.18215

        return latents

    def _read_mask(self, mask_path: str, dest_size=(128, 128)):
        # 마스크 이미지 로드 및 이진화
        mask = Image.open(mask_path).convert("L").resize(dest_size, Image.NEAREST)
        mask = np.array(mask) / 255.0
        mask = (mask >= 0.5).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device).half()
        return mask

    def _create_canny_image(self, image):
        # OpenCV를 사용하여 Canny Edge 이미지 생성
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_image, 100, 200)
        return edges

    def _prepare_control_image(self, image):
        # 이미지를 텐서로 변환 및 전처리
        if not isinstance(image, torch.Tensor):
            image = np.array(image)
            if len(image.shape) == 2:
                image = image[:, :, None]
            image = image.transpose(2, 0, 1)  # (C, H, W)
            if image.shape[0] == 1:
                image = np.repeat(image, repeats=3, axis=0)
            image = image / 255.0
            image = torch.from_numpy(image).float()
        return image.to(self.device).half()

    def _get_text_embeddings(self, text):
        # 텍스트 임베딩 생성
        text_input = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_embeddings

    # sample: latent_inputs, model_output: UNet의 output (제거해야할 노이즈 정보)
    # alpha_prod_t: 노이즈 감소율, beta_prod_t 남아있는 노이즈 비율
    
    # pred_x0 함수 정의(노이즈 모두 제거한 최종 이미지)
    def pred_x0(self, sample, model_output, timestep):
        # 알파 및 베타 값 계산
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        # 예측 타입에 따라 pred_original_sample 계산
        if self.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        elif self.scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t ** 0.5 * sample - beta_prod_t ** 0.5 * model_output
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")
        return pred_original_sample

    # pred_epsilon 함수 정의
    def pred_epsilon(self, sample, model_output, timestep):
        # 알파 및 베타 값 계산
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        # 예측 타입에 따라 pred_eps 계산
        if self.scheduler.config.prediction_type == "epsilon":     # 남아있는 노이즈의 정보를 주면
            pred_eps = model_output
        elif self.scheduler.config.prediction_type == "sample":    # 노이즈가 제거된 이후의 정보(분포)를 주면 
            pred_eps = (sample - alpha_prod_t ** 0.5 * model_output) / beta_prod_t ** 0.5
        elif self.scheduler.config.prediction_type == "v_prediction":      # 노이즈와 샘플의 혼합
            pred_eps = beta_prod_t ** 0.5 * sample + alpha_prod_t ** 0.5 * model_output
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")
        return pred_eps

    # sag_masking 함수 정의
    def sag_masking(self, original_latents, attn_map, map_size, t, eps):
        # 어텐션 맵의 크기 및 Latent 크기 가져오기
        bh, hw1, hw2 = attn_map.shape
        b, latent_channel, latent_h, latent_w = original_latents.shape
        h = self.unet.config.attention_head_dim
        if isinstance(h, list):
            h = h[-1]

        # 어텐션 마스크 생성
        attn_map = attn_map.reshape(b, h, hw1, hw2)
        attn_mask = attn_map.mean(1).sum(1) > 1.0
        attn_mask = attn_mask.reshape(b, map_size[0], map_size[1]).unsqueeze(1).repeat(1, latent_channel, 1, 1).type(attn_map.dtype)
        attn_mask = F.interpolate(attn_mask, (latent_h, latent_w), mode='nearest')

        # 블러 적용
        degraded_latents = self.gaussian_blur_2d(original_latents, kernel_size=9, sigma=1.0)
        
        # attention map에서 1인 부분은 gaussian_blur 처리, 0인 부분은 원래 latent -> attention score 낮은 곳에 latent 집중해서 골고루 attention하여 generation 가능하도록 
        degraded_latents = degraded_latents * attn_mask + original_latents * (1 - attn_mask)

        # 블러처리된 latent 정보, eps(남아 있는 노이즈의 양) 정보 활용하여, 기존의 노이즈와 새로 더해진 노이즈가 합쳐 -> 다음 스텝에서 사용할 노이즈 업데이트
        degraded_latents = self.scheduler.add_noise(degraded_latents, noise=eps, timesteps=t[None])

        return degraded_latents

    # gaussian_blur_2d 함수 정의
    def gaussian_blur_2d(self, img, kernel_size, sigma):
        # 가우시안 커널 생성
        ksize_half = (kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        x_kernel = pdf / pdf.sum()
        x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)
        kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
        kernel2d = kernel2d.expand(img.shape[1], 1, kernel2d.shape[0], kernel2d.shape[1])
        padding = kernel_size // 2
        # 이미지에 패딩 적용 및 컨볼루션
        img = F.pad(img, (padding, padding, padding, padding), mode="reflect")
        img = F.conv2d(img, kernel2d, groups=img.shape[1])
        return img



from segment_anything_hq import sam_model_registry, SamPredictor
class SamHQImageProcessor:
    def __init__(self, img_path, device, coord = (2,2,14,14), model_type='vit_h', sam_checkpoint='pretrained_checkpoint/sam_hq_vit_h.pth'):
        self.img_path = img_path
        self.device = device
        # SAM 모델과 predictor 로드
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.x1, self.y1, self.x2, self.y2 = coord
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        # 이미지 로드
        self.pf_image = self._load_image()

    def _load_image(self):
        # 이미지 로드 및 준비
        image = Image.open(self.img_path).convert("RGB")
        image_np = np.array(image)
        return image_np

    def process_image(self):
        # predictor에 이미지 설정
        self.predictor.set_image(self.pf_image)
        
        # 경계 박스를 입력 프롬프트로 사용하여 세그멘테이션 수행
        h, w, _ = self.pf_image.shape
        box = np.array([[self.x1 * w / 16, self.y1 * h / 16], [self.x2 * w / 16, self.y2 * h / 16]])
        
        # 마스크 예측
        masks, scores, logits = self.predictor.predict(
            box=box,  # 경계 박스를 프롬프트로 사용
            multimask_output=True
        )
        # 점수가 가장 높은 마스크 선택
        max_score_index = np.argmax(scores)
        max_score_mask = masks[max_score_index]
        return max_score_mask

    def save_mask_image(self, max_score_mask):
        # 마스크 이미지를 저장
        mask_resized = max_score_mask.astype(np.uint8)
        mask_image = np.where(
            mask_resized.reshape(mask_resized.shape[0], mask_resized.shape[1], 1) != 0,
            [0, 0, 0, 255],
            [255, 255, 255, 0]
        ).astype(np.uint8)
        mask_image_pil = Image.fromarray(mask_image, 'RGBA')
        mask_image_pil.save('mask.png')

    def save_masked_image(self, max_score_mask):
        # 마스크된 이미지를 저장
        mask_resized = max_score_mask.astype(np.uint8)
        masked_image = self.pf_image * mask_resized[:, :, np.newaxis]
        alpha_channel = (mask_resized * 255).astype(np.uint8)
        masked_image_rgba = np.dstack((masked_image, alpha_channel))
        masked_image_pil = Image.fromarray(masked_image_rgba, 'RGBA')
        masked_image_pil.save('image.png')

    def run(self):
        # 전체 프로세스를 실행
        max_score_mask = self.process_image()
        self.save_mask_image(max_score_mask)
        self.save_masked_image(max_score_mask)



class ImageGrid:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = Image.open(image_path)
        self.width, self.height = self.img.size
        self.grid_size_x = self.width / 16
        self.grid_size_y = self.height / 16

    def draw_grid(self):
        # 이미지와 격자 표시하기
        fig, ax = plt.subplots()
        ax.imshow(self.img)

        # 격자 그리기
        for i in range(17):  # 16개의 셀을 위해 17개의 선이 필요
            # 수직선 그리기
            ax.add_line(plt.Line2D((i * self.grid_size_x, i * self.grid_size_x), (0, self.height), color="red"))
            # 수평선 그리기
            ax.add_line(plt.Line2D((0, self.width), (i * self.grid_size_y, i * self.grid_size_y), color="red"))

        # 격자 번호 추가
        for i in range(17):
            # 왼쪽에 행 번호 추가 (0부터 시작)
            ax.text(-self.grid_size_x / 2, (i) * self.grid_size_y, str(i), va="center", ha="center", color="blue", fontsize=8)
            # 위쪽에 열 번호 추가 (0부터 시작)
            ax.text((i) * self.grid_size_x, -self.grid_size_y / 2, str(i), va="center", ha="center", color="blue", fontsize=8)

        # 표시하기
        plt.axis("off")
        plt.show()



# ImageGridDisplay 클래스 정의
class ImageGridDisplay:
    def __init__(self, img1_path, img2_path, img3_path, img4_path):
        # 이미지 경로 저장
        self.img1_path = img1_path
        self.img2_path = img2_path
        self.img3_path = img3_path
        self.img4_path = img4_path
        self.images = []
        # 이미지 로드 메서드 호출
        self.load_images()

    def load_images(self):
        # 이미지를 OpenCV로 로드하고 RGB로 변환하여 저장
        self.images.append(cv2.cvtColor(cv2.imread(self.img1_path), cv2.COLOR_BGR2RGB))
        self.images.append(cv2.cvtColor(cv2.imread(self.img2_path), cv2.COLOR_BGR2RGB))
        self.images.append(cv2.cvtColor(cv2.imread(self.img3_path), cv2.COLOR_BGR2RGB))
        self.images.append(cv2.cvtColor(cv2.imread(self.img4_path), cv2.COLOR_BGR2RGB))

    def display(self):
        # 이미지를 2x2 그리드로 디스플레이
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        titles = ['Original', 'Mask', 'Canny', 'Output']
        for i, ax in enumerate(axs.flat):
            ax.imshow(self.images[i])
            ax.set_title(titles[i])
            ax.axis('off')

        plt.tight_layout()
        plt.show()