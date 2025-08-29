"""
References:
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing
"""

import torch
from dit import DiT_models
from vae import VAE_models
from torchvision.io import read_video, write_video
from utils import load_prompt, load_actions, sigmoid_beta_schedule
from tqdm import tqdm
from einops import rearrange
from torch import autocast
from safetensors.torch import load_model
import argparse
from pprint import pprint
import os
import logging

logger = logging.getLogger(__name__)


assert torch.cuda.is_available()
device = "cuda:0"


class SingleFrameGenerator:
    """
    单帧生成器类 - 每次调用只生成一帧图像
    """
    def __init__(self, model, vae, prompt_path, n_prompt_frames=1, video_offset=None, ddim_steps=10):
        self.model = model
        self.vae = vae
        self.ddim_steps = ddim_steps
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # sampling params
        self.max_noise_level = 1000
        self.ddim_noise_steps = ddim_steps
        self.noise_range = torch.linspace(-1, self.max_noise_level - 1, self.ddim_noise_steps + 1).to(self.device)
        self.noise_abs_max = 20
        self.stabilization_level = 15
        self.scaling_factor = 0.07843137255
        
        # get alphas
        betas = sigmoid_beta_schedule(self.max_noise_level).float().to(self.device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod = rearrange(self.alphas_cumprod, "T -> T 1 1 1")
        
        # Load and encode prompt
        prompt = load_prompt(prompt_path, video_offset=video_offset, n_prompt_frames=n_prompt_frames)
        prompt = prompt.to(self.device)
        
        # VAE encoding
        self.B = prompt.shape[0]
        H, W = prompt.shape[-2:]
        prompt = rearrange(prompt, "b t c h w -> (b t) c h w")
        with torch.no_grad():
            with autocast("cuda", dtype=torch.half):
                prompt = self.vae.encode(prompt * 2 - 1).mean * self.scaling_factor
        self.x = rearrange(prompt, "(b t) (h w) c -> b t c h w", t=n_prompt_frames, h=H // self.vae.patch_size, w=W // self.vae.patch_size)
        
        self.frame_count = n_prompt_frames  # Track current frame count
        
    def generate_next_frame(self, action_tensor):
        """
        生成下一帧图像
        
        Args:
            action_tensor: Action tensor (B, 1, action_dim) 或 (1, action_dim)
            
        Returns:
            Decoded frame tensor (H, W, C)
        """
        # Ensure action_tensor has correct shape
        if action_tensor.dim() == 2:
            action_tensor = action_tensor.unsqueeze(1)  # Add time dimension
        action_tensor = action_tensor.to(self.device)
        
        # Create actions sequence for current frame count
        current_frame_count = self.frame_count + 1
        actions_for_model = torch.zeros((1, current_frame_count, 25)).to(self.device)
        actions_for_model[:, -1:] = action_tensor[:, -1:]
        
        # Generate noise for new frame
        chunk = torch.randn((self.B, 1, *self.x.shape[-3:]), device=self.device)
        chunk = torch.clamp(chunk, -self.noise_abs_max, +self.noise_abs_max)
        
        # Concatenate with existing frames
        x_with_new = torch.cat([self.x, chunk], dim=1)
        start_frame = max(0, current_frame_count - self.model.max_frames)
        
        # DDIM sampling loop
        for noise_idx in reversed(range(1, self.ddim_noise_steps + 1)):
            # set up noise values
            t_ctx = torch.full((self.B, self.frame_count), self.stabilization_level - 1, dtype=torch.long, device=self.device)
            t = torch.full((self.B, 1), self.noise_range[noise_idx], dtype=torch.long, device=self.device)
            t_next = torch.full((self.B, 1), self.noise_range[noise_idx - 1], dtype=torch.long, device=self.device)
            t_next = torch.where(t_next < 0, t, t_next)
            t = torch.cat([t_ctx, t], dim=1)
            t_next = torch.cat([t_ctx, t_next], dim=1)

            # sliding window
            x_curr = x_with_new.clone()
            x_curr = x_curr[:, start_frame:]
            t = t[:, start_frame:]
            t_next = t_next[:, start_frame:]

            # get model predictions
            with torch.no_grad():
                with autocast("cuda", dtype=torch.half):
                    v = self.model(x_curr, t, actions_for_model[:, start_frame : current_frame_count])

            x_start = self.alphas_cumprod[t].sqrt() * x_curr - (1 - self.alphas_cumprod[t]).sqrt() * v
            x_noise = ((1 / self.alphas_cumprod[t]).sqrt() * x_curr - x_start) / (1 / self.alphas_cumprod[t] - 1).sqrt()

            # get frame prediction
            alpha_next = self.alphas_cumprod[t_next]
            alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
            if noise_idx == 1:
                alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
            x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
            x_with_new[:, -1:] = x_pred[:, -1:]

        # decode the newly generated frame
        latest_frame = x_with_new[:, -1:]
        latest_frame_reshaped = rearrange(latest_frame, "b t c h w -> (b t) (h w) c")
        with torch.no_grad():
            decoded_frame = (self.vae.decode(latest_frame_reshaped.float() / self.scaling_factor) + 1) / 2
        decoded_frame = rearrange(decoded_frame, "(b t) c h w -> b t h w c", t=1)
        
        # Update internal state
        self.x = x_with_new  # Keep all frames for context
        self.frame_count += 1
        
        return decoded_frame[0, 0]  # Return single frame (H, W, C)
        
    def get_prompt_frame(self, frame_idx=0):
        """获取prompt帧作为参考"""
        prompt_frame = self.x[:, frame_idx:frame_idx+1]
        prompt_frame_reshaped = rearrange(prompt_frame, "b t c h w -> (b t) (h w) c")
        with torch.no_grad():
            decoded_prompt = (self.vae.decode(prompt_frame_reshaped.float() / self.scaling_factor) + 1) / 2
        decoded_prompt = rearrange(decoded_prompt, "(b t) c h w -> b t h w c", t=1)
        return decoded_prompt[0, 0]


def generate_single_frame(model, vae, prompt_frames, action_tensor, ddim_steps=10):
    """
    Generate a single frame given prompt frames and action.
    
    Args:
        model: The DiT model
        vae: The VAE model
        prompt_frames: Encoded prompt frames tensor (B, T, C, H, W)
        action_tensor: Action tensor for current frame (B, 1, action_dim)
        ddim_steps: Number of DDIM steps
        
    Returns:
        Single decoded frame tensor (H, W, C)
    """
    device = prompt_frames.device
    B = prompt_frames.shape[0]
    
    # sampling params
    max_noise_level = 1000
    ddim_noise_steps = ddim_steps
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1).to(device)
    noise_abs_max = 20
    stabilization_level = 15
    scaling_factor = 0.07843137255
    
    # get alphas
    betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")
    
    # Generate noise for new frame
    chunk = torch.randn((B, 1, *prompt_frames.shape[-3:]), device=device)
    chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
    
    # Concatenate with prompt frames
    x = torch.cat([prompt_frames, chunk], dim=1)
    current_frame_idx = prompt_frames.shape[1]  # Index of the frame we're generating
    start_frame = max(0, current_frame_idx + 1 - model.max_frames)
    
    # DDIM sampling loop
    for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
        # set up noise values
        t_ctx = torch.full((B, current_frame_idx), stabilization_level - 1, dtype=torch.long, device=device)
        t = torch.full((B, 1), noise_range[noise_idx], dtype=torch.long, device=device)
        t_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
        t_next = torch.where(t_next < 0, t, t_next)
        t = torch.cat([t_ctx, t], dim=1)
        t_next = torch.cat([t_ctx, t_next], dim=1)

        # sliding window
        x_curr = x.clone()
        x_curr = x_curr[:, start_frame:]
        t = t[:, start_frame:]
        t_next = t_next[:, start_frame:]

        # get model predictions
        with torch.no_grad():
            with autocast("cuda", dtype=torch.half):
                v = model(x_curr, t, action_tensor[:, start_frame : current_frame_idx + 1])

        x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
        x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / (1 / alphas_cumprod[t] - 1).sqrt()

        # get frame prediction
        alpha_next = alphas_cumprod[t_next]
        alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
        if noise_idx == 1:
            alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
        x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
        x[:, -1:] = x_pred[:, -1:]

    # decode the newly generated frame
    latest_frame = x[:, -1:]
    latest_frame_reshaped = rearrange(latest_frame, "b t c h w -> (b t) (h w) c")
    with torch.no_grad():
        decoded_frame = (vae.decode(latest_frame_reshaped.float() / scaling_factor) + 1) / 2
    decoded_frame = rearrange(decoded_frame, "(b t) c h w -> b t h w c", t=1)
    
    return decoded_frame[0, 0], x  # Return decoded frame and updated encoded frames


def generate_video_frames(model, vae, prompt_path, actions_generator=None, actions_path=None, 
                         max_frames=1000, n_prompt_frames=1, video_offset=None, ddim_steps=10, 
                         stop_action_key=None, output_path=None, fps=20):
    """
    Generate video frames as a generator that can be stopped by specific action.
    
    Args:
        model: The DiT model
        vae: The VAE model  
        prompt_path: Path to prompt image/video
        actions_generator: Generator function that yields (action_tensor, action_dict) tuples
        actions_path: Path to actions file (used if actions_generator is None)
        max_frames: Maximum number of frames to generate
        n_prompt_frames: Number of prompt frames
        video_offset: Video offset for reading prompt
        ddim_steps: Number of DDIM steps
        stop_action_key: Action key that triggers stopping (e.g., 'ESC')
        output_path: Path to save video when stopping
        fps: FPS for saved video
        
    Yields:
        (frame_tensor, frame_idx, should_stop) tuples
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # sampling params
    max_noise_level = 1000
    ddim_noise_steps = ddim_steps
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1)
    noise_abs_max = 20
    stabilization_level = 15

    # get prompt image/video
    x = load_prompt(
        prompt_path,
        video_offset=video_offset,
        n_prompt_frames=n_prompt_frames,
    )

    # sampling inputs
    x = x.to(device)

    # vae encoding
    B = x.shape[0]
    H, W = x.shape[-2:]
    scaling_factor = 0.07843137255
    x = rearrange(x, "b t c h w -> (b t) c h w")
    with torch.no_grad():
        with autocast("cuda", dtype=torch.half):
            x = vae.encode(x * 2 - 1).mean * scaling_factor
    x = rearrange(x, "(b t) (h w) c -> b t c h w", t=n_prompt_frames, h=H // vae.patch_size, w=W // vae.patch_size)
    x = x[:, :n_prompt_frames]

    # get alphas
    betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")

    # emit initial prompt frames
    for frame_idx in range(n_prompt_frames):
        # decode prompt frame
        prompt_frame = x[:, frame_idx:frame_idx+1]
        prompt_frame_reshaped = rearrange(prompt_frame, "b t c h w -> (b t) (h w) c")
        with torch.no_grad():
            decoded_prompt = (vae.decode(prompt_frame_reshaped.float() / scaling_factor) + 1) / 2
        decoded_prompt = rearrange(decoded_prompt, "(b t) c h w -> b t h w c", t=1)
        yield (decoded_prompt[0, 0], frame_idx, False)

    # Initialize collections for video saving
    all_frames = []
    
    # Add prompt frames to the collection
    for frame_idx in range(n_prompt_frames):
        prompt_frame = x[:, frame_idx:frame_idx+1]
        prompt_frame_reshaped = rearrange(prompt_frame, "b t c h w -> (b t) (h w) c")
        with torch.no_grad():
            decoded_prompt = (vae.decode(prompt_frame_reshaped.float() / scaling_factor) + 1) / 2
        decoded_prompt = rearrange(decoded_prompt, "(b t) c h w -> b t h w c", t=1)
        all_frames.append(decoded_prompt[0, 0])

    # sampling loop - generate one frame at a time
    for i in range(n_prompt_frames, max_frames):
        # Get next action
        should_stop = False
        actions_tensor = None
        
        if actions_generator is not None:
            try:
                action_result = next(actions_generator)
                if isinstance(action_result, tuple) and len(action_result) == 2:
                    actions_tensor, action_dict = action_result
                    
                    # Check for stop action
                    if stop_action_key and action_dict.get(stop_action_key, 0) > 0:
                        should_stop = True
                        print(f"Stop action '{stop_action_key}' detected. Stopping generation.")
                else:
                    # Handle backward compatibility - assume it's just the tensor
                    actions_tensor = action_result
            except StopIteration:
                should_stop = True
                print("Action generator exhausted. Stopping generation.")
        else:
            # Load from file (basic support)
            if not hasattr(generate_video_frames, '_file_actions'):
                generate_video_frames._file_actions = load_actions(actions_path, action_offset=video_offset)
            
            if i < generate_video_frames._file_actions.shape[1]:
                actions_tensor = generate_video_frames._file_actions[:, i:i+1]
            else:
                should_stop = True

        if should_stop:
            # Save video before stopping
            if output_path and len(all_frames) > 0:
                save_generated_video(all_frames, output_path, fps)
                print(f"Generation stopped. Video saved with {len(all_frames)} frames.")
            break

        # Prepare actions for model
        if actions_tensor is None:
            # Create zero tensor if no action available
            actions_tensor = torch.zeros((1, 1, 25)).to(device)
        
        # Ensure actions_tensor is on the right device and has correct shape
        if actions_tensor.dim() == 2:
            actions_tensor = actions_tensor.unsqueeze(1)  # Add time dimension
        actions_tensor = actions_tensor.to(device)

        # Create actions sequence for current frame count
        current_frame_count = i + 1
        actions_for_model = torch.zeros((1, current_frame_count, 25)).to(device)
        # Use the current action for the latest frame
        actions_for_model[:, -1:] = actions_tensor[:, -1:]

        # Generate single frame using the new function
        logger.info(f"Action for frame {i}: {actions_tensor}")
        decoded_frame, x = generate_single_frame(model, vae, x, actions_for_model, ddim_steps)
        
        # Add frame to collection
        all_frames.append(decoded_frame)
        
        # Yield the newly generated frame
        yield (decoded_frame, i, should_stop)

    # Final save if we reached max_frames without stopping
    if output_path and len(all_frames) > 0:
        save_generated_video(all_frames, output_path, fps)
        print(f"Generation completed. Video saved with {len(all_frames)} frames.")


def save_generated_video(frames, output_path, fps=20):
    """Save generated frames to video file"""
    if len(frames) == 0:
        print("No frames to save.")
        return
    
    # Stack frames into video tensor
    frames_tensor = torch.stack(frames, dim=0)  # Shape: (T, H, W, C)
    frames_tensor = frames_tensor.unsqueeze(0)  # Add batch dimension: (1, T, H, W, C)
    
    # Save video
    frames_save = torch.clamp(frames_tensor, 0, 1)
    frames_save = (frames_save * 255).byte()
    write_video(output_path, frames_save[0].cpu(), fps=fps)
    print(f"Video saved to {output_path} with {len(frames)} frames.")


def generate_video(model, vae, prompt_path, actions_generator=None, actions_path=None, num_frames=32, 
                   n_prompt_frames=1, video_offset=None, ddim_steps=10, 
                   output_path=None, fps=20, frame_callback=None):
    """
    Generate video using the diffusion model (compatibility wrapper).
    
    Args:
        model: The DiT model
        vae: The VAE model  
        prompt_path: Path to prompt image/video
        actions_generator: Generator function that yields action tensors for each frame (optional)
        actions_path: Path to actions file (used if actions_generator is None)
        num_frames: Number of frames to generate
        n_prompt_frames: Number of prompt frames
        video_offset: Video offset for reading prompt
        ddim_steps: Number of DDIM steps
        output_path: Path to save video (optional)
        fps: FPS for saved video
        frame_callback: Optional callback for each generated frame
        
    Returns:
        Generated video tensor of shape (1, T, H, W, C)
    """
    # Use the new frame generator for compatibility
    all_frames = []
    
    for frame_tensor, frame_idx, should_stop in generate_video_frames(
        model, vae, prompt_path, actions_generator, actions_path, 
        num_frames, n_prompt_frames, video_offset, ddim_steps, 
        None, None, fps  # Don't auto-save in compatibility mode
    ):
        all_frames.append(frame_tensor)
        
        # Call frame callback if provided
        if frame_callback:
            frame_callback(frame_tensor, frame_idx, num_frames)
        
        # Stop if we've generated enough frames
        if len(all_frames) >= num_frames or should_stop:
            break
    
    # Stack frames into video tensor
    if len(all_frames) == 0:
        return torch.zeros((1, 0, 360, 640, 3))  # Empty video tensor
    
    frames_tensor = torch.stack(all_frames, dim=0)  # Shape: (T, H, W, C)
    frames_tensor = frames_tensor.unsqueeze(0)  # Add batch dimension: (1, T, H, W, C)
    
    # Save video if output path provided
    if output_path:
        save_generated_video(all_frames, output_path, fps)
    
    return torch.clamp(frames_tensor, 0, 1)


def main(args):
    # load DiT checkpoint
    model = DiT_models["DiT-S/2"]()
    print(f"loading Oasis-500M from oasis-ckpt={os.path.abspath(args.oasis_ckpt)}...")
    if args.oasis_ckpt.endswith(".pt"):
        ckpt = torch.load(args.oasis_ckpt, weights_only=True)
        model.load_state_dict(ckpt, strict=False)
    elif args.oasis_ckpt.endswith(".safetensors"):
        load_model(model, args.oasis_ckpt)
    model = model.to(device).eval()

    # load VAE checkpoint
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    print(f"loading ViT-VAE-L/20 from vae-ckpt={os.path.abspath(args.vae_ckpt)}...")
    if args.vae_ckpt.endswith(".pt"):
        vae_ckpt = torch.load(args.vae_ckpt, weights_only=True)
        vae.load_state_dict(vae_ckpt)
    elif args.vae_ckpt.endswith(".safetensors"):
        load_model(vae, args.vae_ckpt)
    vae = vae.to(device).eval()

    # Generate video using the reusable function
    generate_video(
        model=model,
        vae=vae,
        prompt_path=args.prompt_path,
        actions_path=args.actions_path,
        num_frames=args.num_frames,
        n_prompt_frames=args.n_prompt_frames,
        video_offset=args.video_offset,
        ddim_steps=args.ddim_steps,
        output_path=args.output_path,
        fps=args.fps
    )


if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument(
        "--oasis-ckpt",
        type=str,
        help="Path to Oasis DiT checkpoint.",
        default="oasis500m.safetensors",
    )
    parse.add_argument(
        "--vae-ckpt",
        type=str,
        help="Path to Oasis ViT-VAE checkpoint.",
        default="vit-l-20.safetensors",
    )
    parse.add_argument(
        "--num-frames",
        type=int,
        help="How many frames should the output be?",
        default=32,
    )
    parse.add_argument(
        "--prompt-path",
        type=str,
        help="Path to image or video to condition generation on.",
        default="sample_data/sample_image_0.png",
    )
    parse.add_argument(
        "--actions-path",
        type=str,
        help="File to load actions from (.actions.pt or .one_hot_actions.pt)",
        default="sample_data/sample_actions_0.one_hot_actions.pt",
    )
    parse.add_argument(
        "--video-offset",
        type=int,
        help="If loading prompt from video, index of frame to start reading from.",
        default=None,
    )
    parse.add_argument(
        "--n-prompt-frames",
        type=int,
        help="If the prompt is a video, how many frames to condition on.",
        default=1,
    )
    parse.add_argument(
        "--output-path",
        type=str,
        help="Path where generated video should be saved.",
        default="video.mp4",
    )
    parse.add_argument(
        "--fps",
        type=int,
        help="What framerate should be used to save the output?",
        default=20,
    )
    parse.add_argument("--ddim-steps", type=int, help="How many DDIM steps?", default=10)

    args = parse.parse_args()
    print("inference args:")
    pprint(vars(args))
    main(args)
