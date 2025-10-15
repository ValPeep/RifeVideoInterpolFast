import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from comfy.model_management import get_torch_device

# Import your model definition (RIFE, etc.)
from .train_log.RIFE_HDv3 import Model
from .model.pytorch_msssim import ssim_matlab
from pathlib import Path
import os
class VideoInterpolationNode:
    @classmethod
    def INPUT_TYPES(cls):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_model_path = os.path.join(current_dir, "train_log")
        return {
            "required": {
                "images": ("IMAGE",),
                "skip_frames": ("INT", {"default": 0, "min": 0, "max": 100}),
                "model": ("STRING", {"default": default_model_path}),
                "fp16": ("BOOLEAN", {"default": True}),
                "multi": ("INT", {"default": 2, "min": 1, "max": 8}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("interpolated_frames",)
    FUNCTION = "interpolate"
    CATEGORY = "Custom/Video"
    
    def _pad_image(self, img, padding, fp16):
        if fp16:
            return F.pad(img, padding).half()
        return F.pad(img, padding)
    
    def _make_inference(self, model, I0, I1, n, scale=1.0):
        if model.version >= 3.9:
            res = []
            for i in range(n):
                res.append(model.inference(I0, I1, (i + 1) / (n + 1), scale))
            return res
        else:
            middle = model.inference(I0, I1, scale)
            if n == 1:
                return [middle]
            first_half = self._make_inference(model, I0, middle, n // 2, scale)
            second_half = self._make_inference(model, middle, I1, n // 2, scale)
            return [*first_half, middle, *second_half] if n % 2 else [*first_half, *second_half]
    
    def interpolate(self, images, skip_frames, model, fp16, multi):
        device = get_torch_device()
        # Save ALL original states
        original_grad_enabled = torch.is_grad_enabled()
        original_cudnn_enabled = torch.backends.cudnn.enabled
        original_cudnn_benchmark = torch.backends.cudnn.benchmark
        original_default_dtype = torch.get_default_dtype()
        try:
            torch.set_grad_enabled(False)
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if fp16:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)
        
            model_obj = Model()
            if not hasattr(model_obj, "version"):
                model_obj.version = 0
            model_obj.load_model(model, -1)
            model_obj.eval()
            model_obj.device()
            
            print(f"🚀 Loaded RIFE model from {model}")
            print(f"Processing {len(images)} frames, skipping {skip_frames}")
            
            # ComfyUI images are in B H W C format, convert to process
            imgs = images[skip_frames:]
            
            # Get dimensions from first frame
            h, w = imgs[0].shape[0:2]
            
            # Calculate padding
            tmp = max(128, 128)  # scale is always 1.0
            ph = ((h - 1) // tmp + 1) * tmp
            pw = ((w - 1) // tmp + 1) * tmp
            padding = (0, pw - w, 0, ph - h)
            
            all_outputs = []
            pbar = tqdm(total=len(imgs) - 1)
            
            # Convert first frame to tensor (ComfyUI format is B H W C with values 0-1)
            lastframe = imgs[0].cpu().numpy()  # H W C format, 0-1 range
            
            # Convert to CHW format and prepare for model
            I1 = torch.from_numpy(np.transpose(lastframe, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float()
            I1 = self._pad_image(I1, padding, fp16)
            
            # DO NOT add first frame here - it will be added in the loop
            
            temp = None
            scale = 1.0
            
            for frame_idx in range(1, len(imgs)):
                if temp is not None:
                    frame = temp
                    temp = None
                else:
                    frame = imgs[frame_idx].cpu().numpy()  # H W C format, 0-1 range
                
                I0 = I1
                I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float()
                I1 = self._pad_image(I1, padding, fp16)
                
                # Check for static frames using SSIM
                I0_small = F.interpolate(I0, (32, 32), mode="bilinear", align_corners=False)
                I1_small = F.interpolate(I1, (32, 32), mode="bilinear", align_corners=False)
                ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
                
                break_flag = False
                
                # Handle very similar frames (static content)
                if ssim > 0.996:
                    frame_idx += 1  # Move to next frame
                    
                    # Try to get next frame
                    if frame_idx < len(imgs):
                        frame = imgs[frame_idx].cpu().numpy()  # Get next frame
                        temp = frame  # Save THIS (next) frame to temp
                    else:
                        break_flag = True
                        frame = lastframe  # Use lastframe like the original
                    
                    I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float()
                    I1 = self._pad_image(I1, padding, fp16)
                    I1 = model_obj.inference(I0, I1, scale=scale)
                    I1_small = F.interpolate(I1, (32, 32), mode="bilinear", align_corners=False)
                    ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
                    
                    # Update frame numpy array from the interpolated result
                    frame = (I1[0] * 1.0).cpu().numpy().transpose(1, 2, 0)[:h, :w]
                
                # Handle scene changes (low SSIM)
                if ssim < 0.2:
                    # For scene changes, just duplicate the first frame
                    output = []
                    for i in range(multi - 1):
                        output.append(I0)
                else:
                    # Normal interpolation
                    output = self._make_inference(model_obj, I0, I1, multi - 1, scale)
                
                # IMPORTANT: Add lastframe FIRST (this is the previous frame)
                lastframe_tensor = torch.from_numpy(lastframe)
                all_outputs.append(lastframe_tensor)
                
                # Add interpolated frames
                for mid in output:
                    mid_tensor = mid[0][:, :h, :w].permute(1, 2, 0).cpu()  # Convert to H W C
                    all_outputs.append(mid_tensor)
                
                # Update lastframe to current frame for next iteration
                lastframe = frame
                pbar.update(1)
                
                if break_flag:
                    break
            
            # Add the final lastframe (equivalent to write_buffer.put(lastframe) at the end)
            final_frame_tensor = torch.from_numpy(lastframe)
            all_outputs.append(final_frame_tensor)
            
            pbar.close()
            
            # Stack all outputs into a single tensor (B H W C format)
            output_tensor = torch.stack(all_outputs, dim=0)
            
            print(f"✅ Interpolation complete - produced {len(all_outputs)} frames")
            
            return (output_tensor,)
        finally:
            if model_obj is not None:
                del model_obj
            # Reset ALL global states to original values
            torch.set_grad_enabled(original_grad_enabled)
            torch.backends.cudnn.enabled = original_cudnn_enabled
            torch.backends.cudnn.benchmark = original_cudnn_benchmark
            
            if fp16:
                # Reset default tensor type
                torch.set_default_tensor_type(torch.FloatTensor)
            
            # Also ensure we're back to float32 default dtype
            torch.set_default_dtype(torch.float32)
            
            # Clear any cached tensors that might be on wrong device
            # if torch.cuda.is_available():
                # torch.cuda.empty_cache()

NODE_CLASS_MAPPINGS = {
    "VideoInterpolationNode": VideoInterpolationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoInterpolationNode": "🎞️ Video Interpolation (RIFE)"
}