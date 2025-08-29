"""
Video Streaming Server for Real-time Video Generation
Provides HTTP streaming endpoint for generated videos
"""

import os
import io
import cv2
import torch
import numpy as np
from threading import Thread, Lock
import queue
import time
from flask import Flask, Response, render_template_string, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import base64
import json
from collections import deque
import datetime
import logging

from dit import DiT_models
from vae import VAE_models
from utils import load_prompt, load_actions, sigmoid_beta_schedule
from torchvision.io import read_video, write_video
from tqdm import tqdm
from einops import rearrange
from torch import autocast
from safetensors.torch import load_model
from generate import generate_video, generate_video_frames, SingleFrameGenerator

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for video generation
device = "cuda:0" if torch.cuda.is_available() else "cpu"
generation_lock = Lock()
frame_queue = queue.Queue(maxsize=100)
is_generating = False
generation_thread = None

# Action streaming variables
action_buffer = deque(maxlen=10)  # Buffer to store real-time actions
action_lock = Lock()
target_fps = 10  # Target FPS for action collection
last_action_time = None
current_client_id = None

# Action template (matching ACTION_KEYS from utils.py)
DEFAULT_ACTION = {
    "inventory": 0, "ESC": 0, "hotbar.1": 0, "hotbar.2": 0, "hotbar.3": 0, 
    "hotbar.4": 0, "hotbar.5": 0, "hotbar.6": 0, "hotbar.7": 0, "hotbar.8": 0, 
    "hotbar.9": 0, "forward": 0, "back": 0, "left": 0, "right": 0, 
    "cameraX": 0, "cameraY": 0, "jump": 0, "sneak": 0, "sprint": 0, 
    "swapHands": 0, "attack": 0, "use": 0, "pickItem": 0, "drop": 0
}

# Real-time action processing functions
def add_action_to_buffer(action_data):
    """Add action to the buffer with timestamp"""
    global action_buffer, action_lock
    
    with action_lock:
        timestamp = datetime.datetime.now()
        # Merge with default action template
        complete_action = DEFAULT_ACTION.copy()
        complete_action.update(action_data)
        
        action_buffer.append({
            'action': complete_action,
            'timestamp': timestamp
        })

def create_realtime_action_generator_with_stop(max_frames=1000):
    """Create a real-time action generator that yields (action_tensor, action_dict) tuples"""
    global action_buffer, action_lock, is_generating
    
    from utils import one_hot_actions
    import time
    
    def realtime_action_generator():
        """Generator that yields (action_tensor, action_dict) tuples in real-time"""
        frame_duration = 1.0 / target_fps  # Time per frame in seconds
        start_time = time.time()
        
        for frame_idx in range(max_frames):
            if not is_generating:  # Check if generation was stopped
                break
                
            # Calculate expected time for this frame
            expected_frame_time = start_time + (frame_idx * frame_duration)
            current_time = time.time()
            
            # Wait if we're ahead of schedule
            if current_time < expected_frame_time:
                time.sleep(expected_frame_time - current_time)
            
            # Get the most recent action from buffer
            with action_lock:
                if len(action_buffer) > 0:
                    # Use the most recent action
                    latest_action = action_buffer[-1]['action'].copy()
                else:
                    # Use default action if no actions available
                    latest_action = DEFAULT_ACTION.copy()
            
            # Convert to tensor format
            action_tensor = one_hot_actions([latest_action])
            
            # Yield both tensor and dict for stop action detection
            yield (action_tensor[0:1], latest_action)
    
    return realtime_action_generator()

def create_realtime_action_generator(num_frames):
    """Create a real-time action generator that yields actions as they become available (backward compatibility)"""
    global action_buffer, action_lock, is_generating
    
    from utils import one_hot_actions
    import time
    
    def realtime_action_generator():
        """Generator that yields actions in real-time based on buffer state"""
        frame_duration = 1.0 / target_fps  # Time per frame in seconds
        start_time = time.time()
        
        for frame_idx in range(num_frames):
            if not is_generating:  # Check if generation was stopped
                break
                
            # Calculate expected time for this frame
            expected_frame_time = start_time + (frame_idx * frame_duration)
            current_time = time.time()
            
            # Wait if we're ahead of schedule
            if current_time < expected_frame_time:
                time.sleep(expected_frame_time - current_time)
            
            # Get the most recent action from buffer
            with action_lock:
                if len(action_buffer) > 0:
                    # Use the most recent action
                    latest_action = action_buffer[-1]['action'].copy()
                else:
                    # Use default action if no actions available
                    latest_action = DEFAULT_ACTION.copy()
            
            # Convert to tensor format
            action_tensor = one_hot_actions([latest_action])
            yield action_tensor[0:1]  # Yield single frame with batch dimension
    
    return realtime_action_generator()

def create_action_generator(num_frames):
    """Create a generator that yields action tensors for each frame (legacy compatibility)"""
    global action_buffer, action_lock
    
    from utils import one_hot_actions
    
    # Pre-calculate all actions based on current buffer state
    with action_lock:
        if len(action_buffer) == 0:
            # Return default actions if no actions available
            actions = [DEFAULT_ACTION] * num_frames
            actions_tensor = one_hot_actions(actions)
        else:
            # Get recent actions based on fps timing
            actions = []
            current_time = datetime.datetime.now()
            time_per_frame = 1.0 / target_fps
            
            for i in range(num_frames):
                # Calculate the time for this frame
                frame_time = current_time - datetime.timedelta(seconds=(num_frames - i - 1) * time_per_frame)
                
                # Find the closest action in buffer
                closest_action = DEFAULT_ACTION.copy()
                min_time_diff = float('inf')
                
                for buffered in action_buffer:
                    time_diff = abs((buffered['timestamp'] - frame_time).total_seconds())
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_action = buffered['action'].copy()
                
                actions.append(closest_action)
            
            # Convert to one-hot format
            actions_tensor = one_hot_actions(actions)
    
    # Create generator that yields individual frame actions
    def action_generator():
        for frame_idx in range(num_frames):
            yield actions_tensor[frame_idx:frame_idx+1]  # Yield single frame with batch dimension
    
    return action_generator()

def get_actions_for_generation(num_frames):
    """Convert buffered actions to format expected by generate_video (backward compatibility)"""
    generator = create_action_generator(num_frames)
    actions = []
    for action in generator:
        actions.append(action)
    return torch.cat(actions, dim=0).unsqueeze(0)

def clear_action_buffer():
    """Clear the action buffer"""
    global action_buffer, action_lock
    with action_lock:
        action_buffer.clear()

class VideoStreamGenerator:
    def __init__(self, model_path="oasis500m.safetensors", vae_path="vit-l-20.safetensors"):
        self.model = None
        self.vae = None
        self.model_path = model_path
        self.vae_path = vae_path
        self.load_models()
        
    def load_models(self):
        """Load DiT and VAE models"""
        print("Loading models...")
        
        # Load DiT checkpoint
        self.model = DiT_models["DiT-S/2"]()
        print(f"Loading Oasis-500M from {os.path.abspath(self.model_path)}...")
        
        if os.path.exists(self.model_path):
            if self.model_path.endswith(".pt"):
                ckpt = torch.load(self.model_path, weights_only=True)
                self.model.load_state_dict(ckpt, strict=False)
            elif self.model_path.endswith(".safetensors"):
                load_model(self.model, self.model_path)
        else:
            print(f"Warning: Model file {self.model_path} not found. Using random weights.")
            
        self.model = self.model.to(device).eval()

        # Load VAE checkpoint
        self.vae = VAE_models["vit-l-20-shallow-encoder"]()
        print(f"Loading ViT-VAE-L/20 from {os.path.abspath(self.vae_path)}...")
        
        if os.path.exists(self.vae_path):
            if self.vae_path.endswith(".pt"):
                vae_ckpt = torch.load(self.vae_path, weights_only=True)
                self.vae.load_state_dict(vae_ckpt)
            elif self.vae_path.endswith(".safetensors"):
                load_model(self.vae, self.vae_path)
        else:
            print(f"Warning: VAE file {self.vae_path} not found. Using random weights.")
            
        self.vae = self.vae.to(device).eval()
        print("Models loaded successfully!")

    def generate_video_stream(self, prompt_path, actions_path=None, num_frames=32, 
                            n_prompt_frames=1, video_offset=None, ddim_steps=10, use_realtime_actions=False):
        """Generate video frames and stream them using the shared generation function"""
        global frame_queue, is_generating
        
        def frame_callback(frame_tensor, frame_idx, total_frames):
            """Callback function to handle each generated frame"""
            global is_generating
            
            if not is_generating:  # Check if generation was stopped
                return
            
            # Convert tensor to numpy
            frame_data = torch.clamp(frame_tensor, 0, 1).cpu().numpy()
            frame_data = (frame_data * 255).astype(np.uint8)

            # Convert to JPEG for streaming
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR))
            frame_bytes = buffer.tobytes()
            
            # Add to queue
            if not frame_queue.full():
                frame_queue.put({
                    'frame': frame_bytes,
                    'frame_idx': frame_idx,
                    'total_frames': total_frames,
                    'status': 'generating'
                })
            
            time.sleep(0.1)  # Small delay to prevent overwhelming the client
        
        try:
            is_generating = True
            
            # Handle real-time actions or file-based actions
            if use_realtime_actions:
                # Create real-time action generator that yields actions as they arrive
                actions_generator = create_realtime_action_generator(num_frames)
                
                # Use the shared generation function with callback
                generate_video(
                    model=self.model,
                    vae=self.vae,
                    prompt_path=prompt_path,
                    actions_generator=actions_generator,
                    num_frames=num_frames,
                    n_prompt_frames=n_prompt_frames,
                    video_offset=video_offset,
                    ddim_steps=ddim_steps,
                    output_path="output.mp4",  # Save generated video
                    fps=20,
                    frame_callback=frame_callback
                )
            else:
                # Use traditional file-based actions
                generate_video(
                    model=self.model,
                    vae=self.vae,
                    prompt_path=prompt_path,
                    actions_path=actions_path,
                    num_frames=num_frames,
                    n_prompt_frames=n_prompt_frames,
                    video_offset=video_offset,
                    ddim_steps=ddim_steps,
                    output_path="output.mp4",  # Save generated video
                    fps=20,
                    frame_callback=frame_callback
                )

            # Signal completion
            if not frame_queue.full():
                frame_queue.put({
                    'frame': None,
                    'frame_idx': num_frames,
                    'total_frames': num_frames,
                    'status': 'completed'
                })
                
        except Exception as e:
            print(f"Error in video generation: {e}")
            if not frame_queue.full():
                frame_queue.put({
                    'frame': None,
                    'frame_idx': -1,
                    'total_frames': num_frames,
                    'status': 'error',
                    'error': str(e)
                })
        finally:
            is_generating = False

# Global generator instance
generator = VideoStreamGenerator()

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    global current_client_id
    print(f'Client connected: {request.sid}')
    current_client_id = request.sid
    emit('connected', {'message': 'Connected to action stream'})

@socketio.on('disconnect')
def handle_disconnect():
    global current_client_id
    print(f'Client disconnected: {request.sid}')
    if current_client_id == request.sid:
        current_client_id = None

@socketio.on('action')
def handle_action(data):
    """Handle incoming action data from client"""
    global current_client_id, last_action_time
 
    if current_client_id != request.sid:
        current_client_id = request.sid
    
    try:
        # Validate and add action to buffer
        action_data = data.get('action', {})
        
        # Throttle actions to target FPS
        current_time = datetime.datetime.now()
        if last_action_time is not None:
            time_diff = (current_time - last_action_time).total_seconds()
            min_time_between_actions = 1.0 / target_fps
            if time_diff < min_time_between_actions:
                return  # Skip this action to maintain FPS limit
        
        last_action_time = current_time
        add_action_to_buffer(action_data)
        
        # Send confirmation back to client
        emit('action_received', {
            'timestamp': current_time.isoformat(),
            'buffer_size': len(action_buffer),
            'ready_for_generation': len(action_buffer) >= 5  # Signal when ready
        })
        
    except Exception as e:
        emit('error', {'message': f'Error processing action: {str(e)}'})

@socketio.on('set_fps')
def handle_set_fps(data):
    """Allow client to set target FPS"""
    global target_fps
    try:
        new_fps = int(data.get('fps', 20))
        if 1 <= new_fps <= 60:
            target_fps = new_fps
            emit('fps_updated', {'fps': target_fps})
        else:
            emit('error', {'message': 'FPS must be between 1 and 60'})
    except Exception as e:
        emit('error', {'message': f'Error setting FPS: {str(e)}'})

@socketio.on('clear_actions')
def handle_clear_actions():
    """Clear the action buffer"""
    clear_action_buffer()
    emit('actions_cleared', {'message': 'Action buffer cleared'})

@socketio.on('start_realtime_generation')
def handle_start_realtime_generation(data):
    """Start real-time video generation with current actions"""
    global is_generating, generation_thread
    
    if is_generating:
        emit('error', {'message': 'Generation already in progress'})
        return
    
    try:
        # Get parameters
        prompt_path = data.get('prompt_path', 'sample_data/sample_image_0.png')
        num_frames = int(data.get('num_frames', 1))  # Shorter for real-time
        ddim_steps = int(data.get('ddim_steps', 8))   # Faster generation
        logger.info(f"Starting real-time generation with {num_frames} frames and {ddim_steps} steps")
        
        # Check if we have enough actions
        if len(action_buffer) < 3:
            emit('error', {'message': 'Need at least 3 actions to start generation'})
            return
        
        # Capture client ID before starting thread
        client_id = request.sid
        
        # Start generation in background thread
        def realtime_generation():
            global is_generating
            try:
                is_generating = True
                
                # Create real-time action generator
                actions_generator = create_realtime_action_generator(num_frames)
                
                def frame_callback(frame_tensor, frame_idx, total_frames):
                    """Send frames to client via WebSocket"""
                    if not is_generating:
                        return
                    
                    # Convert tensor to image bytes
                    import cv2
                    frame_data = torch.clamp(frame_tensor, 0, 1).cpu().numpy()
                    frame_data = (frame_data * 255).astype(np.uint8)
                    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR))
                    frame_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                    
                    # Emit frame to client using captured client_id
                    socketio.emit('realtime_frame', {
                        'frame': frame_b64,
                        'frame_idx': frame_idx,
                        'total_frames': total_frames
                    }, room=client_id)
                
                # Generate video with real-time actions
                generate_video(
                    model=generator.model,
                    vae=generator.vae,
                    prompt_path=prompt_path,
                    actions_generator=actions_generator,
                    num_frames=num_frames,
                    ddim_steps=ddim_steps,
                    frame_callback=frame_callback
                )
                
                # Signal completion
                socketio.emit('realtime_generation_complete', {
                    'message': 'Real-time generation completed'
                }, room=client_id)
                
            except Exception as e:
                socketio.emit('error', {
                    'message': f'Real-time generation error: {str(e)}'
                }, room=client_id)
            finally:
                is_generating = False
        
        generation_thread = Thread(target=realtime_generation)
        generation_thread.start()
        
        emit('realtime_generation_started', {
            'message': 'Real-time generation started',
            'num_frames': num_frames,
            'ddim_steps': ddim_steps
        })
        
    except Exception as e:
        emit('error', {'message': f'Error starting real-time generation: {str(e)}'})

@socketio.on('start_continuous_generation')
def handle_start_continuous_generation(data):
    """Start continuous video generation that stops on specific action (e.g., ESC)"""
    global is_generating, generation_thread
    
    if is_generating:
        emit('error', {'message': 'Generation already in progress'})
        return
    
    try:
        # Get parameters
        prompt_path = data.get('prompt_path', 'sample_data/sample_image_0.png')
        max_frames = int(data.get('max_frames', 100))  # Maximum frames
        ddim_steps = int(data.get('ddim_steps', 8))    # Fast generation
        stop_action = data.get('stop_action', 'ESC')   # Default stop on ESC
        import time
        output_path = data.get('output_path', f'realtime_video_{int(time.time())}.mp4')
        
        # Check if we have enough actions
        if len(action_buffer) < 3:
            emit('error', {'message': 'Need at least 3 actions to start generation'})
            return
        
        # Capture client ID before starting thread
        client_id = request.sid
        
        # Start generation in background thread
        def continuous_generation():
            global is_generating
            try:
                is_generating = True
                
                # Create real-time action generator with stop detection
                actions_generator = create_realtime_action_generator_with_stop(max_frames)
                
                def frame_callback(frame_tensor, frame_idx, should_stop):
                    """Send frames to client via WebSocket"""
                    if not is_generating:
                        return
                    
                    # Convert tensor to image bytes
                    import cv2
                    frame_data = torch.clamp(frame_tensor, 0, 1).cpu().numpy()
                    frame_data = (frame_data * 255).astype(np.uint8)
                    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR))
                    frame_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                    
                    # Emit frame to client using captured client_id
                    socketio.emit('continuous_frame', {
                        'frame': frame_b64,
                        'frame_idx': frame_idx,
                        'should_stop': should_stop
                    }, room=client_id)
                
                # Generate video with frame generator
                frame_count = 0
                for frame_tensor, frame_idx, should_stop in generate_video_frames(
                    model=generator.model,
                    vae=generator.vae,
                    prompt_path=prompt_path,
                    actions_generator=actions_generator,
                    max_frames=max_frames,
                    ddim_steps=ddim_steps,
                    stop_action_key=stop_action,
                    output_path=output_path,
                    fps=target_fps
                ):
                    if not is_generating:
                        break
                    
                    frame_callback(frame_tensor, frame_idx, should_stop)
                    frame_count += 1
                    
                    if should_stop:
                        break
                
                # Signal completion
                socketio.emit('continuous_generation_complete', {
                    'message': f'Continuous generation completed with {frame_count} frames',
                    'output_path': output_path,
                    'frame_count': frame_count
                }, room=client_id)
                
            except Exception as e:
                socketio.emit('error', {
                    'message': f'Continuous generation error: {str(e)}'
                }, room=client_id)
            finally:
                is_generating = False
        
        generation_thread = Thread(target=continuous_generation)
        generation_thread.start()
        
        emit('continuous_generation_started', {
            'message': 'Continuous generation started',
            'max_frames': max_frames,
            'ddim_steps': ddim_steps,
            'stop_action': stop_action,
            'output_path': output_path
        })
        
    except Exception as e:
        emit('error', {'message': f'Error starting continuous generation: {str(e)}'})

# 全局单帧生成器实例
single_frame_generator = None

# Continuous single frame generation variables
is_generating_continuous_frames = False
continuous_frame_thread = None

@socketio.on('init_single_frame_generator')
def handle_init_single_frame_generator(data):
    """初始化单帧生成器"""
    global single_frame_generator
    
    try:
        logger.info(f"Initializing single frame generator with data: {data}")
        prompt_path = data.get('prompt_path', 'sample_data/sample_image_0.png')
        ddim_steps = int(data.get('ddim_steps', 20))
        
        # 创建单帧生成器
        single_frame_generator = SingleFrameGenerator(
            model=generator.model,
            vae=generator.vae,
            prompt_path=prompt_path,
            n_prompt_frames=1,
            ddim_steps=ddim_steps
        )
        
        # 获取并发送prompt帧
        prompt_frame = single_frame_generator.get_prompt_frame()
        
        # 转换为base64
        import cv2
        frame_data = torch.clamp(prompt_frame, 0, 1).cpu().numpy()
        frame_data = (frame_data * 255).astype(np.uint8)
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR))
        frame_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        emit('single_frame_generator_ready', {
            'message': 'Single frame generator initialized',
            'prompt_frame': frame_b64,
            'frame_count': single_frame_generator.frame_count
        })
        
    except Exception as e:
        emit('error', {'message': f'Error initializing single frame generator: {str(e)}'})

@socketio.on('generate_single_frame')
def handle_generate_single_frame(data):
    """生成单帧图像"""
    global single_frame_generator
    
    if single_frame_generator is None:
        emit('error', {'message': 'Single frame generator not initialized. Call init_single_frame_generator first.'})
        return
    
    try:
        # 获取最新action
        with action_lock:
            if len(action_buffer) > 0:
                latest_action = action_buffer[-1]['action'].copy()
            else:
                latest_action = DEFAULT_ACTION.copy()
        
        logger.info(f"Latest action: {latest_action}")
        # 转换为tensor
        from utils import one_hot_actions
        action_tensor = one_hot_actions([latest_action])
        logger.info(f"Action tensor: {action_tensor}")
        
        # 生成单帧
        start_time = time.time()
        frame = single_frame_generator.generate_next_frame(action_tensor)
        generation_time = time.time() - start_time
        
        # 转换为base64
        import cv2
        frame_data = torch.clamp(frame, 0, 1).cpu().numpy()
        frame_data = (frame_data * 255).astype(np.uint8)
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR))
        frame_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        emit('single_frame_generated', {
            'frame': frame_b64,
            'frame_count': single_frame_generator.frame_count,
            'generation_time': round(generation_time, 3),
            'action_used': latest_action
        })
        
        logger.info(f"Generated single frame, frame_count: {single_frame_generator.frame_count}")
        
    except Exception as e:
        emit('error', {'message': f'Error generating single frame: {str(e)}'})

@socketio.on('reset_single_frame_generator')
def handle_reset_single_frame_generator():
    """重置单帧生成器"""
    global single_frame_generator
    single_frame_generator = None
    emit('single_frame_generator_reset', {'message': 'Single frame generator reset'})

def continuous_frame_generation_worker(client_id):
    """Background worker for continuous single frame generation"""
    global is_generating_continuous_frames, single_frame_generator
    
    frame_count = 0
    while is_generating_continuous_frames:
        try:
            if single_frame_generator is None:
                socketio.emit('error', {'message': 'Single frame generator not initialized'}, room=client_id)
                break
            
            # 获取最新action
            with action_lock:
                if len(action_buffer) > 0:
                    latest_action = action_buffer[-1]['action'].copy()
                else:
                    latest_action = DEFAULT_ACTION.copy()
            
            # 转换为tensor
            logger.info(f"Latest action: {latest_action}")
            from utils import one_hot_actions
            action_tensor = one_hot_actions([latest_action])
            
            # 生成单帧
            start_time = time.time()
            try:
                frame = single_frame_generator.generate_next_frame(action_tensor)
            except Exception as e:
                logger.error(f"Error generating next frame: {e}")
                frame = None
                generation_time = 0
                socketio.emit('error', {'message': f'Error generating next frame: {str(e)}'}, room=client_id)
                break
            generation_time = time.time() - start_time
            
            # 转换为base64
            import cv2
            frame_data = torch.clamp(frame, 0, 1).cpu().numpy()
            frame_data = (frame_data * 255).astype(np.uint8)
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR))
            frame_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
            
            frame_count += 1
            
            # 发送帧到客户端
            socketio.emit('continuous_frame_generated', {
                'frame': frame_b64,
                'frame_count': single_frame_generator.frame_count,
                'generation_time': round(generation_time, 3),
                'action_used': latest_action,
                'continuous_frame_index': frame_count
            }, room=client_id)
            
            logger.info(f"Generated continuous frame {frame_count}, total frame_count: {single_frame_generator.frame_count}")
            
            # 控制生成频率，避免过快
            time.sleep(0.1)  # 10 FPS
            
        except Exception as e:
            logger.error(f"Error in continuous frame generation: {str(e)}")
            socketio.emit('error', {'message': f'Error in continuous frame generation: {str(e)}'}, room=client_id)
            break
    
    # 生成结束时发送通知
    socketio.emit('continuous_frame_generation_stopped', {
        'message': 'Continuous frame generation stopped',
        'total_frames_generated': frame_count
    }, room=client_id)

@socketio.on('start_continuous_single_frame_generation')
def handle_start_continuous_single_frame_generation(data):
    """开始连续单帧生成"""
    global is_generating_continuous_frames, continuous_frame_thread, single_frame_generator
    
    if single_frame_generator is None:
        emit('error', {'message': 'Single frame generator not initialized. Call init_single_frame_generator first.'})
        return
    
    if is_generating_continuous_frames:
        emit('error', {'message': 'Continuous frame generation is already running'})
        return
    
    try:
        # 获取客户端ID
        client_id = request.sid
        
        # 开始连续生成
        is_generating_continuous_frames = True
        continuous_frame_thread = Thread(
            target=continuous_frame_generation_worker,
            args=(client_id,),
            daemon=True
        )
        continuous_frame_thread.start()
        
        emit('continuous_frame_generation_started', {
            'message': 'Continuous single frame generation started',
            'frame_rate': '10 FPS'
        })
        
        logger.info("Started continuous single frame generation")
        
    except Exception as e:
        is_generating_continuous_frames = False
        emit('error', {'message': f'Error starting continuous frame generation: {str(e)}'})

@socketio.on('stop_continuous_single_frame_generation')
def handle_stop_continuous_single_frame_generation():
    """停止连续单帧生成"""
    global is_generating_continuous_frames, continuous_frame_thread
    
    if not is_generating_continuous_frames:
        emit('error', {'message': 'No continuous frame generation is running'})
        return
    
    try:
        # 停止生成
        is_generating_continuous_frames = False
        
        # 等待线程结束
        if continuous_frame_thread and continuous_frame_thread.is_alive():
            continuous_frame_thread.join(timeout=2.0)
        
        continuous_frame_thread = None
        
        emit('continuous_frame_generation_stop_confirmed', {
            'message': 'Continuous single frame generation stopped'
        })
        
        logger.info("Stopped continuous single frame generation")
        
    except Exception as e:
        emit('error', {'message': f'Error stopping continuous frame generation: {str(e)}'})

@app.route('/')
def index():
    """Serve the main page with video player"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-time Video Generation with Live Action Control</title>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container { 
                max-width: 1400px; 
                margin: 0 auto; 
                background-color: rgba(255, 255, 255, 0.95); 
                padding: 30px; 
                border-radius: 20px; 
                box-shadow: 0 20px 40px rgba(0,0,0,0.1); 
                backdrop-filter: blur(10px);
            }
            h1 { 
                text-align: center; 
                color: #333; 
                margin-bottom: 30px;
                font-size: 2.5em;
                background: linear-gradient(45deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .main-content {
                display: flex;
                gap: 30px;
                flex-wrap: wrap;
            }
            .video-section {
                flex: 2;
                min-width: 600px;
            }
            .control-section {
                flex: 1;
                min-width: 400px;
            }
            .video-container { 
                text-align: center; 
                margin-bottom: 20px;
                background: #000;
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }
            #videoFrame { 
                max-width: 100%; 
                height: auto; 
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            }
            .controls-panel { 
                background: rgba(248, 249, 250, 0.9);
                padding: 25px; 
                border-radius: 15px; 
                margin-bottom: 20px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            .control-group { 
                margin: 15px 0; 
                display: flex;
                align-items: center;
                gap: 10px;
            }
            label { 
                font-weight: 600;
                color: #444;
                min-width: 120px;
            }
            input, select { 
                flex: 1;
                padding: 8px 12px;
                border: 2px solid #e1e5e9;
                border-radius: 8px;
                font-size: 14px;
                transition: all 0.3s ease;
            }
            input:focus, select:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            .button-group {
                display: flex;
                gap: 10px;
                margin: 20px 0;
            }
            button { 
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white; 
                padding: 12px 24px; 
                border: none; 
                border-radius: 10px; 
                cursor: pointer; 
                font-weight: 600;
                font-size: 14px;
                transition: all 0.3s ease;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            button:hover:not(:disabled) { 
                transform: translateY(-2px);
                box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
            }
            button:disabled { 
                background: #6c757d;
                cursor: not-allowed; 
                transform: none;
                box-shadow: none;
            }
            .realtime-controls {
                background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
                padding: 25px;
                border-radius: 15px;
                margin-bottom: 20px;
            }
            .realtime-title {
                color: #2d3748;
                font-size: 1.4em;
                font-weight: 700;
                margin-bottom: 15px;
                text-align: center;
            }
            .action-display {
                background: rgba(255, 255, 255, 0.8);
                padding: 15px;
                border-radius: 10px;
                margin: 15px 0;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                max-height: 200px;
                overflow-y: auto;
            }
            .input-mode-toggle {
                display: flex;
                gap: 10px;
                margin: 15px 0;
            }
            .mode-btn {
                flex: 1;
                padding: 10px;
                border: 2px solid #e1e5e9;
                border-radius: 8px;
                background: white;
                cursor: pointer;
                transition: all 0.3s ease;
                text-align: center;
                font-weight: 600;
            }
            .mode-btn.active {
                background: #667eea;
                color: white;
                border-color: #667eea;
            }
            .status { 
                margin: 15px 0; 
                padding: 15px; 
                border-radius: 10px; 
                font-weight: 600;
                text-align: center;
            }
            .status.generating { 
                background: linear-gradient(45deg, #d4edda, #c3e6cb);
                color: #155724; 
            }
            .status.error { 
                background: linear-gradient(45deg, #f8d7da, #f5c6cb);
                color: #721c24; 
            }
            .status.completed { 
                background: linear-gradient(45deg, #d1ecf1, #bee5eb);
                color: #0c5460; 
            }
            .status.connected {
                background: linear-gradient(45deg, #cce5ff, #b3d9ff);
                color: #004085;
            }
            .progress { 
                width: 100%; 
                height: 25px; 
                background-color: #e9ecef; 
                border-radius: 15px; 
                overflow: hidden; 
                margin: 15px 0; 
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
            }
            .progress-bar { 
                height: 100%; 
                background: linear-gradient(45deg, #667eea, #764ba2);
                transition: width 0.5s ease; 
                border-radius: 15px;
                position: relative;
            }
            .progress-text {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                color: white;
                font-weight: 600;
                font-size: 12px;
            }
            .action-stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin: 15px 0;
            }
            .stat-item {
                background: rgba(255, 255, 255, 0.8);
                padding: 10px;
                border-radius: 8px;
                text-align: center;
            }
            .stat-value {
                font-size: 1.5em;
                font-weight: 700;
                color: #667eea;
            }
            .stat-label {
                font-size: 0.9em;
                color: #666;
                margin-top: 5px;
            }
            @media (max-width: 768px) {
                .main-content {
                    flex-direction: column;
                }
                .video-section, .control-section {
                    min-width: 100%;
                }
            }
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    </head>
    <body>
        <div class="container">
            <h1>🎮 Real-time Interactive Video Generation</h1>
            
            <div class="main-content">
                <div class="video-section">
                    <div class="video-container">
                        <img id="videoFrame" src="" alt="Generated video frames will appear here" style="display: none;">
                        <div id="placeholder" style="color: white; font-size: 1.2em; padding: 100px 20px;">
                            🎬 Generated video frames will appear here<br>
                            <small style="opacity: 0.7;">Connect and start generation to begin</small>
                        </div>
                    </div>
                    
                    <div class="status" id="status" style="display: none;"></div>
                    
                    <div class="progress" id="progressContainer" style="display: none;">
                        <div class="progress-bar" id="progressBar" style="width: 0%;">
                            <div class="progress-text" id="progressText">0%</div>
                        </div>
                    </div>
                </div>
                
                <div class="live-control-section">
                    <div class="realtime-controls" id="realtimeControls">
                        <div class="realtime-title">🎮 Live Action Control</div>
                        
                                                <div class="button-group">
                            <button onclick="connectWebSocket()" id="connectBtn">🔌 Connect</button>
                            <button onclick="disconnectWebSocket()" id="disconnectBtn" disabled>🔌 Disconnect</button>
                            <button onclick="clearActions()" id="clearBtn" disabled>🗑️ Clear Actions</button>
                        </div>
                        
                        <div class="button-group">
                            <button onclick="startRealtimeGeneration()" id="realtimeGenBtn" disabled>⚡ Generate Now!</button>
                            <button onclick="startContinuousGeneration()" id="continuousGenBtn" disabled>🎬 Start Recording!</button>
                        </div>
                        
                        <div class="button-group">
                            <button onclick="initSingleFrameGenerator()" id="initGenBtn" disabled>🔧 Init Generator</button>
                            <button onclick="generateSingleFrame()" id="singleFrameBtn" disabled>🎨 Generate Frame</button>
                            <button onclick="resetSingleFrameGenerator()" id="resetGenBtn" disabled>🔄 Reset</button>
                        </div>
                        
                        <div class="button-group">
                            <button onclick="startContinuousSingleFrameGeneration()" id="startContFrameBtn" disabled>🎬 Start Continuous</button>
                            <button onclick="stopContinuousSingleFrameGeneration()" id="stopContFrameBtn" disabled>⏹️ Stop Continuous</button>
                        </div>
                        
                        <div class="control-group">
                            <label>Stop Action:</label>
                            <select id="stopAction">
                                <option value="ESC">ESC Key</option>
                                <option value="attack">X Key</option>
                                <option value="use">Right Click</option>
                                <option value="jump">Space</option>
                            </select>
                        </div>
            
                        <div class="action-stats">
                            <div class="stat-item">
                                <div class="stat-value" id="actionCount">0</div>
                                <div class="stat-label">Actions Sent</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value" id="bufferSize">0</div>
                                <div class="stat-label">Buffer Size</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value" id="currentFps">0</div>
                                <div class="stat-label">Current FPS</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value" id="frameCount">0</div>
                                <div class="stat-label">Generated Frames</div>
                            </div>
                        </div>
                        
                        <div style="text-align: center; margin: 15px 0; color: #2d3748;">
                            <strong>🎮 Controls:</strong><br>
                            <small>WASD: Movement | Mouse: Camera | Space: Jump | Shift: Sneak<br>
                            X: Attack | E: Use | Q: Drop | 1-9: Hotbar</small>
            </div>
            
                        <div class="action-display" id="actionDisplay">
                            Action stream will appear here...
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Generation Settings moved to bottom -->
            <div class="control-section" style="margin-top: 30px; border-top: 2px solid #e2e8f0;">
                <div class="controls-panel">
                    <h3 style="margin-top: 20px; color: #333;">🔧 Generation Settings</h3>
                    
                    <div class="control-group">
                        <label>Mode:</label>
                        <div class="input-mode-toggle">
                            <div class="mode-btn active" onclick="toggleMode('file')" id="fileModeBtn">📁 File Mode</div>
                            <div class="mode-btn" onclick="toggleMode('realtime')" id="realtimeModeBtn">🎮 Live Mode</div>
                        </div>
                    </div>
                    
                    <div class="control-group">
                        <label>Prompt:</label>
                        <input type="text" id="promptPath" value="sample_data/sample_image_0.png">
                    </div>
                    
                    <div class="control-group" id="actionsGroup">
                        <label>Actions:</label>
                        <input type="text" id="actionsPath" value="sample_data/sample_actions_0.one_hot_actions.pt">
                    </div>
                    
                    <div class="control-group">
                        <label>Frames:</label>
                        <input type="number" id="numFrames" value="32" min="1" max="100">
                    </div>
                    
                    <div class="control-group">
                        <label>DDIM Steps:</label>
                        <input type="number" id="ddimSteps" value="20" min="1" max="50">
                    </div>
                    
                    <div class="control-group">
                        <label>Target FPS:</label>
                        <input type="number" id="targetFps" value="20" min="1" max="60">
                    </div>
                    
                    <div class="button-group">
                        <button onclick="startGeneration()" id="startBtn">🚀 Start Generation</button>
                        <button onclick="stopGeneration()" id="stopBtn" disabled>⏹️ Stop</button>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Generation state
            let isGenerating = false;
            let eventSource = null;
            let currentFrame = 0;
            let totalFrames = 0;
            let currentMode = 'file';

            // WebSocket and action tracking
            let socket = null;
            let isConnected = false;
            let actionCount = 0;
            let lastActionTime = 0;
            let fpsCalculation = { 
                timestamps: [], 
                currentFps: 0 
            };
            
            // Current action state
            let currentAction = {
                forward: 0, back: 0, left: 0, right: 0,
                jump: 0, sneak: 0, sprint: 0, attack: 0, use: 0,
                cameraX: 0, cameraY: 0,
                'hotbar.1': 0, 'hotbar.2': 0, 'hotbar.3': 0, 'hotbar.4': 0, 'hotbar.5': 0,
                'hotbar.6': 0, 'hotbar.7': 0, 'hotbar.8': 0, 'hotbar.9': 0,
                inventory: 0, ESC: 0, swapHands: 0, pickItem: 0, drop: 0
            };
            
            // Mode switching
            function toggleMode(mode) {
                currentMode = mode;
                document.getElementById('fileModeBtn').classList.toggle('active', mode === 'file');
                document.getElementById('realtimeModeBtn').classList.toggle('active', mode === 'realtime');
                document.getElementById('realtimeControls').style.display = mode === 'realtime' ? 'block' : 'none';
                document.getElementById('actionsGroup').style.display = mode === 'file' ? 'flex' : 'none';
            }
            
            // WebSocket connection
            function connectWebSocket() {
                if (socket) return;
                
                try {
                    socket = io();
                    
                    socket.on('connect', () => {
                        isConnected = true;
                        updateConnectionStatus('Connected to action stream', 'connected');
                        document.getElementById('connectBtn').disabled = true;
                        document.getElementById('disconnectBtn').disabled = false;
                        document.getElementById('clearBtn').disabled = false;
                        setupInputHandlers();
                    });
                    
                    socket.on('disconnect', () => {
                        isConnected = false;
                        updateConnectionStatus('Disconnected from server', 'error');
                        document.getElementById('connectBtn').disabled = false;
                        document.getElementById('disconnectBtn').disabled = true;
                        document.getElementById('clearBtn').disabled = true;
                        document.getElementById('realtimeGenBtn').disabled = true;
                        document.getElementById('continuousGenBtn').disabled = true;
                        document.getElementById('initGenBtn').disabled = true;
                        document.getElementById('singleFrameBtn').disabled = true;
                        document.getElementById('resetGenBtn').disabled = true;
                        document.getElementById('startContFrameBtn').disabled = true;
                        document.getElementById('stopContFrameBtn').disabled = true;
                        removeInputHandlers();
                    });
                    
                    socket.on('action_received', (data) => {
                        document.getElementById('bufferSize').textContent = data.buffer_size;
                        
                        // Enable immediate generation if enough actions
                        if (data.ready_for_generation) {
                            updateStatus('Ready for real-time generation! 🚀', 'completed');
                            document.getElementById('realtimeGenBtn').disabled = false;
                            document.getElementById('continuousGenBtn').disabled = false;
                            document.getElementById('initGenBtn').disabled = false;
                        }
                    });
                    
                    socket.on('realtime_frame', (data) => {
                        // Display real-time generated frames
                        const img = document.getElementById('videoFrame');
                        img.src = 'data:image/jpeg;base64,' + data.frame;
                        img.style.display = 'block';
                        document.getElementById('placeholder').style.display = 'none';
                        
                        updateStatus('Real-time frame ' + (data.frame_idx + 1) + '/' + data.total_frames, 'generating');
                    });
                    
                    socket.on('realtime_generation_started', (data) => {
                        updateStatus('Real-time generation started! Frames: ' + data.num_frames, 'generating');
                        document.getElementById('realtimeGenBtn').disabled = true;
                    });
                    
                    socket.on('realtime_generation_complete', (data) => {
                        updateStatus('Real-time generation completed! 🎉', 'completed');
                        document.getElementById('realtimeGenBtn').disabled = false;
                    });
                    
                    socket.on('continuous_frame', (data) => {
                        // Display continuous generated frames
                        const img = document.getElementById('videoFrame');
                        img.src = 'data:image/jpeg;base64,' + data.frame;
                        img.style.display = 'block';
                        document.getElementById('placeholder').style.display = 'none';
                        
                        updateStatus('Recording frame ' + (data.frame_idx + 1) + (data.should_stop ? ' (stopping...)' : ''), 'generating');
                    });
                    
                    socket.on('continuous_generation_started', (data) => {
                        updateStatus('Recording started! Press ' + data.stop_action + ' to stop and save. Max: ' + data.max_frames + ' frames', 'generating');
                        document.getElementById('continuousGenBtn').disabled = true;
                        document.getElementById('realtimeGenBtn').disabled = true;
                    });
                    
                    socket.on('continuous_generation_complete', (data) => {
                        updateStatus('Recording completed! ' + data.frame_count + ' frames saved to ' + data.output_path, 'completed');
                        document.getElementById('continuousGenBtn').disabled = false;
                        document.getElementById('realtimeGenBtn').disabled = false;
                    });
                    
                    socket.on('fps_updated', (data) => {
                        updateStatus('FPS updated to ' + data.fps, 'completed');
                    });
                    
                    socket.on('actions_cleared', (data) => {
                        updateStatus('Action buffer cleared', 'completed');
                        document.getElementById('bufferSize').textContent = '0';
                    });
                    
                    // 单帧生成器事件
                    socket.on('single_frame_generator_ready', (data) => {
                        updateStatus('Single frame generator ready! 🎨', 'completed');
                        document.getElementById('singleFrameBtn').disabled = false;
                        document.getElementById('resetGenBtn').disabled = false;
                        document.getElementById('startContFrameBtn').disabled = false;
                        document.getElementById('frameCount').textContent = data.frame_count;
                        
                        // 显示prompt帧
                        const img = document.getElementById('videoFrame');
                        img.src = 'data:image/jpeg;base64,' + data.prompt_frame;
                        img.style.display = 'block';
                        document.getElementById('placeholder').style.display = 'none';
                    });
                    
                    socket.on('single_frame_generated', (data) => {
                        // 显示生成的单帧
                        const img = document.getElementById('videoFrame');
                        img.src = 'data:image/jpeg;base64,' + data.frame;
                        
                        // 更新统计信息
                        document.getElementById('frameCount').textContent = data.frame_count;
                        
                        // 显示生成信息
                        const activeActions = Object.keys(data.action_used).filter(key => data.action_used[key] > 0);
                        const actionText = activeActions.length > 0 ? activeActions.join(', ') : 'none';
                        updateStatus('Frame ' + data.frame_count + ' generated! (' + data.generation_time + 's, actions: ' + actionText + ')', 'completed');
                    });
                    
                    socket.on('single_frame_generator_reset', (data) => {
                        updateStatus('Generator reset', 'completed');
                        document.getElementById('singleFrameBtn').disabled = true;
                        document.getElementById('resetGenBtn').disabled = true;
                        document.getElementById('startContFrameBtn').disabled = true;
                        document.getElementById('stopContFrameBtn').disabled = true;
                        document.getElementById('frameCount').textContent = '0';
                    });
                    
                    // Continuous single frame generation events
                    socket.on('continuous_frame_generation_started', (data) => {
                        updateStatus('Continuous frame generation started! 🎬 Rate: ' + data.frame_rate, 'generating');
                        document.getElementById('startContFrameBtn').disabled = true;
                        document.getElementById('stopContFrameBtn').disabled = false;
                    });
                    
                    socket.on('continuous_frame_generated', (data) => {
                        // Display the generated continuous frame
                        const img = document.getElementById('videoFrame');
                        img.src = 'data:image/jpeg;base64,' + data.frame;
                        img.style.display = 'block';
                        document.getElementById('placeholder').style.display = 'none';
                        
                        // Update statistics
                        document.getElementById('frameCount').textContent = data.frame_count;
                        
                        // Show active actions
                        const activeActions = Object.keys(data.action_used).filter(key => data.action_used[key] > 0);
                        const actionText = activeActions.length > 0 ? activeActions.join(', ') : 'none';
                        updateStatus('Continuous Frame #' + data.continuous_frame_index + ' (total: ' + data.frame_count + ') - ' + data.generation_time + 's - Actions: ' + actionText, 'generating');
                    });
                    
                    socket.on('continuous_frame_generation_stopped', (data) => {
                        updateStatus('Continuous generation stopped. Generated ' + data.total_frames_generated + ' frames total', 'completed');
                        document.getElementById('startContFrameBtn').disabled = false;
                        document.getElementById('stopContFrameBtn').disabled = true;
                    });
                    
                    socket.on('continuous_frame_generation_stop_confirmed', (data) => {
                        updateStatus('Continuous generation stopped successfully', 'completed');
                        document.getElementById('startContFrameBtn').disabled = false;
                        document.getElementById('stopContFrameBtn').disabled = true;
                    });
                    
                    socket.on('error', (data) => {
                        updateStatus('Error: ' + data.message, 'error');
                    });
                    
                } catch (error) {
                    updateStatus('Connection failed: ' + error.message, 'error');
                }
            }
            
            function disconnectWebSocket() {
                if (socket) {
                    socket.disconnect();
                    socket = null;
                }
                removeInputHandlers();
            }
            
            function clearActions() {
                if (socket && isConnected) {
                    socket.emit('clear_actions');
                    document.getElementById('realtimeGenBtn').disabled = true;
                    document.getElementById('continuousGenBtn').disabled = true;
                }
            }
            
            function startRealtimeGeneration() {
                if (!socket || !isConnected) {
                    updateStatus('Please connect to WebSocket first', 'error');
                    return;
                }
                
                const params = {
                    prompt_path: document.getElementById('promptPath').value,
                    num_frames: Math.min(parseInt(document.getElementById('numFrames').value), 16), // Limit for real-time
                    ddim_steps: Math.min(parseInt(document.getElementById('ddimSteps').value), 10)    // Limit for speed
                };
                
                socket.emit('start_realtime_generation', params);
                updateStatus('Starting real-time generation...', 'generating');
            }
            
            function startContinuousGeneration() {
                if (!socket || !isConnected) {
                    updateStatus('Please connect to WebSocket first', 'error');
                    return;
                }
                
                const params = {
                    prompt_path: document.getElementById('promptPath').value,
                    max_frames: parseInt(document.getElementById('numFrames').value) * 3, // Allow longer recording
                    ddim_steps: Math.min(parseInt(document.getElementById('ddimSteps').value), 8),  // Limit for speed
                    stop_action: document.getElementById('stopAction').value,
                    output_path: 'continuous_video_' + Date.now() + '.mp4'
                };
                
                socket.emit('start_continuous_generation', params);
                updateStatus('Starting continuous recording... Use controls to interact and press ' + params.stop_action + ' to stop!', 'generating');
            }
            
            function initSingleFrameGenerator() {
                if (!socket || !isConnected) {
                    updateStatus('Please connect to WebSocket first', 'error');
                    return;
                }
                
                const params = {
                    prompt_path: document.getElementById('promptPath').value,
                    ddim_steps: Math.min(parseInt(document.getElementById('ddimSteps').value), 32)
                };
                
                socket.emit('init_single_frame_generator', params);
                updateStatus('Initializing single frame generator...', 'generating');
            }
            
            function generateSingleFrame() {
                if (!socket || !isConnected) {
                    updateStatus('Please connect to WebSocket first', 'error');
                    return;
                }
                
                socket.emit('generate_single_frame', {});
                updateStatus('Generating single frame...', 'generating');
            }
            
            function resetSingleFrameGenerator() {
                if (!socket || !isConnected) {
                    updateStatus('Please connect to WebSocket first', 'error');
                    return;
                }
                
                socket.emit('reset_single_frame_generator');
                updateStatus('Resetting generator...', 'generating');
            }
            
            function startContinuousSingleFrameGeneration() {
                if (!socket || !isConnected) {
                    updateStatus('Please connect to WebSocket first', 'error');
                    return;
                }
                
                socket.emit('start_continuous_single_frame_generation', {});
                updateStatus('Starting continuous single frame generation...', 'generating');
                
                // Update button states
                document.getElementById('startContFrameBtn').disabled = true;
                document.getElementById('stopContFrameBtn').disabled = false;
            }
            
            function stopContinuousSingleFrameGeneration() {
                if (!socket || !isConnected) {
                    updateStatus('Please connect to WebSocket first', 'error');
                    return;
                }
                
                socket.emit('stop_continuous_single_frame_generation');
                updateStatus('Stopping continuous single frame generation...', 'generating');
                
                // Update button states
                document.getElementById('startContFrameBtn').disabled = false;
                document.getElementById('stopContFrameBtn').disabled = true;
            }
            
            function updateConnectionStatus(message, type) {
                updateStatus(message, type);
            }
            
            // Input handling
            let keyStates = {};
            let mouseMovement = { x: 0, y: 0 };
            
            function setupInputHandlers() {
                // Keyboard events
                document.addEventListener('keydown', handleKeyDown);
                document.addEventListener('keyup', handleKeyUp);
                
                // Mouse events
                document.addEventListener('mousedown', handleMouseDown);
                document.addEventListener('mouseup', handleMouseUp);
                document.addEventListener('mousemove', handleMouseMove);
                
                // Prevent context menu on right click
                document.addEventListener('contextmenu', (e) => e.preventDefault());
                
                // Start action sending loop
                startActionLoop();
            }
            
            function removeInputHandlers() {
                document.removeEventListener('keydown', handleKeyDown);
                document.removeEventListener('keyup', handleKeyUp);
                document.removeEventListener('mousedown', handleMouseDown);
                document.removeEventListener('mouseup', handleMouseUp);
                document.removeEventListener('mousemove', handleMouseMove);
                document.removeEventListener('contextmenu', (e) => e.preventDefault());
                
                stopActionLoop();
            }
            
            function handleKeyDown(event) {
                const key = event.key.toLowerCase();
                keyStates[key] = true;
                
                // Map keys to actions
                switch(key) {
                    case 'w': currentAction.forward = 1; break;
                    case 's': currentAction.back = 1; break;
                    case 'a': currentAction.left = 1; break;
                    case 'd': currentAction.right = 1; break;
                    case ' ': currentAction.jump = 1; event.preventDefault(); break;
                    case 'shift': currentAction.sneak = 1; break;
                    case 'control': currentAction.sprint = 1; break;
                    case 'e': currentAction.use = 1; break;
                    case 'q': currentAction.drop = 1; break;
                    case 'x': currentAction.attack = 1; break;
                    case 'escape': currentAction.ESC = 1; break;
                    case 'tab': currentAction.inventory = 1; event.preventDefault(); break;
                    case 'f': currentAction.swapHands = 1; break;
                    case 'r': currentAction.pickItem = 1; break;
                    case '1': case '2': case '3': case '4': case '5':
                    case '6': case '7': case '8': case '9':
                        const hotbarKey = 'hotbar.' + key;
                        currentAction[hotbarKey] = 1;
                        break;
                }
            }
            
            function handleKeyUp(event) {
                const key = event.key.toLowerCase();
                keyStates[key] = false;
                
                // Reset actions on key up
                switch(key) {
                    case 'w': currentAction.forward = 0; break;
                    case 's': currentAction.back = 0; break;
                    case 'a': currentAction.left = 0; break;
                    case 'd': currentAction.right = 0; break;
                    case ' ': currentAction.jump = 0; break;
                    case 'shift': currentAction.sneak = 0; break;
                    case 'control': currentAction.sprint = 0; break;
                    case 'e': currentAction.use = 0; break;
                    case 'q': currentAction.drop = 0; break;
                    case 'x': currentAction.attack = 0; break;
                    case 'escape': currentAction.ESC = 0; break;
                    case 'tab': currentAction.inventory = 0; break;
                    case 'f': currentAction.swapHands = 0; break;
                    case 'r': currentAction.pickItem = 0; break;
                    case '1': case '2': case '3': case '4': case '5':
                    case '6': case '7': case '8': case '9':
                        const hotbarKey = 'hotbar.' + key;
                        currentAction[hotbarKey] = 0;
                        break;
                }
            }
            
            function handleMouseDown(event) {
                if (event.button === 2) { // Right click
                    currentAction.use = 1;
                }
            }
            
            function handleMouseUp(event) {
                if (event.button === 2) { // Right click
                    currentAction.use = 0;
                }
            }
            
            function handleMouseMove(event) {
                // Calculate camera movement (normalized to -1 to 1)
                const sensitivity = 0.01;
                const deltaX = event.movementX * sensitivity;
                const deltaY = event.movementY * sensitivity;
                
                // Clamp values and convert to expected range
                currentAction.cameraX = Math.max(-1, Math.min(1, deltaX));
                currentAction.cameraY = Math.max(-1, Math.min(1, -deltaY)); // Invert Y axis
                
                // Reset camera movement after a short delay to avoid continuous movement
                setTimeout(() => {
                    currentAction.cameraX = 0;
                    currentAction.cameraY = 0;
                }, 50);
            }
            
            // Action sending loop
            let actionLoopId = null;
            
            function startActionLoop() {
                if (actionLoopId) return;
                
                const targetFps = parseInt(document.getElementById('targetFps').value) || 20;
                const interval = 1000 / targetFps;
                
                actionLoopId = setInterval(() => {
                    if (socket && isConnected) {
                        sendAction();
                    }
                }, interval);
            }
            
            function stopActionLoop() {
                if (actionLoopId) {
                    clearInterval(actionLoopId);
                    actionLoopId = null;
                }
            }
            
            function sendAction() {
                const now = Date.now();
                
                // Send action to server
                socket.emit('action', { 
                    action: { ...currentAction } 
                });
                
                // Update statistics
                actionCount++;
                document.getElementById('actionCount').textContent = actionCount;
                
                // Calculate FPS
                fpsCalculation.timestamps.push(now);
                
                // Keep only timestamps from last second
                fpsCalculation.timestamps = fpsCalculation.timestamps.filter(
                    timestamp => now - timestamp < 1000
                );
                
                fpsCalculation.currentFps = fpsCalculation.timestamps.length;
                document.getElementById('currentFps').textContent = fpsCalculation.currentFps;
                
                // Update action display
                updateActionDisplay();
                
                lastActionTime = now;
            }
            
            function updateActionDisplay() {
                const display = document.getElementById('actionDisplay');
                const activeActions = Object.entries(currentAction)
                    .filter(([key, value]) => value !== 0)
                    .map(([key, value]) => key + ': ' + value)
                    .join('\\n');
                
                const timestamp = new Date().toLocaleTimeString();
                const actionText = activeActions || 'No active actions';
                
                display.textContent = '[' + timestamp + '] ' + actionText;
                display.scrollTop = display.scrollHeight;
            }
            
            // FPS update handler
            document.getElementById('targetFps').addEventListener('change', (event) => {
                const fps = parseInt(event.target.value);
                if (socket && isConnected) {
                    socket.emit('set_fps', { fps: fps });
                }
                
                // Restart action loop with new FPS
                if (actionLoopId) {
                    stopActionLoop();
                    startActionLoop();
                }
            });
            
            // Video generation functions
            function startGeneration() {
                if (isGenerating) return;
                
                if (currentMode === 'realtime' && !isConnected) {
                    updateStatus('Please connect to WebSocket first for realtime mode', 'error');
                    return;
                }

                const params = {
                    prompt_path: document.getElementById('promptPath').value,
                    num_frames: parseInt(document.getElementById('numFrames').value),
                    ddim_steps: parseInt(document.getElementById('ddimSteps').value),
                    n_prompt_frames: 1,
                    realtime_actions: currentMode === 'realtime' ? 'true' : 'false'
                };
                
                if (currentMode === 'file') {
                    params.actions_path = document.getElementById('actionsPath').value;
                }

                const queryString = new URLSearchParams(params).toString();
                
                // Start SSE connection
                eventSource = new EventSource('/stream?' + queryString);
                
                eventSource.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    
                    if (data.status === 'generating' && data.frame) {
                        // Update frame
                        const img = document.getElementById('videoFrame');
                        img.src = 'data:image/jpeg;base64,' + data.frame;
                        img.style.display = 'block';
                        document.getElementById('placeholder').style.display = 'none';
                        
                        // Update progress
                        currentFrame = data.frame_idx + 1;
                        totalFrames = data.total_frames;
                        updateProgress();
                        
                        // Update status
                        updateStatus('Generating frame ' + currentFrame + '/' + totalFrames, 'generating');
                        
                    } else if (data.status === 'completed') {
                        updateStatus('Generation completed!', 'completed');
                        stopGeneration();
                        
                    } else if (data.status === 'error') {
                        updateStatus('Error: ' + (data.error || 'Unknown error'), 'error');
                        stopGeneration();
                    }
                };
                
                eventSource.onerror = function(event) {
                    updateStatus('Connection error', 'error');
                    stopGeneration();
                };

                isGenerating = true;
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('progressContainer').style.display = 'block';
                updateStatus('Starting generation...', 'generating');
            }

            function stopGeneration() {
                if (eventSource) {
                    eventSource.close();
                    eventSource = null;
                }
                
                // Stop generation on server
                fetch('/stop', { method: 'POST' });
                
                isGenerating = false;
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
                
                if (document.getElementById('status').className.includes('generating')) {
                    updateStatus('Generation stopped', 'error');
                }
            }

            function updateStatus(message, type) {
                const statusEl = document.getElementById('status');
                statusEl.textContent = message;
                statusEl.className = 'status ' + type;
                statusEl.style.display = 'block';
            }

            function updateProgress() {
                if (totalFrames > 0) {
                    const progress = (currentFrame / totalFrames) * 100;
                    document.getElementById('progressBar').style.width = progress + '%';
                    document.getElementById('progressText').textContent = Math.round(progress) + '%';
                }
            }

            // Cleanup on page unload
            window.addEventListener('beforeunload', function() {
                stopGeneration();
                disconnectWebSocket();
            });
            
            // Initialize page
            document.addEventListener('DOMContentLoaded', function() {
                updateStatus('Ready to generate. Choose a mode and start!', 'completed');
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/stream')
def video_stream():
    """Server-Sent Events endpoint for video streaming"""
    global generation_thread, is_generating, frame_queue
    
    # Get parameters from query string outside the generator
    prompt_path = request.args.get('prompt_path', 'sample_data/sample_image_0.png')
    actions_path = request.args.get('actions_path', 'sample_data/sample_actions_0.one_hot_actions.pt')
    num_frames = int(request.args.get('num_frames', 32))
    ddim_steps = int(request.args.get('ddim_steps', 10))
    n_prompt_frames = int(request.args.get('n_prompt_frames', 1))
    use_realtime_actions = request.args.get('realtime_actions', 'false').lower() == 'true'
    
    def generate():
        global is_generating
        
        with generation_lock:
            if is_generating:
                yield f"data: {json.dumps({'status': 'error', 'error': 'Generation already in progress'})}\n\n"
                return
                
            # Clear the queue
            while not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Start generation in background thread
            generation_thread = Thread(
                target=generator.generate_video_stream,
                args=(prompt_path, actions_path, num_frames, n_prompt_frames, None, ddim_steps, use_realtime_actions)
            )
            generation_thread.start()
        
        # Stream frames as they become available
        while True:
            try:
                # Get frame from queue with timeout
                frame_data = frame_queue.get(timeout=30)
                
                if frame_data['status'] == 'completed':
                    yield f"data: {json.dumps({'status': 'completed'})}\n\n"
                    break
                elif frame_data['status'] == 'error':
                    yield f"data: {json.dumps({'status': 'error', 'error': frame_data.get('error', 'Unknown error')})}\n\n"
                    break
                elif frame_data['frame'] is not None:
                    # Encode frame as base64
                    frame_b64 = base64.b64encode(frame_data['frame']).decode('utf-8')
                    
                    response_data = {
                        'status': 'generating',
                        'frame': frame_b64,
                        'frame_idx': frame_data['frame_idx'],
                        'total_frames': frame_data['total_frames']
                    }
                    
                    yield f"data: {json.dumps(response_data)}\n\n"
                
                if not is_generating:
                    break
                    
            except queue.Empty:
                # Timeout - check if generation is still running
                if not is_generating:
                    yield f"data: {json.dumps({'status': 'error', 'error': 'Generation timed out'})}\n\n"
                    break
                continue
            except Exception as e:
                yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"
                break
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/stop', methods=['POST'])
def stop_generation():
    """Stop the current generation"""
    global is_generating, generation_thread
    
    with generation_lock:
        is_generating = False
        
        if generation_thread and generation_thread.is_alive():
            # Wait for thread to finish
            generation_thread.join(timeout=5)
    
    return jsonify({'status': 'stopped'})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': device,
        'is_generating': is_generating,
        'cuda_available': torch.cuda.is_available()
    })

if __name__ == '__main__':
    print(f"Starting video streaming server on device: {device}")
    print("Available at: http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
