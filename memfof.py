# Added December 3, 2025 by Clara Burgert using Claude

import rp
import torch
import torchvision.transforms
import numpy as np

class MemfofOpticalFlow(rp.CachedInstances):
    def __init__(self, device, version='MEMFOF-Tartan-T-TSKH'):
        """
        MEMFOF: Multi-frame optical flow that processes exactly 3 frames at a time.

        Available versions:
        - 'MEMFOF-Tartan-T-TSKH' (default, best for real-world videos)
        - 'MEMFOF-Tartan-T-TSKH-kitti' (fine-tuned on KITTI dataset)
        - 'MEMFOF-Tartan-T-TSKH-sintel' (fine-tuned on Sintel dataset)
        - 'MEMFOF-Tartan-T-TSKH-spring' (fine-tuned on Spring dataset)
        """

        # Install memfof package if not available
        try:
            from memfof import MEMFOF
        except ImportError:
            print("The 'memfof' package is not installed. Attempting to install it now...")
            rp.pip_import("memfof", "git+https://github.com/msu-video-group/memfof", auto_yes=True)
            from memfof import MEMFOF

        models = {
            'MEMFOF-Tartan': 'egorchistov/optical-flow-MEMFOF-Tartan',
            'MEMFOF-Tartan-T': 'egorchistov/optical-flow-MEMFOF-Tartan-T',
            'MEMFOF-Tartan-T-TSKH': 'egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH',
            'MEMFOF-Tartan-T-TSKH-kitti': 'egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH-kitti',
            'MEMFOF-Tartan-T-TSKH-sintel': 'egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH-sintel',
            'MEMFOF-Tartan-T-TSKH-spring': 'egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH-spring',
        }
        assert version in models, f"Version must be one of {list(models.keys())}"

        model_id = models[version]
        model = MEMFOF.from_pretrained(model_id)
        model = model.eval().to(device)

        self.version = version
        self.device = device
        self.model = model

    def _preprocess_frames(self, frames):
        """Convert frames to tensor format expected by MEMFOF"""
        if rp.is_torch_tensor(frames):
            # Assume it's already in the right format
            frames_tensor = frames.to(self.device).float()
        else:
            # Convert list of images to tensor
            processed = []
            for frame in frames:
                if rp.is_image(frame):
                    frame = rp.as_float_image(rp.as_rgb_image(frame))
                    frame = rp.as_torch_image(frame)
                frame = frame.to(self.device).float()
                processed.append(frame)
            frames_tensor = torch.stack(processed, dim=0)

        # Ensure shape is THW3 or BTHW3
        if frames_tensor.ndim == 4:  # THW3
            frames_tensor = frames_tensor.unsqueeze(0)  # Add batch dimension

        # Rearrange from BTHW3 to BT3HW
        if frames_tensor.shape[-1] == 3:  # Channels last
            frames_tensor = frames_tensor.permute(0, 1, 4, 2, 3)

        # Get dimensions
        B, T, C, H, W = frames_tensor.shape

        # Pad to multiple of 16
        new_H = ((H + 15) // 16) * 16
        new_W = ((W + 15) // 16) * 16

        if new_H != H or new_W != W:
            frames_resized = torch.nn.functional.interpolate(
                frames_tensor.reshape(B * T, C, H, W),
                size=(new_H, new_W),
                mode='bilinear',
                align_corners=False
            ).reshape(B, T, C, new_H, new_W)
        else:
            frames_resized = frames_tensor

        # Scale to [0, 255] range
        frames_resized = frames_resized * 255.0

        return frames_resized, (H, W)

    def __call__(self, *frames):
        """
        Process frames and return optical flows.

        Args:
            *frames: Can be:
                - Exactly 3 frames: Returns (backward_flow, forward_flow)
                - 2 frames: For RAFT compatibility, duplicates last frame
                - Video (list/array): Processed with process_video()

        Returns:
            - For 3 frames: Tuple (backward_flow, forward_flow)
              backward_flow: from second_frame → first_frame
              forward_flow: from second_frame → third_frame
            - For 2 frames: Single flow tensor from first → second (RAFT compatibility)
            - For video: Calls process_video() internally
        """

        # Handle different input formats
        if len(frames) == 1:
            # Single argument - could be a video array/tensor or list of frames
            arg = frames[0]
            if isinstance(arg, list) or (hasattr(arg, 'shape') and len(arg.shape) >= 4):
                # It's a video, use process_video
                return self.process_video(arg)
            else:
                raise ValueError("Single frame not supported, need at least 2 frames")

        elif len(frames) == 2:
            # RAFT compatibility mode - duplicate last frame
            frames_to_process = [frames[0], frames[1], frames[1]]
            frames_tensor, (orig_H, orig_W) = self._preprocess_frames(frames_to_process)

            with torch.no_grad():
                output = self.model(frames_tensor)
                # Get forward flow from frame 0 to frame 1
                forward_flow = output["flow"][-1][0, 0]  # [2, H, W]

                # Resize if needed
                if forward_flow.shape[1] != orig_H or forward_flow.shape[2] != orig_W:
                    resize = torchvision.transforms.Resize((orig_H, orig_W))
                    forward_flow = resize(forward_flow[None])[0]
                    forward_flow[0] *= orig_W / frames_tensor.shape[-1]
                    forward_flow[1] *= orig_H / frames_tensor.shape[-2]

            return forward_flow

        elif len(frames) == 3:
            # Standard MEMFOF 3-frame processing
            frames_tensor, (orig_H, orig_W) = self._preprocess_frames(frames)

            with torch.no_grad():
                output = self.model(frames_tensor)
                # Extract backward and forward flows
                flows = output["flow"][-1][0]  # [2, 2, H, W]
                backward_flow = flows[0]  # Flow from frame 1 → frame 0
                forward_flow = flows[1]   # Flow from frame 1 → frame 2

                # Resize if needed
                if backward_flow.shape[1] != orig_H or backward_flow.shape[2] != orig_W:
                    resize = torchvision.transforms.Resize((orig_H, orig_W))
                    scale_h = orig_H / backward_flow.shape[1]
                    scale_w = orig_W / backward_flow.shape[2]

                    backward_flow = resize(backward_flow[None])[0]
                    backward_flow[0] *= scale_w
                    backward_flow[1] *= scale_h

                    forward_flow = resize(forward_flow[None])[0]
                    forward_flow[0] *= scale_w
                    forward_flow[1] *= scale_h

            return backward_flow, forward_flow

        else:
            # More than 3 frames - treat as video
            return self.process_video(list(frames))

    def process_video(self, video):
        """
        Process a video and return optical flows for all consecutive frame pairs.

        For a video with N frames, returns N-1 optical flows.
        Uses sliding 3-frame windows with proper edge handling.

        Args:
            video: Either:
                - List of frames (numpy arrays, PIL images, or torch tensors)
                - Video array (THW3 or T3HW format)

        Returns:
            List of N-1 flow tensors, where flow[i] is from frame[i] → frame[i+1]

        Example:
            For video [f0, f1, f2, f3, f4]:
            - Window 1: [f0, f0, f1] → get forward flow f0→f1
            - Window 2: [f0, f1, f2] → get forward flow f1→f2
            - Window 3: [f1, f2, f3] → get forward flow f2→f3
            - Window 4: [f2, f3, f4] → get forward flow f3→f4
            Returns: [flow_01, flow_12, flow_23, flow_34]
        """

        # Convert to list of frames if needed
        if not isinstance(video, list):
            if hasattr(video, 'shape'):
                # It's a tensor/array
                if rp.is_torch_tensor(video):
                    frames = [video[i] for i in range(video.shape[0])]
                else:
                    # NumPy array
                    frames = [video[i] for i in range(len(video))]
            else:
                frames = list(video)
        else:
            frames = video

        n_frames = len(frames)
        if n_frames < 2:
            raise ValueError(f"Need at least 2 frames for video processing, got {n_frames}")

        flows = []
        fmap_cache = [None] * 3

        with torch.no_grad():
            for i in range(n_frames - 1):
                # Construct 3-frame window
                if i == 0:
                    # First window: duplicate first frame
                    window = [frames[0], frames[0], frames[1]]
                elif i == n_frames - 2:
                    # Last window: duplicate last frame
                    window = [frames[i], frames[i+1], frames[i+1]]
                else:
                    # Middle windows: use proper 3-frame window
                    window = [frames[i], frames[i+1], frames[i+2] if i+2 < n_frames else frames[i+1]]

                # Process window
                frames_tensor, (orig_H, orig_W) = self._preprocess_frames(window)
                output = self.model(frames_tensor, fmap_cache=fmap_cache)

                # Extract the flow we need
                if i == 0:
                    # First window: use forward flow (duplicate frame → next frame)
                    flow = output["flow"][-1][0, 1]  # Forward flow
                else:
                    # Other windows: use backward flow from second frame
                    flow = output["flow"][-1][0, 0]  # Backward flow (but points forward in our window)

                # Actually, let me reconsider the logic...
                # For each window [A, B, C], MEMFOF gives:
                # - backward_flow: B → A
                # - forward_flow: B → C

                # What we want is flow from frame[i] → frame[i+1]

                if i == 0:
                    # Window [f0, f0, f1]: forward flow is f0→f1 (what we want!)
                    flow = output["flow"][-1][0, 1]
                else:
                    # Window [f_{i-1}, f_i, f_{i+1}]: forward flow is f_i→f_{i+1} (what we want!)
                    flow = output["flow"][-1][0, 1]

                # Resize if needed
                if flow.shape[1] != orig_H or flow.shape[2] != orig_W:
                    resize = torchvision.transforms.Resize((orig_H, orig_W))
                    flow = resize(flow[None])[0]
                    flow[0] *= orig_W / frames_tensor.shape[-1]
                    flow[1] *= orig_H / frames_tensor.shape[-2]

                flows.append(flow)

                # Update cache for efficiency (shift the feature maps)
                fmap_cache = output.get("fmap_cache", [None] * 3)
                if i < n_frames - 2:
                    # Shift cache for next window
                    fmap_cache = [fmap_cache[1], fmap_cache[2], None]

        return flows

    def process_video_bidirectional(self, video):
        """
        Process a video and return both forward and backward flow videos.

        Args:
            video: Video as list of frames or array

        Returns:
            Tuple (forward_flows, backward_flows) where each is a list of N-1 flows
            forward_flows[i]: flow from frame[i] → frame[i+1]
            backward_flows[i]: flow from frame[i+1] → frame[i]
        """
        # Get forward flows
        forward_flows = self.process_video(video)

        # For backward flows, we can negate forward flows or compute separately
        # Let's compute them properly using MEMFOF's backward flow output
        if not isinstance(video, list):
            frames = [video[i] for i in range(len(video))]
        else:
            frames = video

        n_frames = len(frames)
        backward_flows = []
        fmap_cache = [None] * 3

        with torch.no_grad():
            for i in range(n_frames - 1):
                # Construct window centered on the frame pair
                if i == 0:
                    window = [frames[0], frames[1], frames[2] if n_frames > 2 else frames[1]]
                elif i == n_frames - 2:
                    window = [frames[i-1] if i > 0 else frames[0], frames[i], frames[i+1]]
                else:
                    window = [frames[i], frames[i+1], frames[i+2]]

                frames_tensor, (orig_H, orig_W) = self._preprocess_frames(window)
                output = self.model(frames_tensor, fmap_cache=fmap_cache)

                # Get backward flow (from second frame to first in window)
                flow = output["flow"][-1][0, 0]  # Backward flow

                # Resize if needed
                if flow.shape[1] != orig_H or flow.shape[2] != orig_W:
                    resize = torchvision.transforms.Resize((orig_H, orig_W))
                    flow = resize(flow[None])[0]
                    flow[0] *= orig_W / frames_tensor.shape[-1]
                    flow[1] *= orig_H / frames_tensor.shape[-2]

                backward_flows.append(flow)

                # Update cache
                fmap_cache = output.get("fmap_cache", [None] * 3)
                if i < n_frames - 2:
                    fmap_cache = [fmap_cache[1], fmap_cache[2], None]

        return forward_flows, backward_flows