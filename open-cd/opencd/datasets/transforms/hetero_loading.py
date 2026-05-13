import numpy as np
import rasterio
from mmcv.transforms import BaseTransform
from opencd.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadHeteroImagesFromFile(BaseTransform):
    """Custom Loader for Heterogeneous EO-SAR Image Pairs."""

    def transform(self, results: dict) -> dict:
        # Safely unpack Open-CD's list-based paths
        if isinstance(results['img_path'], list):
            pre_path = results['img_path'][0]
            post_path = results['img_path'][1]
        else:
            pre_path = results['img_path']
            post_path = results['img_path2']

        # 1. Load EO (Pre-event)
        with rasterio.open(pre_path) as src:
            eo = src.read().transpose(1, 2, 0).astype(np.float32)
            if eo.max() > 255:
                eo = eo / 10000.0  # Sentinel-2 scaling
            else:
                eo = eo / 255.0
            eo = np.clip(eo, 0, 1)

        # 2. Load SAR (Post-event)
        with rasterio.open(post_path) as src:
            sar = src.read().transpose(1, 2, 0).astype(np.float32)
            sar_db = 10 * np.log10(sar + 1e-8)
            sar_min, sar_max = np.min(sar_db), np.max(sar_db)
            if sar_max > sar_min:
                sar_norm = (sar_db - sar_min) / (sar_max - sar_min)
            else:
                sar_norm = sar_db

        # 3. Package as a LIST! Do NOT concatenate here.
        results['img'] = [eo, sar_norm]
        results['img_shape'] = eo.shape[:2]
        results['ori_shape'] = eo.shape[:2]

        return results

@TRANSFORMS.register_module()
class BinarizeLabels(BaseTransform):
    """Forces ground truth masks to be strictly 0 and 1, preventing CUDA out-of-bounds errors."""
    def transform(self, results: dict) -> dict:
        if 'gt_seg_map' in results:
            gt = results['gt_seg_map']
            # Convert visual masks (255) or multi-class damage indexes (2, 3, 4) strictly to Class 1
            gt = np.where(gt == 255, 1, gt)
            gt = np.where(gt > 1, 1, gt)
            results['gt_seg_map'] = gt
        return results


from mmseg.engine.hooks import SegVisualizationHook
from mmengine.registry import HOOKS
import os
import cv2
import numpy as np

@HOOKS.register_module(force=True)
class HeteroVisHook(SegVisualizationHook):
    def _after_iter(self, runner, batch_idx, data_batch, outputs, mode='val'):
        if getattr(self, 'draw', False) is False or mode == 'train':
            return
            
        interval = getattr(self, 'interval', 1)
        if self.every_n_inner_iters(batch_idx, interval):
            for data_sample in outputs:
                try:
                    img_path = data_sample.img_path
                    # Safely extract the EO (Pre-event) path
                    eo_path = img_path[0] if isinstance(img_path, (list, tuple)) else img_path
                        
                    name = os.path.basename(eo_path) if eo_path else f'{mode}_{batch_idx}'
                    
                    # Safely determine the output directory
                    out_dir = getattr(self, 'out_dir', None)
                    if out_dir is None and runner.work_dir is not None:
                        out_dir = os.path.join(runner.work_dir, 'predictions')
                    
                    if out_dir is not None:
                        os.makedirs(out_dir, exist_ok=True)
                        out_file = os.path.join(out_dir, f'{name}_{runner.iter}.png')
                        
                        img = cv2.imread(eo_path)
                        if img is None:
                            img = np.zeros((256, 256, 3), dtype=np.uint8)
                            
                        panels = []
                        
                        # THE ULTIMATE FIX: Bypass the brittle Visualizer API completely!
                        # We use native OpenCV to draw the masks and save them side-by-side.
                        
                        # 1. Draw Ground Truth (Green Overlay)
                        if getattr(self, 'draw_gt', True) and hasattr(data_sample, 'gt_sem_seg'):
                            gt = data_sample.gt_sem_seg.data.squeeze().cpu().numpy()
                            gt_overlay = img.copy()
                            gt_overlay[gt == 1] = [0, 255, 0] # BGR Green
                            panels.append(cv2.addWeighted(gt_overlay, 0.6, img, 0.4, 0))
                            
                        # 2. Draw Prediction (Red Overlay)
                        if getattr(self, 'draw_pred', True) and hasattr(data_sample, 'pred_sem_seg'):
                            pred = data_sample.pred_sem_seg.data.squeeze().cpu().numpy()
                            pred_overlay = img.copy()
                            pred_overlay[pred == 1] = [0, 0, 255] # BGR Red
                            panels.append(cv2.addWeighted(pred_overlay, 0.6, img, 0.4, 0))
                            
                        if not panels:
                            panels.append(img)
                            
                        # Concatenate side-by-side and save
                        vis_img = np.concatenate(panels, axis=1)
                        cv2.imwrite(out_file, vis_img)
                        
                except Exception as e:
                    # Fail silently to guarantee the evaluation loop never crashes
                    pass
