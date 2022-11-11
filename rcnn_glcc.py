# Copyright (c) Facebook, Inc. and its affiliates.
# Modified for GLCC.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

# ROI Heads
import inspect
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, ROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.backbone.resnet import BottleneckBlock, ResNet
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers

from detectron2.layers import cross_entropy

#__all__ = ["GeneralizedRCNN", "ProposalNetwork"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN_GLCC(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        glcc_on: bool = False,
        glcc_output: bool = False,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        self.glcc_on = glcc_on
        self.glcc_output = glcc_output  # For inference
        
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "glcc_on": cfg.MODEL.GLCC_ON,
            "glcc_output": cfg.MODEL.GLCC_OUTPUT  # For inference
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)
            
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        # GLCC labels
        if self.glcc_on:
            fish_classes = torch.as_tensor(
                [ x['fish_class'] for x in batched_inputs], device=self.device )   #glcc
        else:
            fish_classes = None

        # Backbone
        features = self.backbone(images.tensor)

        # RPN
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
            
        # Head
        #_, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, fish_classes)  #glcc
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        #    losses.update( loss_glcc=detector_losses['loss_glcc'] )
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses
    
    # RCNN
    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            #results, _ = self.roi_heads(images, features, proposals, None)
            results, pred_fishes = self.roi_heads(images, features, proposals, None)  #glcc
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
            
        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            results = GeneralizedRCNN_GLCC._postprocess(results, batched_inputs, images.image_sizes)
            
        if self.glcc_on and self.glcc_output:
            return results, pred_fishes
        else:
            return results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results




@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads_GLCC(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    See :paper:`ResNet` Appendix A.
    """

    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        pooler: ROIPooler,
        res5: nn.Module,
        box_predictor: nn.Module,
        mask_head: Optional[nn.Module] = None,
        glcc: nn.Module = None,  #glcc
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            in_features (list[str]): list of backbone feature map names to use for
                feature extraction
            pooler (ROIPooler): pooler to extra region features from backbone
            res5 (nn.Sequential): a CNN to compute per-region features, to be used by
                ``box_predictor`` and ``mask_head``. Typically this is a "res5"
                block from a ResNet.
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_head (nn.Module): transform features to make mask predictions
        """
        super().__init__(**kwargs)
        self.in_features = in_features
        self.pooler = pooler
        if isinstance(res5, (list, tuple)):
            res5 = nn.Sequential(*res5)
        self.res5 = res5
        self.box_predictor = box_predictor
        self.mask_on = mask_head is not None
        if self.mask_on:
            self.mask_head = mask_head
        
        #glcc
        self.glcc_on = glcc is not None
        if self.glcc_on:
            self.glcc = glcc

    @classmethod
    def from_config(cls, cfg, input_shape):
        # fmt: off
        ret = super().from_config(cfg)
        in_features = ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        mask_on           = cfg.MODEL.MASK_ON
        glcc_on           = cfg.MODEL.GLCC_ON  #glcc
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(in_features) == 1

        ret["pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        # Compatbility with old moco code. Might be useful.
        # See notes in StandardROIHeads.from_config
        if not inspect.ismethod(cls._build_res5_block):
            logger.warning(
                "The behavior of _build_res5_block may change. "
                "Please do not depend on private methods."
            )
            cls._build_res5_block = classmethod(cls._build_res5_block)

        ret["res5"], out_channels = cls._build_res5_block(cfg)
        ret["box_predictor"] = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

        if mask_on:
            ret["mask_head"] = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
            )
        if glcc_on:
            ret['glcc'] = GLCC()  #glcc
        return ret

    @classmethod
    def _build_res5_block(cls, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = ResNet.make_stage(
            BottleneckBlock,
            3,
            stride_per_block=[2, 1, 1],
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features: List[torch.Tensor], boxes: List[Boxes]):
        x = self.pooler(features, boxes)
        return self.res5(x)

    # Head
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        fish_classes: torch.Tensor = None,   #glcc
    ):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]  # List[Boxes], 1000 boxes 
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )  # Tensor, (1000, 2048, 7, 7 )
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))  # Tuple[ scores:Tensor (1000,3), proposal_deltas:Tensor (1000,8) ]

        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)

            if self.glcc_on:
                pred_instances, roi_idxs = self.box_predictor.inference(predictions, proposals)  # Boxes
                pred_fishes, check_results = self.glcc( pred_instances, roi_idxs, box_features )
                if any( check_results ):
                    losses.update( loss_glcc = cross_entropy(pred_fishes, fish_classes[check_results] ) )
                else:
                    breakpoint()
            #if self.mask_on:
                #proposals, fg_selection_masks = select_foreground_proposals(
                #    proposals, self.num_classes
                #)
                # Since the ROI feature transform is shared bvetween boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                #mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                #del box_features
                #losses.update(self.mask_head(mask_features, proposals))
            return [], losses
        else:
            pred_instances, roi_idxs = self.box_predictor.inference(predictions, proposals)  # Boxes
            #pred_instances = self.forward_with_given_boxes(features, pred_instances)  # Mask, keypoints
            if self.glcc_on:
                pred_fishes, check_results = self.glcc( pred_instances, roi_idxs, box_features )
                if torch.is_tensor( pred_fishes ):
                    pred_fishes = pred_fishes.argmax(dim=1)

            return pred_instances, pred_fishes
            #return pred_instances, {}

        
    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            feature_list = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(feature_list, [x.pred_boxes for x in instances])
            return self.mask_head(x, instances)
        else:
            return instances



class GLCC( nn.Module ):
    def __init__(self):
        super().__init__()
        self.glcc = nn.Sequential(
            nn.Conv2d(  4096, 2048, (1,1) ),  # (1,2048,7,7)
            nn.AdaptiveAvgPool2d( 1 ),        # (1,2048,1,1)
            nn.Flatten(),                     # (1,2048)
            nn.Linear(2048, 3)                # (1,3)
        )
        
    def forward(self, pred_instances, roi_idxs, box_features):
        feats = self._get_features( pred_instances, roi_idxs, box_features)
        check_results = [f is not None for f in feats]

        # Choose inputs
        if all( check_results ):
            x = torch.cat( feats, dim=0 )
        elif check_results[0] == True:
            x = feats[0]
        elif check_results[1] == True:
            x = feats[1]
        else:
            return None, check_results

        # Output
        y = self.glcc( x )
        return y, check_results

    
    def _get_features(self, pred_instances, roi_idxs, box_features) -> torch.Tensor:
        # roi_idxs: indices of regions sorted by scores
        
        # Get global / local features
        feats = []
        for instances, ridxs in zip( pred_instances, roi_idxs ):
            
            # Check-up for global / local instances
            classes = instances.pred_classes.to('cpu')
            scores = instances.scores
            is_local = 0 in classes
            is_global = 1 in classes
            
            # Get features
            local_feat = global_feat = None
            
            if is_local:
                idx = ridxs[ classes==0 ][0]   # Get the best scored Index
                local_feat = box_features[idx:idx+1]
                assert local_feat.shape == torch.Size([1,2048,7,7])
                
            if is_global:
                idx = ridxs[ classes==1 ][0]    # Index at roi
                global_feat = box_features[idx:idx+1]
                assert global_feat.shape == torch.Size([1,2048,7,7])

            # Concatenate global and local features
            if is_local and is_global:
                feats.append( torch.cat( (global_feat, local_feat), dim=1 ) )
            else:
                feats.append( None )

        return feats

        

if __name__=='__main__':
    x = torch.rand( (1,4096,7,7) )
    m1 = nn.Conv2d(  4096, 2048, (1,1) )
    m2 =  nn.AdaptiveAvgPool2d( 1 )
    m3 = nn.Flatten()
    m4 = nn.Linear(2048, 4)
    y1 = m1(x)
    y2 = m2(y1)
    y3 = m3(y2)
    y4 = m4(y3)
    
    # RCNN + GLCC
    #from detectron2.config import get_cfg
    #from detectron2.modeling import build_model

    #cfg = get_cfg()
    #cfg.merge_from_file("./conf/glcc.yaml")
    #model = build_model(cfg)
