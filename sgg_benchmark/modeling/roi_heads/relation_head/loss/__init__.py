from .loss import RelationLossComputation
from .hierarchical_loss import RelationHierarchicalLossComputation

def make_roi_relation_loss_evaluator(cfg, pred_prop, pred_weight):

    if "Hierarchical" in cfg.model.roi_relation_head.predictor:
        loss_evaluator = RelationHierarchicalLossComputation(
            cfg.model.attribute_on,
            cfg.model.roi_attribute_head.num_attributes,
            cfg.model.roi_attribute_head.max_attributes,
            cfg.model.roi_attribute_head.attribute_bgfg_sample,
            cfg.model.roi_attribute_head.attribute_bgfg_ratio,
            cfg.model.roi_relation_head.label_smoothing_loss,
            pred_prop,
        )
    else:
        from sgg_benchmark.data import get_dataset_statistics
        statistics = get_dataset_statistics(cfg, False)
        loss_evaluator = RelationLossComputation(
            cfg,
            cfg.model.attribute_on,
            cfg.model.roi_attribute_head.num_attributes,
            cfg.model.roi_attribute_head.max_attributes,
            cfg.model.roi_attribute_head.attribute_bgfg_sample,
            cfg.model.roi_attribute_head.attribute_bgfg_ratio,
            cfg.model.roi_relation_head.loss,
            pred_weight,
            statistics,
        )

    return loss_evaluator