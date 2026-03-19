import os, torch, numpy as np, json
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from .sgg_metrics import _compute_pred_matches, _triplet, SGRecall, SGNoGraphConstraintRecall, SGZeroShotRecall, SGNGZeroShotRecall, SGMeanRecall, SGRecallRelative, SGMeanRecallRelative, SGF1Score, SGPairAccuracy, SGWeightedRecall, SGWeightedMeanRecall, SGInformativeRecall
from sgg_benchmark.config.paths_catalog import DatasetCatalog
from sgg_benchmark.structures.box_ops import box_convert
from sgg_benchmark.utils.miscellaneous import argsort_desc

def do_sgg_evaluation(cfg, dataset, dataset_name, predictions, output_folder, logger, iou_types, informative=False, **kwargs):
    metrics_to_eval = {'relations': ['recall', 'mean_recall', 'f1_score', 'zeroshot_recall'], 'bbox': ['mAP']}
    if cfg.test.informative: metrics_to_eval['relations'].append('informative_recall')

    metrics_map = {
        'recall': SGRecall, 'recall_nogc': SGNoGraphConstraintRecall, 'zeroshot_recall': SGZeroShotRecall,
        'ng_zeroshot_recall': SGNGZeroShotRecall, 'mean_recall': SGMeanRecall, 
        'recall_relative': SGRecallRelative, 'mean_recall_relative': SGMeanRecallRelative, 
        'f1_score': SGF1Score, 'pair_accuracy': SGPairAccuracy,
        'weighted_recall': SGWeightedRecall, 'weighted_mean_recall': SGWeightedMeanRecall,
        'informative_recall': SGInformativeRecall
    }

    metrics_to_eval = {k: metrics_map[k] for k in metrics_to_eval['relations'] if k in metrics_map}

    # Identify seen triplets from training statistics for dynamic zero-shot calculation
    seen_triplet_set = None
    try:
        from sgg_benchmark.data.build import get_dataset_statistics
        stats = get_dataset_statistics(cfg)
        if 'fg_matrix' in stats:
            fg_matrix = stats['fg_matrix']
            if torch.is_tensor(fg_matrix): fg_matrix = fg_matrix.cpu().numpy()
            seen_indices = np.where(fg_matrix > 0)
            seen_triplet_set = set(zip(seen_indices[0], seen_indices[1], seen_indices[2]))
            logger.info(f"Dynamically loaded {len(seen_triplet_set)} seen triplets from training statistics for zero-shot evaluation.")
    except Exception as e:
        logger.warning(f"Could not compute seen triplets from training statistics: {e}. Falling back to zeroshot_file if available.")

    # Zeroshot triplets fallback
    data_dir = DatasetCatalog.DATA_DIR
    try:
        zero_shot_file = DatasetCatalog.DATASETS[dataset_name].get('zeroshot_file')
        zeroshot_triplet = torch.load(os.path.join(data_dir, zero_shot_file), map_location="cpu").long().numpy() if zero_shot_file else np.array([])
    except: zeroshot_triplet = np.array([])

    mode = 'predcls' if cfg.model.roi_relation_head.use_gt_box and cfg.model.roi_relation_head.use_gt_object_label else \
           'sgcls' if cfg.model.roi_relation_head.use_gt_box else 'sgdet'
    
    num_rel_category = cfg.model.roi_relation_head.num_classes
    multiple_preds, iou_thres = cfg.test.relation.multiple_preds, cfg.test.relation.iou_threshold

    groundtruths = [dataset.get_groundtruth(i, evaluation=True) for i in range(len(predictions))]
    
    result_str = '\n' + '=' * 100 + '\n'
    if "bbox" in iou_types:
        mAp = evaluate_detection(dataset, predictions, groundtruths, mode, logger)
        result_str += f'Detection evaluation mAp={mAp:.4f}\n' + '=' * 100 + '\n'

    if "relations" in iou_types:
        result_dict = {}
        evaluators = {}
        for k, v in metrics_to_eval.items():
            if "zeroshot" in k:
                evaluators["eval_" + k] = v(result_dict, seen_triplet_set=seen_triplet_set)
            else:
                evaluators["eval_" + k] = v(result_dict, num_rel_category, dataset.ind_to_predicates) if "mean" in k else v(result_dict)
            evaluators["eval_" + k].register_container(mode)

        global_container = {
            'zeroshot_triplet': zeroshot_triplet, 'result_dict': result_dict, 'mode': mode,
            'multiple_preds': multiple_preds, 'num_rel_category': num_rel_category, 'iou_thres': iou_thres
        }
        if informative:
            stats = dataset.get_statistics()
            global_container.update({'ind_to_predicates': stats['rel_classes'], 'ind_to_classes': stats['obj_classes']})

        for gt, pred in tqdm(zip(groundtruths, predictions), desc='SGG Eval', total=len(groundtruths)):
            evaluate_relation_of_one_image(gt, pred, global_container, evaluators, informative)

        for k, v in evaluators.items():
            if "mean" in k: v.calculate(global_container, None, mode)
            if "f1" not in k: result_str += v.generate_print_string(mode)

        if "eval_f1_score" in evaluators:
            evaluators['eval_f1_score'].calculate(result_dict, None, mode)
            result_str += evaluators['eval_f1_score'].generate_print_string(mode)
        result_str += '=' * 100 + '\n'

    logger.info(result_str)
    if output_folder:
        save_results(output_folder, result_dict, dataset_name, mode, cfg)
    return result_dict if "relations" in iou_types else {}

def evaluate_detection(dataset, predictions, groundtruths, mode, logger):
    anns = []
    for image_id, gt in enumerate(groundtruths):
        for cls, box in zip(gt['labels'].tolist(), gt['boxes'].tolist()):
            anns.append({'area': (box[3]-box[1])*(box[2]-box[0]), 'bbox': [box[0], box[1], box[2]-box[0], box[3]-box[1]],
                         'category_id': cls, 'id': len(anns), 'image_id': image_id, 'iscrowd': 0})
    fauxcoco = COCO(); fauxcoco.dataset = {'info': {}, 'images': [{'id': i} for i in range(len(groundtruths))],
                                           'categories': [{'id': i, 'name': n} for i, n in enumerate(dataset.ind_to_classes) if n != '__background__'],
                                           'annotations': anns}; fauxcoco.createIndex()
    cocolike = []
    for image_id, prediction in enumerate(predictions):
        box = box_convert(prediction["boxes"], prediction["mode"], "xywh").cpu().numpy()
        score = prediction['pred_scores'].cpu().numpy()
        label = prediction['pred_labels'].cpu().numpy()
        if mode == 'predcls':
            label, score = prediction['labels'].cpu().numpy(), np.ones(len(label))
        if len(box) > 0: cocolike.append(np.column_stack((np.full(len(box), image_id), box, score, label)))
    
    if not cocolike: return 0.0
    res = fauxcoco.loadRes(np.concatenate(cocolike, 0))
    coco_eval = COCOeval(fauxcoco, res, 'bbox')
    coco_eval.params.imgIds = list(range(len(groundtruths)))
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    return coco_eval.stats[1]

def evaluate_relation_of_one_image(gt, pred, global_container, evaluators, informative=False):
    mode, iou_thres = global_container['mode'], global_container['iou_thres']
    gt_rels = gt['relation_tuple'].long().cpu().numpy()
    if len(gt_rels) == 0: return

    gt_boxes = box_convert(gt["boxes"], gt["mode"], "xyxy").cpu().numpy()
    gt_classes = gt['labels'].long().cpu().numpy()
    pred_rel_inds = pred['rel_pair_idxs'].long().cpu().numpy()
    rel_scores = pred['pred_rel_scores'].cpu().numpy()
    pred_boxes = box_convert(pred["boxes"], pred["mode"], "xyxy").cpu().numpy()
    pred_classes = pred['pred_labels'].long().cpu().numpy()
    obj_scores = pred['pred_scores'].cpu().numpy()

    if mode == 'predcls':
        pred_boxes, pred_classes, obj_scores = gt_boxes, gt_classes, np.ones(len(gt_classes))

    local_container = {'gt_rels': gt_rels, 'gt_boxes': gt_boxes, 'gt_classes': gt_classes,
                       'pred_rel_inds': pred_rel_inds, 'rel_scores': rel_scores,
                       'pred_boxes': pred_boxes, 'pred_classes': pred_classes, 'obj_scores': obj_scores}
    if informative: local_container['informative_rels'] = gt.get('informative_rels')

    pred_rels = np.column_stack((pred_rel_inds, 1 + rel_scores[:, 1:].argmax(1)))
    pred_scores = rel_scores[:, 1:].max(1)
    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
    pred_triplets, pred_triplet_boxes, _ = _triplet(pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)

    local_container['pred_to_gt'] = _compute_pred_matches(gt_triplets, pred_triplets, gt_triplet_boxes, pred_triplet_boxes, iou_thres)

    if any(k in evaluators for k in ['eval_recall_nogc', 'eval_ng_zeroshot_recall']):
        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        nogc_scores = (obj_scores_per_rel[:, None] * rel_scores[:, 1:])
        nogc_inds = argsort_desc(nogc_scores)[:100]
        nogc_pred_rels = np.column_stack((pred_rel_inds[nogc_inds[:, 0]], nogc_inds[:, 1] + 1))
        nogc_pred_triplets, nogc_pred_triplet_boxes, _ = _triplet(nogc_pred_rels, pred_classes, pred_boxes, rel_scores[nogc_inds[:, 0], nogc_inds[:, 1]+1], obj_scores)
        local_container['nogc_pred_to_gt'] = _compute_pred_matches(gt_triplets, nogc_pred_triplets, gt_triplet_boxes, nogc_pred_triplet_boxes, iou_thres)

    for k, v in evaluators.items():
        if "zeroshot" in k: v.prepare_zeroshot(global_container, local_container)
        if "pair_accuracy" in k: v.prepare_gtpair(local_container)
        if "mean" in k: v.collect_mean_recall_items(global_container, local_container, mode)
        elif "f1" not in k: v.calculate(global_container, local_container, mode)

def save_results(output_folder, result_dict, dataset_name, mode, cfg):
    out_file = os.path.join(output_folder, f'eval_results_top_{cfg.test.top_k}.json')
    with open(out_file, 'w') as f: json.dump(result_dict, f)
    if "test" in dataset_name:
        res_file = os.path.join(output_folder, 'results.json')
        if os.path.exists(res_file):
            with open(res_file, 'r') as f: res = json.load(f)
            for k in [20, 50, 100]:
                for m in ['recall', 'mean_recall', 'f1_score']:
                    if f'{mode}_{m}' in result_dict:
                        val = result_dict[f'{mode}_{m}'][k]
                        res[f'{mode}_{m}@{k}'] = float(np.mean(val)) if isinstance(val, list) else float(val)
            with open(res_file, 'w') as f: json.dump(res, f)

