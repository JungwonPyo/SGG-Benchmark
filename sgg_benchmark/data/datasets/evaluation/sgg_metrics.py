import torch
import torch.nn.functional as F
import numpy as np
from functools import reduce
from abc import ABC, abstractmethod
from sgg_benchmark.utils.miscellaneous import intersect_2d, argsort_desc, bbox_overlaps

class SceneGraphEvaluation(ABC):
    def __init__(self, result_dict):
        self.result_dict = result_dict
 
    @abstractmethod
    def register_container(self, mode): pass
    
    @abstractmethod
    def generate_print_string(self, mode): pass

    def calculate(self, global_container, local_container, mode): pass

    def collect_mean_recall_items(self, global_container, local_container, mode): pass

    def weight_function(self, position, k, mode="linear"):
        if k == 'relative': return 1.0
        if mode == "linear": return (k - position) / k
        if mode == "log": return np.log(k - position + 1) / np.log(k + 1)
        return 1.0

class SGRecall(SceneGraphEvaluation):
    def __init__(self, result_dict, name='R', key='recall', k_list=[20, 50, 100]):
        super().__init__(result_dict)
        self.name = name
        self.key = key
        self.k_list = k_list

    def register_container(self, mode):
        self.result_dict[f'{mode}_{self.key}'] = {k: [] for k in self.k_list}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[f'{mode}_{self.key}'].items():
            val = np.mean(v) if len(v) > 0 else 0.0
            result_str += f'    {self.name} @ {k}: {val:.4f}; '
        return result_str + f' for mode={mode}.\n'

    def calculate(self, global_container, local_container, mode):
        pred_to_gt = local_container.get('pred_to_gt', [])
        gt_rels = local_container.get('gt_rels', [])
        for k in self.k_list:
            match = reduce(np.union1d, pred_to_gt[:k]) if len(pred_to_gt) > 0 else []
            rec_i = float(len(match)) / float(gt_rels.shape[0]) if gt_rels.shape[0] > 0 else 0.0
            self.result_dict[f'{mode}_{self.key}'][k].append(rec_i)

class SGMeanRecall(SceneGraphEvaluation):
    def __init__(self, result_dict, num_rel, ind_to_predicates, print_detail=True, name='mR', key='mean_recall', k_list=[20, 50, 100]):
        super().__init__(result_dict)
        self.num_rel = num_rel
        self.print_detail = print_detail
        self.name = name
        self.key = key
        self.k_list = k_list
        self.rel_name_list = [ind_to_predicates[i] for i in sorted(ind_to_predicates.keys()) if i > 0] if isinstance(ind_to_predicates, dict) else ind_to_predicates[1:]

    def register_container(self, mode):
        self.result_dict[f'{mode}_{self.key}'] = {k: 0.0 for k in self.k_list}
        self.result_dict[f'{mode}_{self.key}_collect'] = {k: [[] for _ in range(self.num_rel)] for k in self.k_list}
        self.result_dict[f'{mode}_{self.key}_list'] = {k: [] for k in self.k_list}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[f'{mode}_{self.key}'].items():
            result_str += f'   {self.name} @ {k}: {float(v):.4f}; '
        result_str += f' for mode={mode}.\n'
        if self.print_detail:
            result_str += '----------------------- Details ------------------------\n'
            for n, r in zip(self.rel_name_list, self.result_dict[f'{mode}_{self.key}_list'][max(self.k_list)]):
                result_str += f'({n}:{r:.4f}) '
            result_str += '\n--------------------------------------------------------\n'
        return result_str

    def collect_mean_recall_items(self, global_container, local_container, mode):
        pred_to_gt = local_container.get('pred_to_gt', [])
        gt_rels = local_container.get('gt_rels', [])
        for k in self.k_list:
            match = reduce(np.union1d, pred_to_gt[:k]) if len(pred_to_gt) > 0 else []
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel
            for idx in range(gt_rels.shape[0]):
                recall_count[int(gt_rels[idx, 2])] += 1
            for idx in range(len(match)):
                recall_hit[int(gt_rels[int(match[idx]), 2])] += 1
            for n in range(1, self.num_rel):
                if recall_count[n] > 0:
                    self.result_dict[f'{mode}_{self.key}_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))

    def calculate(self, global_container, local_container, mode):
        for k in self.k_list:
            recalls = [np.mean(self.result_dict[f'{mode}_{self.key}_collect'][k][i]) if self.result_dict[f'{mode}_{self.key}_collect'][k][i] else 0.0 for i in range(1, self.num_rel)]
            self.result_dict[f'{mode}_{self.key}_list'][k] = recalls
            self.result_dict[f'{mode}_{self.key}'][k] = np.mean(recalls) if recalls else 0.0

class SGF1Score(SceneGraphEvaluation):
    def register_container(self, mode):
        self.result_dict[f'{mode}_f1_score'] = {20: 0.0, 50: 0.0, 100: 0.0}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[f'{mode}_f1_score'].items():
            result_str += f'    F1 @ {k}: {v:.4f}; '
        return result_str + f' for mode={mode}.\n'

    def calculate(self, global_container, local_container, mode):
        res = self.result_dict
        for k in res[f'{mode}_f1_score']:
            r_list = res.get(f'{mode}_recall', {}).get(k, [])
            r = np.mean(r_list) if r_list else 0.0
            mr = res.get(f'{mode}_mean_recall', {}).get(k, 0.0)
            res[f'{mode}_f1_score'][k] = 2 * r * mr / (r + mr) if (r + mr) > 0 else 0.0

class SGNoGraphConstraintRecall(SGRecall):
    def __init__(self, result_dict):
        super().__init__(result_dict, name='ng-R', key='recall_nogc')
    def calculate(self, global_container, local_container, mode):
        old_pred_to_gt = local_container.get('pred_to_gt')
        local_container['pred_to_gt'] = local_container.get('nogc_pred_to_gt', [])
        super().calculate(global_container, local_container, mode)
        local_container['pred_to_gt'] = old_pred_to_gt

class SGZeroShotRecall(SGRecall):
    def __init__(self, result_dict, seen_triplet_set=None):
        super().__init__(result_dict, name='zR', key='zeroshot_recall')
        self.seen_triplet_set = seen_triplet_set
    def prepare_zeroshot(self, global_container, local_container):
        gt_rels, gt_classes = local_container['gt_rels'], local_container['gt_classes']
        gt_triplets = np.column_stack((gt_classes[gt_rels[:,0]], gt_classes[gt_rels[:,1]], gt_rels[:,2]))
        if self.seen_triplet_set is not None:
            self.zeroshot_idx = [i for i, t in enumerate(gt_triplets) if tuple(t) not in self.seen_triplet_set]
        else:
            zeroshot_triplets = global_container.get('zeroshot_triplet', np.array([]))
            self.zeroshot_idx = np.where(intersect_2d(gt_triplets, zeroshot_triplets).sum(-1) > 0)[0].tolist() if len(zeroshot_triplets) > 0 else []
    def calculate(self, global_container, local_container, mode):
        pred_to_gt = local_container.get('pred_to_gt', [])
        if not self.zeroshot_idx:
            return
        for k in self.k_list:
            match = reduce(np.union1d, pred_to_gt[:k]) if len(pred_to_gt) > 0 else []
            hits = len(set(match) & set(self.zeroshot_idx))
            self.result_dict[f'{mode}_{self.key}'][k].append(float(hits) / len(self.zeroshot_idx))

class SGPairAccuracy(SceneGraphEvaluation):
    def register_container(self, mode):
        self.result_dict[f'{mode}_accuracy_hit'] = {k: [] for k in [20, 50, 100]}
        self.result_dict[f'{mode}_accuracy_count'] = {k: [] for k in [20, 50, 100]}
    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k in self.result_dict[f'{mode}_accuracy_hit']:
            hit, count = np.mean(self.result_dict[f'{mode}_accuracy_hit'][k]), np.mean(self.result_dict[f'{mode}_accuracy_count'][k])
            result_str += f'    A @ {k}: {hit/count:.4f}; ' if count > 0 else f'    A @ {k}: 0.0000; '
        return result_str + f' for mode={mode}.\n'
    def prepare_gtpair(self, local_container):
        pred_pair_idx = local_container['pred_rel_inds'][:, 0] * 1024 + local_container['pred_rel_inds'][:, 1]
        gt_pair_idx = local_container['gt_rels'][:, 0] * 1024 + local_container['gt_rels'][:, 1]
        self.pred_pair_in_gt = (pred_pair_idx[:, None] == gt_pair_idx[None, :]).any(-1)
    def calculate(self, global_container, local_container, mode):
        if mode == 'sgdet': return
        pred_to_gt = local_container.get('pred_to_gt', [])
        for k in [20, 50, 100]:
            gt_pair_matches = [p for p, flag in zip(pred_to_gt[:k], self.pred_pair_in_gt[:k]) if flag]
            match = reduce(np.union1d, gt_pair_matches) if gt_pair_matches else []
            self.result_dict[f'{mode}_accuracy_hit'][k].append(float(len(match)))
            self.result_dict[f'{mode}_accuracy_count'][k].append(float(len(local_container['gt_rels'])))

class SGNGZeroShotRecall(SGZeroShotRecall):
    def __init__(self, result_dict, seen_triplet_set=None): super().__init__(result_dict, seen_triplet_set=seen_triplet_set); self.name, self.key = 'ng-zR', 'ng_zeroshot_recall'
    def calculate(self, global_container, local_container, mode):
        old_pred_to_gt = local_container.get('pred_to_gt')
        local_container['pred_to_gt'] = local_container.get('nogc_pred_to_gt', [])
        super().calculate(global_container, local_container, mode)
        local_container['pred_to_gt'] = old_pred_to_gt

class SGRecallRelative(SGRecall):
    def __init__(self, result_dict): super().__init__(result_dict, name='R-rel', key='recall_relative', k_list=['relative'])
    def calculate(self, global_container, local_container, mode):
        old_pred_to_gt = local_container.get('pred_to_gt')
        local_container['pred_to_gt'] = old_pred_to_gt[:len(local_container['gt_rels'])] if old_pred_to_gt is not None else []
        super().calculate(global_container, local_container, mode)
        local_container['pred_to_gt'] = old_pred_to_gt

class SGMeanRecallRelative(SGMeanRecall):
    def __init__(self, result_dict, num_rel, ind_to_predicates, print_detail=True):
        super().__init__(result_dict, num_rel, ind_to_predicates, print_detail, name='mR-rel', key='mean_recall_relative', k_list=['relative'])
    def collect_mean_recall_items(self, global_container, local_container, mode):
        old_pred_to_gt = local_container.get('pred_to_gt')
        local_container['pred_to_gt'] = old_pred_to_gt[:len(local_container['gt_rels'])] if old_pred_to_gt is not None else []
        super().collect_mean_recall_items(global_container, local_container, mode)
        local_container['pred_to_gt'] = old_pred_to_gt

def _triplet(relations, classes, boxes, predicate_scores=None, class_scores=None):
    sub_id, ob_id, pred_label = relations[:, 0], relations[:, 1], relations[:, 2]
    triplets = np.column_stack((classes[sub_id], pred_label, classes[ob_id]))
    triplet_boxes = np.column_stack((boxes[sub_id], boxes[ob_id]))
    triplet_scores = np.column_stack((class_scores[sub_id], predicate_scores, class_scores[ob_id])) if predicate_scores is not None else None
    return triplets, triplet_boxes, triplet_scores

def _compute_pred_matches(gt_triplets, pred_triplets, gt_boxes, pred_boxes, iou_thres):
    keeps = intersect_2d(gt_triplets, pred_triplets)
    pred_to_gt = [[] for _ in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(keeps.any(1))[0], gt_boxes[keeps.any(1)], keeps[keeps.any(1)]):
        boxes = pred_boxes[keep_inds]
        inds = (bbox_overlaps(gt_box[None,:4], boxes[:,:4])[0] >= iou_thres) & (bbox_overlaps(gt_box[None,4:], boxes[:,4:])[0] >= iou_thres)
        for i in np.where(keep_inds)[0][inds]: pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt


class SGWeightedRecall(SGRecall):
    def __init__(self, result_dict): super().__init__(result_dict, name='wR', key='weighted_recall')
    def calculate(self, global_container, local_container, mode):
        pred_to_gt, gt_rels = local_container.get('pred_to_gt', []), local_container.get('gt_rels', [])
        for k in self.k_list:
            indices = [i for i, x in enumerate(pred_to_gt[:k]) if x]
            weighted_sum = sum(self.weight_function(idx-i, k) for i, idx in enumerate(indices))
            self.result_dict[f'{mode}_{self.key}'][k].append(weighted_sum / len(gt_rels) if len(gt_rels) > 0 else 0.0)

class SGWeightedMeanRecall(SGMeanRecall):
    def __init__(self, result_dict, num_rel, ind_to_predicates, print_detail=True):
        super().__init__(result_dict, num_rel, ind_to_predicates, print_detail, name='wmR', key='weighted_mean_recall')
    def collect_mean_recall_items(self, global_container, local_container, mode):
        pred_to_gt, gt_rels = local_container.get('pred_to_gt', []), local_container.get('gt_rels', [])
        for k in self.k_list:
            match = reduce(np.union1d, pred_to_gt[:k]) if len(pred_to_gt) > 0 else []
            recall_hit, recall_count = [0] * self.num_rel, [0] * self.num_rel
            for idx in range(gt_rels.shape[0]): recall_count[int(gt_rels[idx, 2])] += 1
            for i, m in enumerate(match):
                weight = self.weight_function(i, k)
                recall_hit[int(gt_rels[int(m), 2])] += weight
            for n in range(1, self.num_rel):
                if recall_count[n] > 0: self.result_dict[f'{mode}_{self.key}_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))

class SGInformativeRecall(SceneGraphEvaluation):
    def __init__(self, result_dict, sim='mpnet'):
        super().__init__(result_dict)
        self.sim = sim
        self._model = None
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            models = {'glove': 'average_word_embeddings_glove.6B.300d', 'uae_large': 'WhereIsAI/UAE-Large-V1', 
                      'bert_large': 'bert-large-nli-mean-tokens', 'minilm': 'all-MiniLM-L6-v2', 
                      'mpnet': 'all-mpnet-base-v2', 'clip': 'CLIP-ViT-B-32'}
            self._model = SentenceTransformer(models.get(self.sim, models['mpnet']))
        return self._model
    def register_container(self, mode): self.result_dict[f'{mode}_informative_recall'] = {k: [] for k in [5, 10, 20, 50, 100]}
    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[f'{mode}_informative_recall'].items():
            result_str += f'    IR @ {k}: {np.mean(v):.4f}; '
        return result_str + f' for mode={mode}.\n'
    def calculate(self, global_container, local_container, mode):
        from sentence_transformers import util
        gt_inf = local_container.get('informative_rels', [])
        if not gt_inf: return
        pred_rels = np.column_stack((local_container['pred_rel_inds'], 1 + local_container['rel_scores'][:, 1:].argmax(1)))
        pred_triplets, _, _ = _triplet(pred_rels, local_container['pred_classes'], local_container['pred_boxes'], local_container['rel_scores'][:, 1:].max(1), local_container['obj_scores'])
        pred_str = [f"{global_container['ind_to_classes'][t[0]]} {global_container['ind_to_predicates'][t[1]]} {global_container['ind_to_classes'][t[2]]}" for t in pred_triplets]
        
        gt_emb = self.model.encode(gt_inf, batch_size=256, device='cuda')
        pred_emb = self.model.encode(pred_str, batch_size=256, device='cuda')
        cos_sim = util.cos_sim(pred_emb, gt_emb)
        pred_to_gt = [torch.where(row > 0.8)[0].tolist() for row in cos_sim]

        for k in self.result_dict[f'{mode}_informative_recall']:
            match = reduce(np.union1d, pred_to_gt[:k]) if any(pred_to_gt[:k]) else []
            self.result_dict[f'{mode}_informative_recall'][k].append(len(match) / len(gt_inf) if gt_inf else 0.0)

