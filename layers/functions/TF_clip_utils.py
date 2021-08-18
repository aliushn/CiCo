import torch


def UnFold_candidate_clip(candidate_clip, remove_blank=False):
    '''
    UnFold candidate_clip to multi-frame candidates.
    For example, the size of 'box' in the candidates are [n1, 4], ..., [n_k, 4],
    the size of 'box' in the clip candidate should be [n1+..+nk, 4].
    :param candidates: [candidate_1, ..., candidate_k]
    :param remove_blank: whether remove those blank masks that are estimated from other frames
    :return: a clip candidate
    '''
    candidates = []
    T, N = candidate_clip['mask'].size()[:2]
    if remove_blank:
        # whether add some tracked masks
        cond1 = candidate_clip['tracked_mask'] <= 10
        # whether tracked masks are greater than a small threshold, which removes some false positives
        cond2 = candidate_clip['mask'].gt(0.5).sum(dim=(2, 3)) > 2
        # a declining weights (0.8) to remove some false positives that cased by consecutively track to segment
        cond3 = candidate_clip['score'].clone() > 0.05
        cond4 = candidate_clip['box_ids'].clone().squeeze(-1) > -1
        keep_clip = cond1 & cond2 & cond3 & cond4

    for i in range(T):
        candidate_cur = {}
        for k, v in candidate_clip.items():
            if remove_blank:
                candidate_cur[k] = v[i][keep_clip[i]].clone()
            else:
                candidate_cur[k] = v[i]
        candidates.append(candidate_cur)

    return candidates


def Fold_candidates(candidates, img_metas=None):
    '''
    Fold candidates of multi-frames to a clip candidate.
    For example, the size of 'box' in the candidates are [n1, 4], ..., [n_k, 4],
    the size of 'box' in the clip candidate should be [n1+..+nk, 4].
    :param candidates: [candidate_1, ..., candidate_k]
    :param img_metas: [img_meta1, ..., img_meat_k]
    :return: a clip candidate
    '''

    candidate_clip = {}
    for k in candidates[0].keys():
        candidate_clip[k] = []
    if img_metas is not None:
        candidate_clip['frame_id'] = []

    for i, candidate in enumerate(candidates):
        n_cur = candidate['box'].size(0)
        if n_cur > 0:
            for k, v in candidate.items():
                if k in {'fpn_feat', 'proto', 'track', 'sem_seg'}:
                    candidate_clip[k].append(v.repeat(n_cur, 1, 1, 1))
                elif k == 'mask':
                    candidate_clip[k].append(v)    # [n_cur, h, w]
                else:
                    candidate_clip[k].append(v.view(n_cur, -1))

            if img_metas is not None:
                candidate_clip['frame_id'].append(torch.tensor(img_metas[i]['frame_id']).expand(n_cur, 1))

    for k, v in candidate_clip.items():
        # at least one object has been detected in this clip
        candidate_clip[k] = torch.cat(v, dim=0)

    return candidate_clip


def Fold_candidates_by_order(candidates, track_by_Gaussian=False, img_metas=None):
    '''
    Fold candidates of multi-frames to a clip candidate.
    For example, the size of 'box' in the candidates are [n1, 4], ..., [n_k, 4],
    the size of 'box' in the clip candidate should be [n1+..+nk, 4].
    :param candidates: [candidate_1, ..., candidate_k], where instances in each candidate is sorted by order
    :param img_metas: [img_meta1, ..., img_meat_k]
    :return: a clip candidate
    '''

    n_dets = candidates[0]['box'].size(0)
    img_level_keys = {'proto', 'fpn_feat', 'sem_seg'}
    if track_by_Gaussian:
        img_level_keys += ['track']

    candidate_clip = dict()
    for k in candidates[0].keys():
        if k in img_level_keys:
            # [T, n_dets, h, w, -1]
            candidate_clip[k] = torch.stack([candidate[k] for candidate in candidates], dim=0).repeat(1, n_dets, 1, 1, 1)
        else:
            candidate_clip[k] = torch.stack([candidate[k] for candidate in candidates], dim=0)  # [T, n_cur, -1]

    if img_metas is not None:
        frame_id_clip = torch.tensor([img_meta['frame_id'] for img_meta in img_metas]).view(-1, 1, 1)
        candidate_clip['frame_id'] = frame_id_clip.repeat(1, n_dets, 1)  # [T, 1, 1]

    return candidate_clip


def select_distinct_track(candidate_clip):
    distinct_track = dict()
    for k, v in candidate_clip.items():
        distinct_track[k] = []

    n_frames, n_dets = candidate_clip['box'].size()[:2]
    if n_dets > 0:
        for i_obj in range(n_dets):
            _, sorted_idx = candidate_clip['score'][:, i_obj].sort(0, descending=True)
            for k, v in candidate_clip.items():
                if k not in {'track_mu', 'track_var'}:
                    distinct_track[k].append(v[sorted_idx[0], i_obj])
                else:
                    if k == 'track_mu':
                        distinct_track[k].append(v[:, i_obj].sum(dim=0) / n_frames)
                    else:
                        distinct_track[k].append(v[:, i_obj].sum(dim=0) / (n_frames**2))

        for k, v in distinct_track.items():
            distinct_track[k] = torch.stack(v, dim=0)

    return distinct_track
