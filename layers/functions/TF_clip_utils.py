import torch


def UnFold_candidate_clip(candidate_clip, remove_blank=True):
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
        keep_clip = candidate_clip['mask'].gt(0.5).sum(dim=(2, 3)) > 7

    for i in range(T):
        candidate_cur = {}
        for k, v in candidate_clip.items():
            if remove_blank:
                candidate_cur[k] = v[i][keep_clip[i]]
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


def Anchor_independented_info(candidates, img_metas=None):
    '''
    select the information of candidates which is independented with anchors, such as proto_data.
    This will be used to predict masks from time t to t+k
    :param candidates: [candidate_1, ..., candidate_k]
    :param img_metas: [img_meta1, ..., img_meat_k]
    :return: [anchor_independented_candidate1, ..., anchor_independented_candidate_k]
    '''
    # add frames_ids for every objects in the clip
    anchor_independented_clip = []
    for i, candidate in enumerate(candidates):
        if img_metas is not None:
            anchor_indepented_cur = {'frame_id': torch.tensor(img_metas[i]['frame_id']).view(1)}
        else:
            anchor_indepented_cur = {}

        for k, v in candidate.items():
            if k in {'fpn_feat', 'proto', 'track', 'sem_seg'}:
                anchor_indepented_cur[k] = v
        anchor_independented_clip.append(anchor_indepented_cur)

    return anchor_independented_clip