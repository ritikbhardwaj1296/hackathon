import numpy as np
from scipy.optimize import linear_sum_assignment
# bboxes are ground truth bounding boxes (data from prompt), candidate_bbox are bounding boxes from YOLO detector

def convert_to_xywh(coordinates):
    """
    Convert [xmin, ymin, xmax, ymax] to [xmin, ymin, width, height].

    Parameters:
    coordinates (list): A list containing the coordinates in [xmin, ymin, xmax, ymax] format.

    Returns:
    list: The converted coordinates in [xmin, ymin, width, height] format.
    """
    xmin, ymin, xmax, ymax = coordinates
    width = xmax - xmin
    height = ymax - ymin
    return [xmin, ymin, width, height]



def iouloss(bboxes, candidate_bbox):
    bbox = np.array(convert_to_xywh(bboxes))
    candidates = []
    # for i in bboxes:
    #     bb = convert_to_xywh(i)
    #     bbox.append(bb)
    for j in candidate_bbox:
        bbc = convert_to_xywh(j)
        candidates.append(bbc)
    # bbox = np.array(bbox)
    candidates = np.array(candidates)

    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def calculate_loss(mat, CONFIDENCE_THRESHOLD):
    x, y = mat.shape[0], mat.shape[1]
        
    if x == y:
        loss_mat = 1 - mat
        row_ind, col_ind = linear_sum_assignment(loss_mat)
        tot_loss = loss_mat[row_ind, col_ind].sum()

    else:
        m_list = []
        t_ls = 0
        for i in range(x):
            a = max(mat(i))
            m_list.append(a)
        for i in range(x):
            a1 = max(m_list)
            m_list.remove(a1)
            t_ls += a1
        tot_loss = x - t_ls

    return tot_loss




def total_iouloss(detections_all, ground_truths):
    total_iou_loss = 0
    b_size = 0
    for i, j in zip(detections_all, ground_truths):
        step_ioul = 0
        subloc_iouloss = {}
        # get_var = ""
        for i1 in i:
            cand_box = []
            for j1 in j:
                if i1[0] == j1[0]:
                    cand_box.append(j1[1])
            loc_iouloss = iouloss(i1[1], cand_box)
            if i1[0] not in subloc_iouloss.keys(): 
                subloc_iouloss[i1[0]] = [loc_iouloss]
            else:
                subloc_iouloss[i1[0]].append(loc_iouloss)
        

        for k, val in subloc_iouloss.items():
            ls = calculate_loss(val)
            step_ioul += ls
        
        total_iou_loss += step_ioul
        b_size += 1

    final_loss = total_iou_loss / b_size

    return final_loss




