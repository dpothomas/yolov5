from pathlib import Path

import val
from utils.general import check_dataset


def test_map_calculation():
    data = "../data/testv3.yaml"
    weights = "../best.pt"
    batch_size = 32
    imgsz = 960  # inference size (pixels)
    print(f"data:{data}")
    data_dict = check_dataset(data)
    print(f"data_dict:{data_dict}")

    results, maps, t, extra_metrics, _, _ = val.run(
        data,
        weights=weights,  # model.pt path(s)
        batch_size=batch_size,  # batch size
        imgsz=imgsz,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task="test",  # train, val, test, speed or study
    )

    mp = results[0]
    mr = results[1]
    map50 = results[2]
    map_scaled = results[3]
    assert abs(mp - 0.9126989624197264) < 1e-6
    assert abs(mr - 0.852006056018168) < 1e-6
    assert abs(map50 - 0.9336614853256442) < 1e-6
    assert abs(map_scaled - 0.8044691826278816) < 1e-6
