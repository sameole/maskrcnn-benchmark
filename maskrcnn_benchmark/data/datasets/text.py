# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import cv2

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

# class ImageDataset(object):
#     def __init__(self, image_file):
#         self.image_file = open(image_file, 'rb')
#         self.lock=multiprocessing.Lock()
#
#     def get_image(self, offset, length):
#         with self.lock:
#             self.image_file.seek(offset)
#             image_byte=self.image_file.read(length)
#         nparr = np.fromstring(image_byte, np.uint8)
#         return cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

class TextDataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(TextDataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]

        # self.json_category_id_to_contiguous_id = {
        #     v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        # }
        # self.contiguous_category_id_to_json_id = {
        #     v: k for k, v in self.json_category_id_to_contiguous_id.items()
        # }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(TextDataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        #classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = [1 for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        file_name = '/disk3/lsxu/torch_all/images/' + str(img_data['id']) +'.jpg'
        img = cv2.imread(file_name)
        img_data['height'] = img.shape[0]
        img_data['width'] = img.shape[1]
        img_data['file_name'] = str(img_data['id']) +'.jpg'
        #print('img_data......',img_data)
        return img_data
