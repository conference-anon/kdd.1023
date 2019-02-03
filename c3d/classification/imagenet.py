import os

from c3d.classification.dataset import ImageDataset


class TinyImageNetDataset(ImageDataset):
    def __init__(self, *args, class_subset=None, **kwargs):
        super(ImageDataset, self).__init__(*args, **kwargs)
        if not all(os.path.isdir(self.get_dir(role))
                   for role in ('train', 'test', 'val')):
            raise ValueError('Not a valid (Tiny-)ImageNet directory!')

        annotations_path = os.path.join(self.data_root, 'val',
                                        'val_annotations.txt')

        if not os.path.isfile(annotations_path):
            raise ValueError("Can't find the validation annotations file!")

        self.validation_labels = {}
        with open(annotations_path, 'r') as f:
            for line in f:
                parts = line.split('\t')
                self.validation_labels[parts[0]] = parts[1]
        self.class_subset = class_subset

    def get_dir(self, role):
        return os.path.join(self.data_root, role)

    def is_train_file(self, file_id):
        return file_id.id[0] == 'train'

    def is_test_file(self, file_id):
        return file_id.id[0] == 'val'

    def label_from_path(self, dirpath, filename):
        rel_dir = os.path.relpath(dirpath, self.data_root)
        parts = tuple(rel_dir.split(os.sep))
        if parts[-1] == 'images':
            result = parts[-2]
        else:
            result = parts[-1]
        if result == 'val':
            result = self.validation_labels[filename]
        return result

    def id_from_path(self, dirpath, filename):
        rel_dir = os.path.relpath(dirpath, self.data_root)
        return tuple(rel_dir.split(os.sep)) + (filename,)

    def file_name(self, file_id):
        return file_id.id[-1]

    def file_dir(self, file_id):
        return os.path.join(*file_id.id[:-1])

    def file_id_filter(self, file_id):
        if not file_id.id[0] in ('train', 'val'):
            return False
        if self.class_subset is not None:
            if file_id.label not in self.class_subset:
                return False
        return True
