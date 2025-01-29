import time
import webdataset as wds
from torch.utils.data import DataLoader, Dataset


class Data_set(Dataset):
    def __init__(self, folder, max_len):
        self.folder = folder
        self.max_len = max_len

    def __len__(self):
        return len(self.data_path)


def prepare_dataloaders(logging, configs):
    train_url = configs.train_settings.data_path
    val_url = configs.valid_settings.data_path
    logging.info(f"train url:{train_url}")
    logging.info(f"val_url url:{val_url}")
    time1 = time.time()
    logging.info("start loading tar files")
    if hasattr(configs.train_settings,"shuffle_dataset"):
           train_dataset = (wds.WebDataset(train_url).shuffle(int(configs.train_settings.shuffle_dataset)).decode().to_tuple("npz"))
    else:
       train_dataset = (wds.WebDataset(train_url).shuffle(2000).decode().to_tuple("npz"))
    
    val_dataset = (wds.WebDataset(val_url).decode().to_tuple("npz"))

    logging.info('build datasets')
    train_dataset = train_dataset.batched(configs.train_settings.batch_size)
    val_dataset = val_dataset.batched(configs.train_settings.batch_size)
    time2 = time.time()

    logging.info("done with loading tar files")
    logging.info(f"time to load train and validation sets: {time2 - time1}")

    train_loader = wds.WebLoader(train_dataset,
                                 batch_size=None, shuffle=False, pin_memory=False)
    val_loader = wds.WebLoader(val_dataset, batch_size=None,
                               shuffle=False, pin_memory=False)

    logging.info("create data loaders")
    if hasattr(configs.train_settings,"shuffle_loader"):
       train_loader = train_loader.unbatched().shuffle(int(configs.train_settings.shuffle_loader)).batched(1, partial=False)
    else:
       train_loader = train_loader.unbatched().shuffle(1000).batched(1, partial=False)
    val_loader = val_loader.unbatched().batched(configs.train_settings.batch_size, partial=False)

    return train_loader, val_loader


if __name__ == '__main__':
    print('test')

