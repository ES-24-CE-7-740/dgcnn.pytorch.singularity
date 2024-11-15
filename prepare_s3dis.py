from torch.utils.data import DataLoader
from data import S3DIS

train_loader = DataLoader(S3DIS(partition='train', num_points=4096, test_area='6'), 
                          num_workers=4, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(S3DIS(partition='test', num_points=4096, test_area='6'), 
                         num_workers=4, batch_size=16, shuffle=True, drop_last=False)
