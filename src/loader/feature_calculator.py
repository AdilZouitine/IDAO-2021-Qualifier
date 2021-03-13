import torch
import torchvision.transforms as transforms

from loader import IdaoDataset

MEAN,STD,MIN,MAX = (-0.3918, -0.2711, -0.0477),(0.0822, 0.0840, 0.0836) #stats over all the dataset after center crop
MIN,MAX = 3.244370937347412 16.407089233398438  # extremum after normalize and centercrop

BASE_TRANSFORM = transforms.Compose(
    [
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(
            MEAN,STD
        ),
    ]
)


class FeaturesCalculator:
    def __init__(self, list_path: List[str], transform=BASE_TRANSFORM,loader):
        self.loader = loader(list_path=list_path, transform=BASE_TRANSFORM)

    def __len__(self) -> int:
        return len(self.list_path)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, float]:

        image, info1, info2 = self.loader[index]
        return compute_feature(image), info1, info2
    
    @classmethod
    def compute_feature(self,image):
        image = torch.flatten(image)
        mean,std,min,max,q1,q2,q3 = image.mean(),image.std(),image.min(),image.max(),torch.quantile(image, 0.05),torch.quantile(image, 0.25),torch.quantile(image, 0.5)
        hist = torch.histc(torch.tensor(image), bins=500, min=int(MIN), max=int(MAX))
        hist = hist.div(hist.sum())
        

