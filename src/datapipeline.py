from torch.utils.data import Dataset, DataLoader
import jsonlines
import easydict

class CCPMDataset(Dataset):
    def __init__(self, path, mode='train', add_prompts = False):
        self.path = path
        self.mode = mode
        self.add_prompts = add_prompts
        self.prompts = ['一:', '二:', '三:', '四:']
        self.data = self.load_data()
    
    def load_data(self) -> list:
        # load data
        data = []
        with open(self.path, 'r+', encoding='utf-8') as f:
            for item in jsonlines.Reader(f):
                dataline = easydict.EasyDict(item)
                if 'answer' in dataline:
                    dataline['answer'] = int(dataline['answer'])
                if self.add_prompts:
                    for i in range(len(dataline['choices'])):
                        dataline['choices'][i] = self.prompts[i] + dataline['choices'][i]
                data.append(dataline)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.mode == 'train' or self.mode == 'valid':
            return data['translation'], data['choices'], data['answer']
        elif self.mode == 'test':
            return data['translation'], data['choices']
        else:
            raise ValueError('mode should be train/valid/test')