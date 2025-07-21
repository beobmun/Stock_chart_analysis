import pandas as pd
import numpy as np
import torch
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
from PIL import Image, ImageChops
import torchvision.transforms as transforms

class DataInfo:
    def __init__(self, info_path, data_path, random_state=42):
        self.info_path = info_path
        self.data_path = data_path
        self.random_state = random_state
        self.snp500_info_df = pd.DataFrame()
        self.snp500_symbols = list()
        self.train_symbols = list()
        self.val_symbols = list()
        self.test_symbpls = list()
        
    def load_info(self):
        try:
            self.snp500_info_df = pd.read_csv(self.info_path)
            self.snp500_symbols = self.snp500_info_df['Symbol'].tolist()
        except FileNotFoundError:
            print(f"File not found: {self.info_path}")
        return self
    
    def train_test_split(self, train_size=0.6, val=True, sector=True):
        if not self.snp500_symbols:
            print("S&P 500 symbols are empty. Please load S&P 500 info first.")
            return self
        if sector:
            self.train_symbols = set(self.snp500_info_df.groupby('GICS Sector').apply(
                lambda x: x.sample(frac=train_size, random_state=self.random_state)
            )['Symbol'].tolist())
            if val:
                temp_df = self.snp500_info_df[~self.snp500_info_df['Symbol'].isin(self.train_symbols)]
                self.val_symbols = set(temp_df.groupby('GICS Sector').apply(
                    lambda x: x.sample(frac=0.5, random_state=self.random_state)
                )['Symbol'].tolist())
                self.test_symbols = set(temp_df['Symbol'].tolist()) - self.val_symbols
            else:
                self.test_symbols = set(self.snp500_symbols) - self.train_symbols
        else:
            self.train_symbols = set(self.snp500_info_df.sample(frac=train_size, random_state=self.random_state)['Symbol'].tolist())
            if val:
                temp_df = self.snp500_info_df[~self.snp500_info_df['Symbol'].isin(self.train_symbols)]
                self.val_symbols = set(temp_df.sample(frac=0.5, random_state=self.random_state)['Symbol'].tolist())
                self.test_symbols = set(temp_df['Symbol'].tolist()) - self.val_symbols
            else:
                self.test_symbols = set(self.snp500_symbols) - self.train_symbols
        
        self.train_symbols = list(self.train_symbols)
        self.val_symbols = list(self.val_symbols)
        self.test_symbols = list(self.test_symbols)
        return self
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, num_days=20, transform=None):
        self.df = df
        self.num_days = num_days
        self.transform = transform
        
    def __len__(self):
        return len(self.df) - self.num_days + 1 - 2
    
    def _trim(self, img):
        bg = Image.new(img.mode, img.size, (255, 255, 255))
        diff = ImageChops.difference(img, bg)
        bbox = diff.getbbox()
        return img.crop(bbox) if bbox else img
    
    def _generate_candle_chart(self, df, start):
        up_color = "#137901"
        down_color = "#d51217"
        marketcolor = mpf.make_marketcolors(up=up_color, down=down_color, 
                                            edge={'up': up_color, 'down': down_color},
                                            wick={'up': up_color, 'down': down_color},
                                            volume={'up': up_color, 'down': down_color},
                                            vcdopcod=True)
        style = mpf.make_mpf_style(base_mpf_style="default",
                                   marketcolors=marketcolor,
                                   facecolor='white',
                                   rc={'xtick.bottom': False, 'xtick.labelbottom': False,
                                       'ytick.left': False, 'ytick.labelleft': False,})
        fig, _ = mpf.plot(df[start:start+self.num_days], type='candle', style=style, volume=True, ylabel='', ylabel_lower='', returnfig=True, figsize=(2.24, 2.24), volume_exponent=0)
        canvas = FigureCanvas(fig)
        canvas.draw()
        
        buf = BytesIO()
        # fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        fig.savefig(buf, format='png', pad_inches=0)
        buf.seek(0)
        plt.close(fig)
        
        return self._trim(Image.open(buf).convert('RGB'))
    
    def _calc_yield(self, df, idx):
        yield_rate = (df.iloc[idx+self.num_days+1]['close'] - df.iloc[idx+self.num_days]['close']) / df.iloc[idx+self.num_days]['close'] * 100
        return yield_rate
    
    def __getitem__(self, idx):
        chart_imgs = list()
        for i in range(3):
            chart_img = self._generate_candle_chart(self.df, idx+i)
            if self.transform is not None:
                t = self.transform(chart_img)
            else:
                t = transforms.ToTensor()(chart_img)
            chart_imgs.append(t)
        chart_imgs = torch.stack(chart_imgs)
        yield_rate = self._calc_yield(self.df, idx)
        return chart_imgs, yield_rate